#include "srbd.hpp"

#include <proxsuite/proxqp/dense/dense.hpp>

namespace srbd {
    using qp_t = proxsuite::proxqp::dense::QP<double>;
    using qp_mat_t = proxsuite::proxqp::dense::Mat<double>;
    using qp_vec_t = proxsuite::proxqp::dense::Vec<double>;

    SingleRigidBodyDynamics::SingleRigidBodyDynamics(const SingleRigidBodyDynamics& other)
    {
        // TO-DO: This is not the best way to handle this!! BUT should be okay for now..
        operator=(other);
    }

    SingleRigidBodyDynamics& SingleRigidBodyDynamics::operator=(const SingleRigidBodyDynamics& other)
    {
        // COM state
        _base_position = other._base_position;
        _base_vel = other._base_vel;
        _base_orientation = other._base_orientation;
        _base_angular_vel = other._base_angular_vel;

        // Feet state
        _feet_positions = other._feet_positions;
        _feet_phases = other._feet_phases;

        // Static ref poses/bounds
        _feet_ref_positions = other._feet_ref_positions;
        _feet_min_bounds = other._feet_min_bounds;
        _feet_max_bounds = other._feet_max_bounds;

        // Phase Handler
        _phase_handler = std::move(other._phase_handler->clone());

        // Terrain
        _terrain = std::move(other._terrain->clone());

        // General state
        _mass = other._mass;
        _inertia = other._inertia;
        _inertia_inv = other._inertia_inv;

        // Global variables
        _dt = other._dt;
        _g = other._g;
        _gravity = other._gravity;

        // Helper variables
        _last_qp_result_valid = other._last_qp_result_valid;

        return *this;
    }

    void SingleRigidBodyDynamics::set_sim_data(double dt, double gravity)
    {
        _dt = dt;
        _g = std::abs(gravity);
        _gravity = Vec3d(0, 0, -_g);
    }

    void SingleRigidBodyDynamics::set_inertial_data(double mass, const RotMat& inertia)
    {
        _mass = mass;
        _inertia = inertia;
        _inertia_inv = inertia.inverse();
    }

    void SingleRigidBodyDynamics::set_feet_data(const std::vector<Vec3d>& feet_ref_positions, const std::vector<Vec3d>& feet_min_bounds, const std::vector<Vec3d>& feet_max_bounds)
    {
        assert((_feet_ref_positions.size() == _feet_min_bounds.size() == _feet_max_bounds.size()) && "Not consistent number of feet!");
        _feet_ref_positions = feet_ref_positions;
        _feet_min_bounds = feet_min_bounds;
        _feet_max_bounds = feet_max_bounds;
    }

    void SingleRigidBodyDynamics::set_data(const Vec3d& base_position, const Vec3d& base_velocity, const RotMat& base_orientation, const Vec3d& base_angular_velocity, const std::vector<Vec3d>& feet_positions, const std::vector<size_t>& feet_phases)
    {
        // Base-related
        _base_position = base_position;
        _base_vel = base_velocity;
        _base_orientation = base_orientation;
        _base_angular_vel = base_angular_velocity;

        // Feet-related
        assert(feet_positions.size() == _feet_ref_positions.size() && feet_phases.size() == _feet_ref_positions.size() && "Wrong number of feet!");
        _feet_positions = feet_positions;
        _feet_phases = feet_phases;
    }

    void SingleRigidBodyDynamics::set_base_position(const Vec3d& pos) { _base_position = pos; }

    void SingleRigidBodyDynamics::set_base_velocity(const Vec3d& vel) { _base_vel = vel; }

    void SingleRigidBodyDynamics::set_base_orientation(const RotMat& rot) { _base_orientation = rot; }

    void SingleRigidBodyDynamics::set_base_angular_velocity(const Vec3d& ang_vel) { _base_angular_vel = ang_vel; }

    void SingleRigidBodyDynamics::set_feet_positions(const std::vector<Vec3d>& feet_positions)
    {
        assert(feet_positions.size() == _feet_ref_positions.size() && "Wrong number of feet!");
        _feet_positions = feet_positions;
    }

    void SingleRigidBodyDynamics::set_feet_phases(const std::vector<size_t>& feet_phases)
    {
        assert(feet_phases.size() == _feet_ref_positions.size() && "Wrong number of feet!");
        _feet_phases = feet_phases;
    }

    std::tuple<Matrix, Matrix, Vector> SingleRigidBodyDynamics::linearize() const
    {
        const size_t n_dim = 9;
        const size_t n_feet = _feet_ref_positions.size();
        const size_t m_dim = n_feet * 3;

        Matrix L_c = skew(_base_angular_vel) * (_inertia - skew(_inertia * _base_angular_vel)); // optimal gain
        // State Space Model X_n+1 = A*X_n + B*U_n
        // Define A Matrix
        Matrix A = Matrix::Zero(n_dim, n_dim);
        // Block of size (p,q), starting at (i,j)	matri_block(i,j,p,q); matri_block<p,q>(i,j);
        A.block(0, 0, 3, 3) = Matrix::Identity(3, 3);
        A.block(0, 3, 3, 3) = Matrix::Identity(3, 3) * _dt;
        A.block(3, 3, 3, 3) = Matrix::Identity(3, 3);
        A.block(6, 6, 3, 3) = Matrix::Identity(3, 3) - _dt * _inertia_inv * L_c;

        // Define B Matrix
        Matrix B = Matrix::Zero(n_dim, m_dim);
        B.block(0, 0, 3, 3 * n_feet) = kron((((0.5 * _dt * _dt) / _mass) * Matrix::Ones(1, n_feet)), Matrix::Identity(3, 3));
        B.block(3, 0, 3, 3 * n_feet) = kron(((_dt / _mass) * Matrix::Ones(1, n_feet)), Matrix::Identity(3, 3));
        // horizontal stack of feet position (skew)
        Matrix r_dtstack = Matrix::Zero(3, 3 * n_feet);
        for (size_t i = 0; i < n_feet; i++) {
            r_dtstack.block(0, 3 * i, 3, 3) = skew(_feet_positions[i] - _base_position);
        }
        B.block(6, 0, 3, 3 * n_feet) = _dt * _inertia_inv * _base_orientation.transpose() * r_dtstack;

        // Define d Matrix
        Vector d = Vector::Zero(n_dim);
        d.segment(0, 3) = 0.5 * _dt * _dt * _gravity;
        d.segment(3, 3) = _dt * _gravity;
        d.segment(6, 3) = _dt * _inertia_inv * (L_c - skew(_base_angular_vel) * _inertia) * _base_angular_vel;

        return std::make_tuple(A, B, d);
    }

    Vector SingleRigidBodyDynamics::state_to_mpc_vec() const
    {
        Vector state(9);
        state << _base_position, _base_vel, _base_angular_vel;
        return state;
    }

    Vector SingleRigidBodyDynamics::state_to_vec() const
    {
        Vector state(12);
        state << _base_position, _base_vel, rotation_matrix_to_euler_zyx(_base_orientation), _base_angular_vel;
        return state;
    }

    std::pair<Matrix, Vec6d> SingleRigidBodyDynamics::inv_mass_matrix() const
    {
        const size_t n_feet = _feet_ref_positions.size();
        Matrix M = Matrix::Zero(6, n_feet * 3);
        Vec6d v;

        for (size_t i = 0; i < n_feet; i++) {
            // Linear part
            M.block(0, i * 3, 3, 3) = Matrix::Identity(3, 3) / _mass;
            // Angular part
            M.block(3, i * 3, 3, 3) = _inertia_inv * _base_orientation.transpose() * skew(_feet_positions.at(i) - _base_position);
        }

        v.head(3) = _gravity;
        v.tail(3) = -_inertia_inv * skew(_base_angular_vel) * (_inertia * _base_angular_vel);

        return {M, v};
    }

    void SingleRigidBodyDynamics::integrate(const std::vector<Vec3d>& feet_forces, const Vec3d& external_force)
    {
        const size_t n_feet = _feet_ref_positions.size();
        Matrix M;
        Vec6d v;
        std::tie(M, v) = inv_mass_matrix();

        Vector F = Eigen::Map<Vector>(const_cast<double*>(&feet_forces[0][0]), static_cast<int>(3 * n_feet));

        // If the foot is in swing phase, it doesn't affect the body
        //   Let's zero out its' force contribution
        for (size_t i = 0; i < n_feet; i++) {
            if (_phase_handler->is_swing(_feet_phases[i])) {
                F.segment(i * 3, 3).setZero();
            }
        }

        Vec6d acc = (M * F) + v;
        // Add external forces
        acc.head(3) += external_force / _mass;

        // // air damping
        // double b = 1.;
        // acc.head(3) -= b * _base_vel;
        // acc.tail(3) -= b * _base_angular_vel;

        _integrate(acc);
    }

    std::vector<Vec3d> SingleRigidBodyDynamics::integrate(const Vec6d& a_desired, const Vec3d& external_force, bool use_external_force_in_qp)
    {
        const size_t n_feet = _feet_ref_positions.size();
        const size_t n_cons = n_feet * 5;

        Matrix M;
        Vec6d v;
        std::tie(M, v) = inv_mass_matrix();

        // Add external forces to QP problem, if asked
        if (use_external_force_in_qp)
            v.head(3) += external_force / _mass;

        // TO-DO: Maybe make the fz==0 an equality for performance
        qp_t qp = qp_t(n_feet * 3, 0, n_cons);
        qp.settings.eps_abs = 1e-3;
        qp.settings.max_iter = 100;
        qp.settings.max_iter_in = 100;
        qp.settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
        // Solve QP
        {
            qp_mat_t H = M.transpose() * M;
            qp_vec_t g = M.transpose() * (v - a_desired);

            qp_mat_t C = qp_mat_t::Zero(n_cons, n_feet * 3);
            qp_vec_t l = qp_vec_t::Zero(n_cons);
            qp_vec_t u = qp_vec_t::Zero(n_cons);
            for (size_t k = 0; k < n_feet; k++) {
                // force/friction coeff constraints
                double friction_coeff = _terrain->friction_coeff(_feet_positions.at(k)[0], _feet_positions.at(k)[1]);
                Vec3d n, t1, t2;
                std::tie(n, t1, t2) = _terrain->normals(_feet_positions.at(k)[0], _feet_positions.at(k)[1]);
                C.block(k * 5, k * 3, 1, 3) << t1[0] + n[0] * friction_coeff, t1[1] + n[1] * friction_coeff, t1[2] + n[2] * friction_coeff;
                C.block(k * 5 + 1, k * 3, 1, 3) << t1[0] - n[0] * friction_coeff, t1[1] - n[1] * friction_coeff, t1[2] - n[2] * friction_coeff;
                C.block(k * 5 + 2, k * 3, 1, 3) << t2[0] + n[0] * friction_coeff, t2[1] + n[1] * friction_coeff, t2[2] + n[2] * friction_coeff;
                C.block(k * 5 + 3, k * 3, 1, 3) << t2[0] - n[0] * friction_coeff, t2[1] - n[1] * friction_coeff, t2[2] - n[2] * friction_coeff;
                C.block(k * 5 + 4, k * 3, 1, 3) << n[0], n[1], n[2];

                if (_phase_handler->is_swing(_feet_phases[k])) { // swing phase, z component should be zero
                    l(k * 5 + 4) = 0.;
                    u(k * 5 + 4) = 0.;
                }
                else {
                    l(k * 5 + 4) = 0.1 * _mass * _g / static_cast<double>(n_feet); // z-component should always be positive (and if in contact, it should produce at least a small force)
                    u(k * 5 + 4) = 5. * _mass * _g; // max force
                }

                // fx >= -mu*fz
                l(k * 5) = 0.;
                u(k * 5) = 1e20;
                // fx <= mu*fz
                l(k * 5 + 1) = -1e20;
                u(k * 5 + 1) = 0.;
                // fy >= -mu*fz
                l(k * 5 + 2) = 0.;
                u(k * 5 + 2) = 1e20;
                // fy <= mu*fz
                l(k * 5 + 3) = -1e20;
                u(k * 5 + 3) = 0.;
            }

            qp.init(H, g, std::nullopt, std::nullopt, C, l, u);
            qp.solve();

            if (qp.results.info.status != proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED)
                _last_qp_result_valid = false;
            else
                _last_qp_result_valid = true;
        }

        // Copy results to forces
        std::vector<Vec3d> feet_forces(n_feet, Vec3d::Zero());
        for (size_t k = 0; k < n_feet; k++) {
            feet_forces.at(k) = qp.results.x.segment(k * 3, 3);
            if (_phase_handler->is_swing(_feet_phases[k]))
                feet_forces.at(k).setZero();
        }

        // Integrate
        Vector F = Eigen::Map<Vector>(const_cast<double*>(&feet_forces[0][0]), static_cast<int>(3 * n_feet));

        Vec6d acc = (M * F) + v;
        // Add external forces for actual simulation
        acc.head(3) += external_force / _mass;

        // // air damping
        // double b = 1.;
        // acc.head(3) -= b * _base_vel;
        // acc.tail(3) -= b * _base_angular_vel;

        _integrate(acc);

        return feet_forces;
    }

    bool SingleRigidBodyDynamics::valid(bool strict) const
    {
        bool finite = is_finite(_base_position) && is_finite(_base_vel) && is_finite(_base_orientation) && is_finite(_base_angular_vel);
        if (!finite || !in_bounds())
            return false;

        return true && (!strict || _last_qp_result_valid);
    }

    bool SingleRigidBodyDynamics::in_bounds() const
    {
        const size_t n_feet = _feet_ref_positions.size();
        for (size_t k = 0; k < n_feet; k++) {
            if (!_phase_handler->is_swing(_feet_phases[k])) {
                Vec3d p = _base_orientation.transpose() * (_feet_positions[k] - _base_position);
                for (size_t idx = 0; idx < 3; idx++) {
                    if (_feet_min_bounds[k][idx] > p[idx] || _feet_max_bounds[k][idx] < p[idx]) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    bool SingleRigidBodyDynamics::fix_invalid(unsigned int max_attempts)
    {
        const size_t n_feet = _feet_ref_positions.size();

        for (unsigned int i = 0; i < max_attempts; i++) {
            if (in_bounds())
                return true;

            for (size_t k = 0; k < n_feet; k++) {
                if (!_phase_handler->is_swing(_feet_phases[k])) {
                    Vec3d p = _base_orientation.transpose() * (_feet_positions[k] - _base_position);
                    for (size_t idx = 0; idx < 3; idx++) {
                        if (_feet_min_bounds[k][idx] > p[idx]) {
                            p[idx] = _feet_min_bounds[k][idx];
                        }
                        else if (_feet_max_bounds[k][idx] < p[idx]) {
                            p[idx] = _feet_max_bounds[k][idx];
                        }
                    }

                    _base_position = _feet_positions[k] - _base_orientation * p;
                }
            }
        }

        return false;
    }

    Eigen::Isometry3d SingleRigidBodyDynamics::base_tf() const
    {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() = _base_orientation;
        T.translation() = _base_position;
        return T;
    }

    Eigen::Isometry3d SingleRigidBodyDynamics::foot_tf(size_t foot_idx) const
    {
        assert(foot_idx < _feet_ref_positions.size());

        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() = _feet_positions[foot_idx];
        return T;
    }

    SingleRigidBodyDynamics SingleRigidBodyDynamics::create_robot(RobotType type)
    {
        const double bx = 0.5, by = 0.4, bz = 0.3;
        const double mass = 1.;
        const double initial_height = 0.5;
        std::vector<Vec3d> feet_ref_positions;
        std::vector<Vec3d> feet_min_bounds;
        std::vector<Vec3d> feet_max_bounds;
        size_t T = 40;
        size_t T_swing = 20;

        auto make_monoped = [&]() {
            feet_ref_positions.clear();
            feet_min_bounds.clear();
            feet_max_bounds.clear();

            feet_ref_positions.push_back(Vec3d(0., 0., -initial_height));
            feet_min_bounds.push_back(feet_ref_positions.back() + Vec3d(-0.4, -0.2, -0.2));
            feet_max_bounds.push_back(feet_ref_positions.back() + Vec3d(0.4, 0.2, 0.2));
            auto robot = SingleRigidBodyDynamics(mass, inertia_tensor(mass * (by * by + bz * bz) / 12., mass * (bx * bx + bz * bz) / 12., mass * (bx * bx + by * by) / 12., 0., 0., 0.), feet_ref_positions, feet_min_bounds, feet_max_bounds, T, T_swing);

            robot.set_base_position(Vec3d(0., 0., initial_height));
            robot.set_feet_positions({Vec3d::Zero()});
            robot.set_feet_phases({T_swing});
            return robot;
        };

        auto make_biped = [&]() {
            feet_ref_positions.clear();
            feet_min_bounds.clear();
            feet_max_bounds.clear();

            feet_ref_positions.push_back(Vec3d(bx / 2., 0., -initial_height));
            feet_min_bounds.push_back(feet_ref_positions.back() + Vec3d(-0.4, -0.2, -0.2));
            feet_max_bounds.push_back(feet_ref_positions.back() + Vec3d(0.4, 0.2, 0.2));
            feet_ref_positions.push_back(Vec3d(-bx / 2., 0., -initial_height));
            feet_min_bounds.push_back(feet_ref_positions.back() + Vec3d(-0.4, -0.2, -0.2));
            feet_max_bounds.push_back(feet_ref_positions.back() + Vec3d(0.4, 0.2, 0.2));
            auto robot = SingleRigidBodyDynamics(mass, inertia_tensor(mass * (by * by + bz * bz) / 12., mass * (bx * bx + bz * bz) / 12., mass * (bx * bx + by * by) / 12., 0., 0., 0.), feet_ref_positions, feet_min_bounds, feet_max_bounds, T, T_swing);

            robot.set_base_position(Vec3d(0., 0., initial_height));
            robot.set_feet_positions({Vec3d(bx / 2., 0., 0.), Vec3d(-bx / 2., 0., 0.)});
            robot.set_feet_phases({0, T_swing});
            return robot;
        };

        auto make_quadruped = [&]() {
            feet_ref_positions.clear();
            feet_min_bounds.clear();
            feet_max_bounds.clear();

            feet_ref_positions.push_back(Vec3d(bx / 2., by / 2., -initial_height));
            feet_min_bounds.push_back(feet_ref_positions.back() + Vec3d(-0.4, -0.2, -0.2));
            feet_max_bounds.push_back(feet_ref_positions.back() + Vec3d(0.4, 0.2, 0.2));
            feet_ref_positions.push_back(Vec3d(-bx / 2., by / 2., -initial_height));
            feet_min_bounds.push_back(feet_ref_positions.back() + Vec3d(-0.4, -0.2, -0.2));
            feet_max_bounds.push_back(feet_ref_positions.back() + Vec3d(0.4, 0.2, 0.2));
            feet_ref_positions.push_back(Vec3d(bx / 2., -by / 2., -initial_height));
            feet_min_bounds.push_back(feet_ref_positions.back() + Vec3d(-0.4, -0.2, -0.2));
            feet_max_bounds.push_back(feet_ref_positions.back() + Vec3d(0.4, 0.2, 0.2));
            feet_ref_positions.push_back(Vec3d(-bx / 2., -by / 2., -initial_height));
            feet_min_bounds.push_back(feet_ref_positions.back() + Vec3d(-0.4, -0.2, -0.2));
            feet_max_bounds.push_back(feet_ref_positions.back() + Vec3d(0.4, 0.2, 0.2));
            auto robot = SingleRigidBodyDynamics(mass, inertia_tensor(mass * (by * by + bz * bz) / 12., mass * (bx * bx + bz * bz) / 12., mass * (bx * bx + by * by) / 12., 0., 0., 0.), feet_ref_positions, feet_min_bounds, feet_max_bounds, T, T_swing);

            robot.set_base_position(Vec3d(0., 0., initial_height));
            robot.set_feet_positions({Vec3d(bx / 2., by / 2., 0.), Vec3d(-bx / 2., by / 2., 0.), Vec3d(bx / 2., -by / 2., 0.), Vec3d(-bx / 2., -by / 2., 0.)});
            robot.set_feet_phases({T_swing, 0, 0, T_swing});
            return robot;
        };

        auto make_hexapod = [&]() {
            feet_ref_positions.clear();
            feet_min_bounds.clear();
            feet_max_bounds.clear();

            feet_ref_positions.push_back(Vec3d(bx / 2., by / 2., -initial_height));
            feet_min_bounds.push_back(feet_ref_positions.back() + Vec3d(-0.4, -0.2, -0.2));
            feet_max_bounds.push_back(feet_ref_positions.back() + Vec3d(0.4, 0.2, 0.2));
            feet_ref_positions.push_back(Vec3d(-bx / 2., by / 2., -initial_height));
            feet_min_bounds.push_back(feet_ref_positions.back() + Vec3d(-0.4, -0.2, -0.2));
            feet_max_bounds.push_back(feet_ref_positions.back() + Vec3d(0.4, 0.2, 0.2));
            feet_ref_positions.push_back(Vec3d(bx / 2., -by / 2., -initial_height));
            feet_min_bounds.push_back(feet_ref_positions.back() + Vec3d(-0.4, -0.2, -0.2));
            feet_max_bounds.push_back(feet_ref_positions.back() + Vec3d(0.4, 0.2, 0.2));
            feet_ref_positions.push_back(Vec3d(-bx / 2., -by / 2., -initial_height));
            feet_min_bounds.push_back(feet_ref_positions.back() + Vec3d(-0.4, -0.2, -0.2));
            feet_max_bounds.push_back(feet_ref_positions.back() + Vec3d(0.4, 0.2, 0.2));
            feet_ref_positions.push_back(Vec3d(0., by / 2., -initial_height));
            feet_min_bounds.push_back(feet_ref_positions.back() + Vec3d(-0.4, -0.2, -0.2));
            feet_max_bounds.push_back(feet_ref_positions.back() + Vec3d(0.4, 0.2, 0.2));
            feet_ref_positions.push_back(Vec3d(0., -by / 2., -initial_height));
            feet_min_bounds.push_back(feet_ref_positions.back() + Vec3d(-0.4, -0.2, -0.2));
            feet_max_bounds.push_back(feet_ref_positions.back() + Vec3d(0.4, 0.2, 0.2));
            auto robot = SingleRigidBodyDynamics(mass, inertia_tensor(mass * (by * by + bz * bz) / 12., mass * (bx * bx + bz * bz) / 12., mass * (bx * bx + by * by) / 12., 0., 0., 0.), feet_ref_positions, feet_min_bounds, feet_max_bounds, T, T_swing);

            robot.set_base_position(Vec3d(0., 0., initial_height));
            robot.set_feet_positions({Vec3d(bx / 2., by / 2., 0.), Vec3d(-bx / 2., by / 2., 0.), Vec3d(bx / 2., -by / 2., 0.), Vec3d(-bx / 2., -by / 2., 0.), Vec3d(0., by / 2., 0.), Vec3d(0., -by / 2., 0.)});
            robot.set_feet_phases({T_swing, 0, 0, T_swing, T_swing, 0});
            return robot;
        };

        auto make_anymal = [&]() {
            feet_ref_positions.clear();
            feet_min_bounds.clear();
            feet_max_bounds.clear();

            // This from towr
            // RotMat inertia = inertia_tensor(0.946438, 1.94478, 2.01835, 0.000938112, -0.00595386, -0.00146328);
            // const double anymal_mass = 29.5;
            // This is from our URDF
            RotMat inertia = inertia_tensor(0.88201174, 1.85452968, 1.97309185, 0.00137526, 0.00062895, 0.00018922);
            const double anymal_mass = 30.4213964625;
            const double x_nominal_b = 0.34;
            const double y_nominal_b = 0.19;
            const double z_nominal_b = -0.42;
            const double dx = 0.15;
            const double dy = 0.1;
            const double dz = 0.1;

            std::vector<Vec3d> feet_positions;

            // NOTE: Left forward
            feet_positions.push_back(Vec3d(x_nominal_b, y_nominal_b, 0.));
            feet_min_bounds.push_back(Vec3d(x_nominal_b - dx, y_nominal_b - dy, z_nominal_b - dz));
            feet_max_bounds.push_back(Vec3d(x_nominal_b + dx, y_nominal_b + dy, z_nominal_b + dz));
            feet_ref_positions.push_back(Vec3d(x_nominal_b, y_nominal_b, z_nominal_b));

            // NOTE: Left back
            feet_positions.push_back(Vec3d(-x_nominal_b, y_nominal_b, 0.));
            feet_min_bounds.push_back(Vec3d(-x_nominal_b - dx, y_nominal_b - dy, z_nominal_b - dz));
            feet_max_bounds.push_back(Vec3d(-x_nominal_b + dx, y_nominal_b + dy, z_nominal_b + dz));
            feet_ref_positions.push_back(Vec3d(-x_nominal_b, y_nominal_b, z_nominal_b));

            // NOTE: Right forward
            feet_positions.push_back(Vec3d(x_nominal_b, -y_nominal_b, 0.));
            feet_min_bounds.push_back(Vec3d(x_nominal_b - dx, -y_nominal_b - dy, z_nominal_b - dz));
            feet_max_bounds.push_back(Vec3d(x_nominal_b + dx, -y_nominal_b + dy, z_nominal_b + dz));
            feet_ref_positions.push_back(Vec3d(x_nominal_b, -y_nominal_b, z_nominal_b));

            // NOTE: Right back
            feet_positions.push_back(Vec3d(-x_nominal_b, -y_nominal_b, 0.));
            feet_min_bounds.push_back(Vec3d(-x_nominal_b - dx, -y_nominal_b - dy, z_nominal_b - dz));
            feet_max_bounds.push_back(Vec3d(-x_nominal_b + dx, -y_nominal_b + dy, z_nominal_b + dz));
            feet_ref_positions.push_back(Vec3d(-x_nominal_b, -y_nominal_b, z_nominal_b));

            auto robot = SingleRigidBodyDynamics(anymal_mass, inertia, feet_ref_positions, feet_min_bounds, feet_max_bounds, T, T_swing);

            robot.set_base_position(Vec3d(0., 0., std::abs(z_nominal_b)));
            robot.set_feet_positions(feet_positions);
            robot.set_feet_phases({T_swing, 0, 0, T_swing});
            return robot;
        };

        switch (type) {
        case RobotType::Monoped:
            return make_monoped();
        case RobotType::Biped:
            return make_biped();
        case RobotType::Quadruped:
            return make_quadruped();
        case RobotType::Hexapod:
            return make_hexapod();
        case RobotType::ANYmal:
            return make_anymal();
        }

        std::cerr << "Error: Invalid robot type, using default (Quadruped)" << std::endl;
        return make_quadruped();
    }

    void SingleRigidBodyDynamics::_integrate(const Vec6d& acc)
    {
        const size_t n_feet = _feet_ref_positions.size();

        // (Semi-implicit) Euler integration
        _base_vel += acc.head(3) * _dt;
        _base_position += _base_vel * _dt;

        _base_angular_vel += acc.tail(3) * _dt;
        _base_orientation *= exp_angular(_base_angular_vel * _dt);

        // Increase phase variables
        double k_foot = _phase_handler->kappa_foot(_dt);
        for (size_t i = 0; i < n_feet; i++) {
            _feet_phases[i]++;
            if (_phase_handler->is_phase_change(_feet_phases[i])) {
                _feet_positions[i].head(2) = _feet_ref_positions[i].head(2) + k_foot * (_base_orientation.transpose() * _base_vel).head(2);
                _feet_positions[i][2] = 0.;

                // _feet_positions[i][0] = std::max(_feet_min_bounds[i][0], std::min(_feet_max_bounds[i][0], _feet_positions[i][0]));
                // _feet_positions[i][1] = std::max(_feet_min_bounds[i][1], std::min(_feet_max_bounds[i][1], _feet_positions[i][1]));
                // _feet_positions[i][2] = std::max(_feet_min_bounds[i][2], std::min(_feet_max_bounds[i][2], _feet_positions[i][2]));

                _feet_positions[i] = _base_position + _base_orientation * _feet_positions[i];
                _feet_positions[i][2] = _terrain->height(_feet_positions[i][0], _feet_positions[i][1]); // get height from terrain
            }
        }
    }
} // namespace srbd
