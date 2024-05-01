#ifndef SRBD_SRBD_HPP
#define SRBD_SRBD_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <memory>

#include <srbd/utils.hpp>

namespace srbd {
    enum RobotType {
        Monoped = 0,
        Biped,
        Quadruped,
        Hexapod,
        ANYmal
    };

    // Phase Handlers
    struct BasePhaseHandler {
        virtual bool is_swing(size_t phase_counter) const = 0;
        virtual bool is_phase_change(size_t phase_counter) const = 0;
        virtual double kappa_foot(double dt) const = 0;

        virtual std::unique_ptr<BasePhaseHandler> clone() const = 0;
    };

    struct DefaultPhaseHandler : public BasePhaseHandler {
        size_t T;
        size_t T_swing;

        DefaultPhaseHandler(size_t T_, size_t T_swing_) : T(T_), T_swing(T_swing_) {}

        bool is_swing(size_t phase_counter) const override
        {
            return ((phase_counter % T) < T_swing);
        }

        bool is_phase_change(size_t phase_counter) const override
        {
            return ((phase_counter % T) == 0) && T_swing > 0;
        }

        double kappa_foot(double dt) const override
        {
            return 2. * (T - T_swing) * dt;
        }

        std::unique_ptr<BasePhaseHandler> clone() const override
        {
            return std::make_unique<DefaultPhaseHandler>(T, T_swing);
        }
    };

    // Terrain
    struct BaseTerrain {
        virtual double height(double, double) const = 0;
        virtual double friction_coeff(double, double) const = 0;
        virtual std::tuple<Vec3d, Vec3d, Vec3d> normals(double, double) const = 0;

        virtual std::unique_ptr<BaseTerrain> clone() const = 0;
    };

    struct FlatTerrain : public BaseTerrain {
        double z = 0.;

        FlatTerrain(double h = 0.) : z(h) {}

        // Terrain function
        double height(double /* x */, double /* y */) const override
        {
            return z;
        }

        double friction_coeff(double /* x */, double /* y */) const override
        {
            return 1.; // friction same everywhere
        }

        std::tuple<Vec3d, Vec3d, Vec3d> normals(double /* x */, double /* y */) const override
        {
            // flat terrain
            Vec3d n;
            n << 0., 0., 1.;
            Vec3d t1;
            t1 << 1., 0., 0.;
            Vec3d t2;
            t2 << 0., 1., 0.;

            return {n, t1, t2};
        }

        std::unique_ptr<BaseTerrain> clone() const override
        {
            return std::make_unique<FlatTerrain>(z);
        }
    };

    struct CubicHeightMapTerrain : public BaseTerrain {
        double a1 = 0., b1 = 0., c1 = 0., d1 = 0.;
        double a2 = 0., b2 = 0., c2 = 0., d2 = 0.;

        CubicHeightMapTerrain(double xa, double xb, double xc, double xd, double ya, double yb, double yc, double yd) : a1(xa), b1(xb), c1(xc), d1(xd), a2(ya), b2(yb), c2(yc), d2(yd) {}

        // Terrain function
        double height(double x, double y) const override
        {
            double x_sq = x * x;
            double zx = a1 * x * x_sq + b1 * x_sq + c1 * x + d1;
            double y_sq = y * y;
            double zy = a2 * y * y_sq + b2 * y_sq + c2 * y + d2;

            return zx * zy;
        }

        double friction_coeff(double /* x */, double /* y */) const override
        {
            return 1.; // friction same everywhere
        }

        std::tuple<Vec3d, Vec3d, Vec3d> normals(double x, double y) const override
        {
            // flat terrain
            Vec3d n;
            n << -deriv_x(x, y), -deriv_y(x, y), 1.;
            n.normalize();
            Vec3d t1;
            t1 << 1., 0., deriv_x(x, y);
            t1.normalize();
            Vec3d t2;
            t2 << 0., 1., deriv_y(x, y);
            t2.normalize();

            return {n, t1, t2};
        }

        std::unique_ptr<BaseTerrain> clone() const override
        {
            return std::make_unique<CubicHeightMapTerrain>(a1, b1, c1, d1, a2, b2, c2, d2);
        }

        double deriv_x(double x, double y) const
        {
            double x_sq = x * x;
            double dx = 3. * a1 * x_sq + 2. * b1 * x + c1;
            double y_sq = y * y;
            double zy = a2 * y * y_sq + b2 * y_sq + c2 * y + d2;
            return dx * zy;
        }

        double deriv_y(double x, double y) const
        {
            double y_sq = y * y;
            double dy = 3. * a2 * y_sq + 2. * b2 * y + c2;
            double x_sq = x * x;
            double zx = a1 * x * x_sq + b1 * x_sq + c1 * x + d1;
            return dy * zx;
        }
    };

    class SingleRigidBodyDynamics {
    public:
        // constructors
        SingleRigidBodyDynamics() {}

        template <typename... Args>
        SingleRigidBodyDynamics(double mass, const RotMat& inertia, const std::vector<Vec3d>& feet_ref_positions, const std::vector<Vec3d>& feet_min_bounds, const std::vector<Vec3d>& feet_max_bounds, Args... args)
        {
            // Body-related
            set_inertial_data(mass, inertia);

            // Feet-related
            set_feet_data(feet_ref_positions, feet_min_bounds, feet_max_bounds);

            // Initialize with DefaultPhaseHandler
            _phase_handler.reset(new DefaultPhaseHandler{args...});

            // Initialize with flat terrain at zero height
            _terrain.reset(new FlatTerrain{0.});

            // Zero-out entities
            _base_position.setZero();
            _base_vel.setZero();
            _base_orientation = RotMat::Identity();
            _base_angular_vel.setZero();

            const size_t n_feet = _feet_ref_positions.size();
            _feet_positions.resize(n_feet, Vec3d::Zero());
            _feet_phases.resize(n_feet, 0);
        }

        // copy constructor
        SingleRigidBodyDynamics(const SingleRigidBodyDynamics& other);

        // Assignment operator
        SingleRigidBodyDynamics& operator=(const SingleRigidBodyDynamics& other);

        // Initialization-related
        // Fixed quantities
        void set_sim_data(double dt, double gravity = 9.81);
        void set_inertial_data(double mass, const RotMat& inertia);
        void set_feet_data(const std::vector<Vec3d>& feet_ref_positions, const std::vector<Vec3d>& feet_min_bounds, const std::vector<Vec3d>& feet_max_bounds);

        // Changing quantities
        void set_data(const Vec3d& base_position, const Vec3d& base_velocity, const RotMat& base_orientation, const Vec3d& base_angular_velocity, const std::vector<Vec3d>& feet_positions, const std::vector<size_t>& feet_phases);

        void set_base_position(const Vec3d& pos);
        void set_base_velocity(const Vec3d& vel);
        void set_base_orientation(const RotMat& rot);
        void set_base_angular_velocity(const Vec3d& ang_vel);
        void set_feet_positions(const std::vector<Vec3d>& feet_positions);
        void set_feet_phases(const std::vector<size_t>& feet_phases);

        // PhaseHandler
        template <typename T, typename... Args>
        void set_phase_handler(Args... args)
        {
            _phase_handler.reset(new T{args...});
        }

        // Terrain
        template <typename T, typename... Args>
        void set_terrain(Args... args)
        {
            _terrain.reset(new T{args...});
        }

        /// MPC-related
        // linearization function #1 [x' = A*x + B*f + d] (useful for MPC)
        std::tuple<Matrix, Matrix, Vector> linearize() const;
        // converting state to vector <9,1> (useful for MPC)
        Vector state_to_mpc_vec() const;
        // converting state to "full" vector <12,1> (useful for RL/Learning)
        Vector state_to_vec() const;

        /// ForwardSim-related
        // linearization function #2 [M*f + v] gives produced acceleration (useful for forward sim)
        std::pair<Matrix, Vec6d> inv_mass_matrix() const;

        // integrating using non-linear dynamics
        void integrate(const std::vector<Vec3d>& feet_forces, const Vec3d& external_force = Vec3d(0., 0., 0.));
        // integrating using desired COM acceleration (this solves a QP)
        std::vector<Vec3d> integrate(const Vec6d& a_desired, const Vec3d& external_force = Vec3d(0., 0., 0.), bool use_external_force_in_qp = false);

        // check if state is valid (finite and within bounds for stance phases)
        bool valid(bool strict = false) const;
        // check if state is within bounds for stance phases
        bool in_bounds() const;
        // Attempt to fix base pose for an invalid state
        bool fix_invalid(unsigned int max_attempts = 10);

        // getters
        const Vec3d& base_position() const { return _base_position; }
        const Vec3d& base_velocity() const { return _base_vel; }
        const RotMat& base_orientation() const { return _base_orientation; }
        const Vec3d& base_angular_velocity() const { return _base_angular_vel; }
        const std::vector<Vec3d>& feet_positions() const { return _feet_positions; }
        const std::vector<Vec3d>& feet_min_bounds() const { return _feet_min_bounds; }
        const std::vector<Vec3d>& feet_max_bounds() const { return _feet_max_bounds; }
        const std::vector<Vec3d>& feet_ref_positions() const { return _feet_ref_positions; }
        const std::vector<size_t>& feet_phases() const { return _feet_phases; }

        size_t n_feet() const { return _feet_ref_positions.size(); }

        double mass() const { return _mass; }
        const RotMat& inertia() const { return _inertia; }
        const RotMat& inertia_inv() const { return _inertia_inv; }

        double timestep() const { return _dt; }
        double gravity() const { return _g; }
        const Vec3d& gravity_vec() const { return _gravity; }

        const BasePhaseHandler& phase_handler() const { return *_phase_handler; }
        const BaseTerrain& terrain() const { return *_terrain; }

        // get center of mass transformation matrix in world frame
        Eigen::Isometry3d base_tf() const;
        // get foot transformation matrix in world frame
        Eigen::Isometry3d foot_tf(size_t foot_idx) const;

        static SingleRigidBodyDynamics create_robot(RobotType type);

    protected:
        // COM state
        Vec3d _base_position;
        Vec3d _base_vel;
        RotMat _base_orientation;
        Vec3d _base_angular_vel; // this is in local (COM) frame

        // Feet state
        std::vector<Vec3d> _feet_positions; // in world frame, these are more of targets rather than states
        std::vector<size_t> _feet_phases;

        // Static ref poses/bounds
        std::vector<Vec3d> _feet_ref_positions; // in COM frame
        std::vector<Vec3d> _feet_min_bounds;
        std::vector<Vec3d> _feet_max_bounds;

        // Phase Handler
        std::unique_ptr<BasePhaseHandler> _phase_handler;

        // Terrain
        std::unique_ptr<BaseTerrain> _terrain;

        // General state
        double _mass; // mass
        RotMat _inertia; // inertia matrix
        RotMat _inertia_inv; // inverse of inertia matrix

        // Global variables
        double _dt = 0.01; // dt
        double _g = 9.81; // gravity
        Vec3d _gravity = Vec3d(0, 0, -_g); // gravity vector

        // Helper variables
        bool _last_qp_result_valid = true;

        // Protected methods
        void _integrate(const Vec6d& acc);
    };
} // namespace srbd

#endif