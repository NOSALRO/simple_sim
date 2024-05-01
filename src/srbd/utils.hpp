#ifndef SRBD_UTILS_HPP
#define SRBD_UTILS_HPP

#include <Eigen/Core>

namespace srbd {
    using Vec3d = Eigen::Matrix<double, 3, 1>;
    using Vec6d = Eigen::Matrix<double, 6, 1>;
    using RotMat = Eigen::Matrix<double, 3, 3>;
    using Matrix = Eigen::Matrix<double, -1, -1>;
    using Vector = Eigen::Matrix<double, -1, 1>;

    //  Convert to inertia tensor from 6d values
    inline RotMat inertia_tensor(double Ixx, double Iyy, double Izz, double Ixy, double Ixz, double Iyz)
    {
        RotMat I;
        I << Ixx, -Ixy, -Ixz,
            -Ixy, Iyy, -Iyz,
            -Ixz, -Iyz, Izz;
        return I;
    }

    // function that computes the kronecker product of two vectors
    inline Matrix kron(const Matrix& vec1, const Matrix& vec2)
    {
        Matrix out = Matrix::Zero(vec1.rows() * vec2.rows(), vec1.cols() * vec2.cols());
        // i loops till rows vec1
        for (int i = 0; i < vec1.rows(); i++) {
            // j loops till rows vec2
            for (int j = 0; j < vec2.rows(); j++) {
                // k loops till cols vec1
                for (int k = 0; k < vec1.cols(); k++) {
                    // l loops till cols vec2
                    for (int l = 0; l < vec2.cols(); l++) {
                        out(i * vec2.rows() + j, k * vec2.cols() + l) = vec1(i, k) * vec2(j, l);
                    }
                }
            }
        }
        return out;
    }

    // create skew symmetric matrix ahat from a 3x1 vector a, such that ahat*b = a x b
    inline RotMat skew(const Vec3d& vec)
    {
        RotMat skew;
        skew << 0., -vec(2), vec(1),
            vec(2), 0., -vec(0),
            -vec(1), vec(0), 0.;
        return skew;
    }

    // Exponential mapping
    // I + sin(t) / t*[S] + (1 - cos(t)) / t^2*[S]^2, where t = |S|
    inline RotMat exp_angular(const Vec3d& vec, double epsilon = 1e-12)
    {
        RotMat ret;
        ret.setIdentity();

        double s2[] = {vec[0] * vec[0], vec[1] * vec[1], vec[2] * vec[2]};
        double s3[] = {vec[0] * vec[1], vec[1] * vec[2], vec[2] * vec[0]};
        double theta = std::sqrt(s2[0] + s2[1] + s2[2]);
        double cos_t = std::cos(theta);
        double alpha = 0.0;
        double beta = 0.0;

        if (theta > epsilon) {
            alpha = std::sin(theta) / theta;
            beta = (1.0 - cos_t) / theta / theta;
        }
        else {
            alpha = 1.0 - theta * theta / 6.0;
            beta = 0.5 - theta * theta / 24.0;
        }

        ret(0, 0) = beta * s2[0] + cos_t;
        ret(1, 0) = beta * s3[0] + alpha * vec[2];
        ret(2, 0) = beta * s3[2] - alpha * vec[1];

        ret(0, 1) = beta * s3[0] - alpha * vec[2];
        ret(1, 1) = beta * s2[1] + cos_t;
        ret(2, 1) = beta * s3[1] + alpha * vec[0];

        ret(0, 2) = beta * s3[2] + alpha * vec[1];
        ret(1, 2) = beta * s3[1] - alpha * vec[0];
        ret(2, 2) = beta * s2[2] + cos_t;

        return ret;
    }

    inline Vec3d log_map(const RotMat& _R)
    {
        Eigen::AngleAxisd aa(_R);
        return aa.angle() * aa.axis();
    }

    // https://www.geometrictools.com/Documentation/EulerAngles.pdf
    inline Vec3d rotation_matrix_to_euler_xyz(const RotMat& R)
    {
        const int X = 0;
        const int Y = 1;
        const int Z = 2;
        Vec3d euler;
        double R02 = R(0, 2);
        if (R02 < 1) {
            if (R02 > -1) {
                euler[Y] = std::asin(R02);
                euler[X] = std::atan2(-R(1, 2), R(2, 2));
                euler[Z] = std::atan2(-R(0, 1), R(0, 0));
            }
            else { // Not a unique solution, we choose one
                euler[Y] = -M_PI / 2.;
                euler[X] = -std::atan2(-R(1, 0), R(1, 1));
                euler[Z] = 0.;
            }
        }
        else { // Not a unique solution, we choose one
            euler[Y] = M_PI / 2.;
            euler[X] = std::atan2(-R(1, 0), R(1, 1));
            euler[Z] = 0.;
        }

        return euler;
    }

    // https://www.geometrictools.com/Documentation/EulerAngles.pdf
    inline Vec3d rotation_matrix_to_euler_zyx(const RotMat& R)
    {
        const int X = 2;
        const int Y = 1;
        const int Z = 0;
        Vec3d euler;
        double R20 = R(2, 0);
        if (R20 < 1) {
            if (R20 > -1) {
                euler[Y] = std::asin(-R20);
                euler[Z] = std::atan2(R(1, 0), R(0, 0));
                euler[X] = std::atan2(R(2, 1), R(2, 2));
            }
            else { // Not a unique solution, we choose one
                euler[Y] = M_PI / 2.;
                euler[Z] = -std::atan2(-R(1, 2), R(1, 1));
                euler[X] = 0.;
            }
        }
        else { // Not a unique solution, we choose one
            euler[Y] = -M_PI / 2.;
            euler[Z] = std::atan2(-R(1, 2), R(1, 1));
            euler[X] = 0.;
        }

        return euler;
    }

    // https://www.geometrictools.com/Documentation/EulerAngles.pdf
    inline Vec3d rotation_matrix_to_euler_yzx(const RotMat& R)
    {
        const int X = 2;
        const int Y = 0;
        const int Z = 1;
        Vec3d euler;
        double R10 = R(1, 0);
        if (R10 < 1) {
            if (R10 > -1) {
                euler[Z] = std::asin(-R10);
                euler[Y] = std::atan2(-R(2, 0), R(0, 0));
                euler[X] = std::atan2(-R(1, 2), R(1, 1));
            }
            else { // Not a unique solution, we choose one
                euler[Z] = -M_PI / 2.;
                euler[Y] = -std::atan2(R(2, 1), R(2, 2));
                euler[X] = 0.;
            }
        }
        else { // Not a unique solution, we choose one
            euler[Z] = M_PI / 2.;
            euler[Y] = std::atan2(R(2, 1), R(2, 2));
            euler[X] = 0.;
        }

        return euler;
    }

    inline double angle_dist(double a, double b)
    {
        double theta = b - a;
        while (theta < -M_PI)
            theta += 2 * M_PI;
        while (theta > M_PI)
            theta -= 2 * M_PI;
        return theta;
    }

    template <typename Derived>
    inline bool is_finite(const Eigen::MatrixBase<Derived>& x)
    {
        return ((x - x).array() == (x - x).array()).all();
    }

    inline Vector angle_wrap_multi(const Vector& theta)
    {
        Vector th = theta;
        for (int i = 0; i < theta.size(); i++) {
            th(i) = angle_dist(0, th(i));
        }
        return th;
    }

    inline Matrix pinv(const Matrix& jac, double l = 0.01)
    {
        int m = jac.rows();
        int n = jac.cols();
        if (n >= m) {
            return jac.transpose() * (jac * jac.transpose() + l * l * Matrix::Identity(m, m)).inverse();
        }
        return (jac.transpose() * jac + l * l * Matrix::Identity(n, n)).inverse() * jac.transpose();
    }
} // namespace srbd

#endif
