#include "helper.hpp"
#include "struct.hpp"
#include <cmath>
#include <iostream>
#include <utility>
using namespace std;
Eigen::Matrix3d rodrigues(double alpha, double phi, double theta) {
  // Eigen::Matrix3d R;
  // Eigen::Vector3d axis(sin(alpha) * cos(phi), sin(alpha) * sin(phi),
  // cos(alpha)); Eigen::AngleAxisd angle_axis(theta, axis); return
  // angle_axis.toRotationMatrix();
  double a_s = sin(alpha);
  double a_c = cos(alpha);
  double p_s = sin(phi);
  double p_c = cos(phi);
  double ax = a_s * p_c;
  double ay = a_s * p_s;
  double az = a_c;

  double c = cos(theta);
  double s = sin(theta);

  Eigen::Matrix3d R;
  R(0, 0) = c + (1 - c) * ax * ax;
  R(0, 1) = (1 - c) * ax * ay - s * az;
  R(0, 2) = (1 - c) * ax * az + s * ay;
  R(1, 0) = (1 - c) * ax * ay + s * az;
  R(1, 1) = c + (1 - c) * ay * ay;
  R(1, 2) = (1 - c) * ay * az - s * ax;
  R(2, 0) = (1 - c) * ax * az - s * ay;
  R(2, 1) = (1 - c) * ay * az + s * ax;
  R(2, 2) = c + (1 - c) * az * az;

  return R;
};
Vector3d quanterion2angles(Quaterniond q) {
  if (q.w() < 0) {
    q.coeffs() = -q.coeffs();
  }
  double theta = 2 * acos(q.w());
  double sin = sqrt(1 - q.w() * q.w());
  double x = q.x() / sin;
  double y = q.y() / sin;
  double z = q.z() / sin;
  double phi = atan2(y, x);
  double alpha = acos(z);
  return Vector3d(alpha, phi, theta);
};
Vector3d quanterion2rotvec(Quaterniond q) {
  if (q.w() < 0) {
    q.coeffs() = -q.coeffs();
  }
  double theta = 2 * acos(q.w());
  double sin = sqrt(1 - q.w() * q.w());
  double x = q.x() / sin;
  double y = q.y() / sin;
  double z = q.z() / sin;

  return Vector3d(x, y, z) * theta;
};
double R_error(Matrix3d R1, Matrix3d R2) {
  Matrix3d R = R1.transpose() * R2;
  double trace = R.trace();
  double error = acos((trace - 1) / 2);
  // return abs(error);
  return error;
};
double t_error(Vector3d t1, Vector3d t2) { return (t1 - t2).norm(); };

vector<double> interval_intersection(double &interval_1_left,
                                     double &interval_1_right,
                                     double &interval_2_left,
                                     double &interval_2_right) {
  vector<double> intersection;
  intersection.reserve(2);
  if (interval_1_right < interval_2_left ||
      interval_2_right < interval_1_left) {
    return intersection;
  }
  intersection.emplace_back(max(interval_1_left, interval_2_left));
  intersection.emplace_back(min(interval_1_right, interval_2_right));
  return intersection;
};

pair<double, double> interval_projection(double a, range interval) {
  double far, near;
  if (a < interval.lower) {
    far = interval.upper;
    near = interval.lower;
  } else if (a <= interval.center) {
    far = interval.upper;
    near = a;
  } else if (a <= interval.upper) {
    far = interval.lower;
    near = a;
  } else {
    far = interval.lower;
    near = interval.upper;
  }
  return make_pair(far, near);
};

Vector3d polar_2_xyz(const Vector2d &angle) {
  double a_s = sin(angle[0]);
  return Vector3d(a_s * cos(angle[1]), a_s * sin(angle[1]), cos(angle[0]));
};

Vector2d xyz_2_polar(Vector3d xyz) {
  Vector2d polar;
  double length = xyz.norm();

  if (length == 0) {
    polar(0) = 0;
    polar(1) = 0;
    return polar;
  } else {
    xyz /= length;
  }

  if (xyz(0) == 0 && xyz(1) == 0) {
    polar(1) = 0;
    polar(0) = acos(xyz(2));
  } else if (xyz(0) == 0) {
    polar(1) = M_PI / 2;
    polar(0) = acos(xyz(2));
  } else {
    polar(1) = atan2(xyz(1), xyz(0));
    polar(0) = atan2(sqrt(xyz(0) * xyz(0) + xyz(1) * xyz(1)), xyz(2));
  }

  if (polar(1) < 0) {
    polar(1) += 2 * M_PI;
  }

  return polar;
};
Vector3d polar_2_xyz(double alpha, double phi) {
  double a_s = sin(alpha);
  return Vector3d(a_s * cos(phi), a_s * sin(phi), cos(alpha));

}; // 新增的重载函数声明
