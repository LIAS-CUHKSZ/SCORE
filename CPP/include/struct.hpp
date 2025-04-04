#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <math.h>
using namespace Eigen;
using namespace std;

struct help_attribute {
  bool east_or_not;
  double norm_of_outer_product;
  double inner_product;
  Vector3d outer_product;
  Vector3d east_outer_product;
  Vector3d west_outer_product;
  Vector2d angle_east_outer_product;
  Vector2d angle_west_outer_product;

  Vector3d east_angle_bisector;
  Vector3d west_angle_bisector;
  Vector3d east_othogonal_angle_bisector;
  Vector3d west_othogonal_angle_bisector;
  Vector2d angle_east_angle_bisector;
  Vector2d angle_west_angle_bisector;
  Vector2d angle_east_othogonal_angle_bisector;
  Vector2d angle_west_othogonal_angle_bisector;
};
struct line_pair {
  // 2d line
  Vector3d line_2d = Vector3d::Zero();
  // 3d line
  Vector3d line_3d = Vector3d::Zero();
  // 3d point
  Vector3d point_3d = Vector3d::Zero();
  // whether the line is inliner
  bool inliner = true;
  // semactic tag
  int line_tag;
  // residual of translation
  double t_res(Matrix3d R, Vector3d t) {
    return (R * line_2d).dot(point_3d - t);
  };
  // residual of rotation
  double angle(Matrix3d R) {
    double theta;
    theta = acos(line_2d.dot(line_3d.transpose() * R)) - M_PI_2;
    return abs(theta);
  };
  help_attribute *attribute = new help_attribute();
};
struct range {
  // interval left and right
  range(double small, double big) {
    if (small > big) {
      lower = big;
      upper = small;

    } else {
      lower = small;
      upper = big;
    };
    width = upper - lower;
    center = (lower + upper) / 2;
  };
  double width;
  double lower;
  double upper;
  double center;
};
// cube of alpha and phi
struct Square {
  range alpha;
  range phi;
  double center[2];
  double lower_bound;
  double upper_bound;
  double theta_hat;
  Square(range x, range y) : alpha(x), phi(y) {
    center[0] = alpha.center;
    center[1] = phi.center;
  };
  // formulate the comparison operator
  bool operator<(const Square &other) const {
    return upper_bound < other.upper_bound;
  };
};
