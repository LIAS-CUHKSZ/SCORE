#pragma once
#include "struct.hpp"
#include <Eigen/Dense>
#include <cmath>
using namespace Eigen;
Matrix3d rodrigues(double alpha, double phi, double theta);
Vector3d quanterion2angles(Quaterniond q);
Vector3d quanterion2rotvec(Quaterniond q);
double R_error(Matrix3d R1, Matrix3d R2);
double t_error(Vector3d t1, Vector3d t2);

vector<double> interval_intersection(double &interval_1_left,
                                     double &interval_1_right,
                                     double &interval_2_left,
                                     double &interval_2_right);
pair<double, double> interval_projection(double a, range interval);
Vector2d xyz_2_polar(Vector3d xyz);
Vector3d polar_2_xyz(const Vector2d &angle);
Vector3d polar_2_xyz(double alpha, double phi);