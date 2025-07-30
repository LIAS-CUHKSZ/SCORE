#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#ifndef HELPER_H
#define HELPER_H

namespace helper
{
// Utility function to read CSV files
std::vector<std::vector<double>> readCSV(const std::string &filename);

// Function to perform semantic matching (simplified version)
// return number of associated 2D lines
int matchLines(const std::vector<std::vector<double>> &lines2D,
                const std::vector<std::vector<double>> &lines3D,
                std::vector<int> &ids, std::vector<Eigen::Vector3d> &n_2D,
                std::vector<Eigen::Vector3d> &v_3D,
                std::vector<Eigen::Vector3d> &endpoints_3D);

// Function to convert rotation matrix to alpha phi(axis polar coordinates) and theta(angle)
// alpha:[0,pi], phi:[0,2pi], theta:[0,pi]
std::vector<double> rot2angle(const Eigen::Matrix3d &R);

// Function to confine the search space of rotation axis around the input axis
// alpha: [0,pi], phi: [0,2*pi], side_length: pi, pi/2, pi/4, ..., delta: scalar, define the ambiguous region
// Returns a matrix where each column represents a branch with [alpha_l, phi_l, alpha_u, phi_u]
std::vector<std::vector<double>> confine_sphere(double alpha, double phi, double side_length, double delta);
}   
#endif // HELPER_H