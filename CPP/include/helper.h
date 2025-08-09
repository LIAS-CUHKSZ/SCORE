/*
Helper functions

Author:  Qingcheng Zeng <qzeng450@connect.hkust-gz.edu.cn>
         Haodong Jiang  <221049033@link.cuhk.edu.cn>
License: MIT
*/

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#ifndef HELPER_H
#define HELPER_H

using pF_Buffer_Gen = std::function<Eigen::MatrixXd(const std::vector<int>&)>;

namespace helper
{
struct ImageData
{
    Eigen::Matrix3d intrinsic_matrix;
    Eigen::Matrix3d R_gt;
    Eigen::Vector3d t_gt;
    std::vector<std::vector<double>> lines2D_data;
    std::vector<int> retrived_3D_line_idx;
    Eigen::Matrix3d R_retrived;
    Eigen::Vector3d t_retrived;
};

// Utility function to read CSV files
template <typename T>
std::vector<std::vector<T>> readCSV(const std::string &filename);


// Utility function to read data for test.cpp
void readTestData(const std::string &data_folder, ImageData &image_data);

// Utility function to read data for Experiment.cpp
void readData(const std::string &data_folder, ImageData &image_data,const std::string query_image_name);

// Function to perform semantic matching (simplified version)
// return the number of 2D lines that are associated with 3D lines
int matchLines(const std::vector<std::vector<double>> &lines2D,
                const std::vector<std::vector<double>> &lines3D,
                std::vector<int> &ids, std::vector<Eigen::Vector3d> &n_2D,
                std::vector<Eigen::Vector3d> &v_3D,
                std::vector<Eigen::Vector3d> &endpoints_3D);

// Function to convert rotation matrix to alpha phi(axis polar coordinates) and theta(angle)
// alpha:[0,pi], phi:[0,2pi], theta:[0,pi]
std::vector<double> rot2angle(const Eigen::Matrix3d &R);

Eigen::Vector3d polarToXyz(double alpha, double phi) noexcept;
  
std::pair<double, double> xyzToPolar(const Eigen::Vector3d &axis) noexcept;

// Function to confine the search space of rotation axis around the input axis
// alpha: [0,pi], phi: [0,2*pi], side_length: pi, pi/2, pi/4, ..., delta: scalar, define the ambiguous region
// Returns a matrix where each column represents a branch with [alpha_l, phi_l, alpha_u, phi_u]
std::vector<std::vector<double>> confine_sphere(double alpha, double phi, double side_length, double delta);

// Function to create classic consensus maximization buffer
Eigen::MatrixXd createCMBuffer(const std::vector<int> &ids);

// Function to create truncated saturation function buffer
Eigen::MatrixXd createTRBuffer(const std::vector<int> &ids);

// Function to create maximum likelihood saturation function buffer
Eigen::MatrixXd createMLBuffer(const std::vector<int> &ids, double q_value, double epsilon, double u);

// Function to calculate the score of a set of inlier ids
double calcScore(const std::vector<int> &inlier_ids,
                 const Eigen::MatrixXd &sat_buffer);
}
#endif // HELPER_H


