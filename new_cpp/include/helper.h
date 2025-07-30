#include <vector>
#include <string>
#include <Eigen/Dense>
#ifndef HELPER_H
#define HELPER_H

namespace helper
{
// Utility function to read CSV files
std::vector<std::vector<double>> readCSV(const std::string &filename);

// Function to perform semantic matching (simplified version)
void matchLines(const std::vector<std::vector<double>> &lines2D,
                const std::vector<std::vector<double>> &lines3D,
                std::vector<int> &ids, std::vector<Eigen::Vector3d> &n_2D,
                std::vector<Eigen::Vector3d> &v_3D,
                std::vector<Eigen::Vector3d> &endpoints_3D);
}   
#endif // HELPER_H