#include "helper.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

std::vector<std::vector<double>> helper::readCSV(const std::string &filename)
{
  std::vector<std::vector<double>> data;
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open())
  {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return data;
  }

  // Skip header line
  std::getline(file, line);

  while (std::getline(file, line))
  {
    std::vector<double> row;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ','))
    {
      try
      {
        row.push_back(std::stod(cell));
      }
      catch (const std::exception &e)
      {
        std::cerr << "Error parsing value: " << cell << std::endl;
        row.push_back(0.0);
      }
    }
    data.push_back(row);
  }

  file.close();
  return data;
}

// Function to perform semantic matching (simplified version)
int helper::matchLines(const std::vector<std::vector<double>> &lines2D,
                const std::vector<std::vector<double>> &lines3D,
                std::vector<int> &ids, std::vector<Eigen::Vector3d> &n_2D,
                std::vector<Eigen::Vector3d> &v_3D,
                std::vector<Eigen::Vector3d> &endpoints_3D)
{
  int associated_2D_line_num = 0;
  // Clear output vectors
  ids.clear();
  n_2D.clear();
  v_3D.clear();
  endpoints_3D.clear();

  // Count total matches first
  int total_matches = 0;
  for (size_t i = 0; i < lines2D.size(); i++)
  {
    int flag = 0;
    double semantic_id_2d = lines2D[i][3]; // semantic_id column
    for (size_t j = 0; j < lines3D.size(); j++)
    {
      double semantic_id_3d = lines3D[j][6]; // semantic_id column
      if (std::abs(semantic_id_2d - semantic_id_3d) < 0.1)
      {
        total_matches++;
        if (!flag)
        {
          associated_2D_line_num++;
          flag = 1;
        }
      }
    }
  }

  // Reserve space
  ids.reserve(total_matches);
  n_2D.reserve(total_matches);
  v_3D.reserve(total_matches);
  endpoints_3D.reserve(total_matches * 2);

  // Perform matching
  for (size_t i = 0; i < lines2D.size(); i++)
  {
    double semantic_id_2d = lines2D[i][3];
    Eigen::Vector3d normal_2d(lines2D[i][0], lines2D[i][1], lines2D[i][2]);

    for (size_t j = 0; j < lines3D.size(); j++)
    {
      double semantic_id_3d = lines3D[j][6];
      if (std::abs(semantic_id_2d - semantic_id_3d) < 0.1)
      {
        // Add match
        ids.push_back(i);
        n_2D.push_back(normal_2d);

        // Calculate 3D line direction vector
        Eigen::Vector3d p1(lines3D[j][0], lines3D[j][1], lines3D[j][2]);
        Eigen::Vector3d p2(lines3D[j][3], lines3D[j][4], lines3D[j][5]);
        Eigen::Vector3d direction = p2 - p1;
        direction.normalize();
        v_3D.push_back(direction);

        // Add endpoints
        endpoints_3D.push_back(p1);
        endpoints_3D.push_back(p2);
      }
    }
  }
  return associated_2D_line_num;
}

std::vector<double> helper::rot2angle(const Eigen::Matrix3d &R)
{
  Eigen::AngleAxisd rotation_vector(R);
  double theta = rotation_vector.angle();
  Eigen::Vector3d axis = rotation_vector.axis();
  double alpha, phi;
  if (axis(0) == 0 && axis(1) == 0)
  {
    phi = 0;
    alpha = std::acos(axis(2));
  }
  else if (axis(0) == 0)
  {
    phi = M_PI / 2;
    alpha = std::acos(axis(2));
  }
  else
  {
    phi = std::atan2(axis(1), axis(0));
    alpha = std::atan2(std::sqrt(axis(0) * axis(0) + axis(1) * axis(1)), axis(2));
  }
  if (phi < 0) // [-pi,pi]-->[0,2pi]
    phi += 2 * M_PI;
  std::vector<double> angles = {alpha, phi, theta};
  return angles;
}

std::vector<std::vector<double>> helper::confine_sphere(double alpha, double phi, double side_length, double delta)
{
  const double eps = 1e-14;
  const double pi = M_PI;
  const double two_pi = 2.0 * M_PI;
  
  std::vector<std::vector<double>> branch; // each branch is a vector of [alpha_l, phi_l, alpha_u, phi_u]
  
  // Handle alpha (north-south) branches
  std::vector<std::vector<double>> branch_ns_list;
  int k_alpha = static_cast<int>(std::floor(alpha / side_length));
  
  if (alpha - k_alpha * side_length <= delta && k_alpha > 0)
  {
    // Add two branches: 
    branch_ns_list.push_back({(k_alpha - 1) * side_length, k_alpha * side_length});
    branch_ns_list.push_back({k_alpha * side_length, (k_alpha + 1) * side_length});
  }
  else if ((k_alpha + 1) * side_length - alpha <= delta && (k_alpha + 1) * side_length + eps < pi)
  {
    // Add two branches: 
    branch_ns_list.push_back({k_alpha * side_length, (k_alpha + 1) * side_length});
    branch_ns_list.push_back({(k_alpha + 1) * side_length, (k_alpha + 2) * side_length});
  }
  else
  {
    // Add single branch: 
    branch_ns_list.push_back({k_alpha * side_length, (k_alpha + 1) * side_length});
  }
  
  // Handle phi (west-east) branches
  std::vector<std::vector<double>> branch_we_list;
  int k_phi = static_cast<int>(std::floor(phi / side_length));
  
  if (phi - k_phi * side_length <= delta)
  {
    // Add two branches: 
    branch_we_list.push_back({(k_phi - 1) * side_length, k_phi * side_length});
    branch_we_list.push_back({k_phi * side_length, (k_phi + 1) * side_length});
  }
  else if ((k_phi + 1) * side_length - phi <= delta)
  {
    // Add two branches: 
    branch_we_list.push_back({k_phi * side_length, (k_phi + 1) * side_length});
    branch_we_list.push_back({(k_phi + 1) * side_length, (k_phi + 2) * side_length});
  }
  else
  {
    // Add single branch: 
    branch_we_list.push_back({k_phi * side_length, (k_phi + 1) * side_length});
  }
  
  // Generate all combinations of alpha and phi branches
  for (const auto& we_branch : branch_we_list)
  {
    double phi_l = we_branch[0];
    double phi_u = we_branch[1];
    
    // Handle phi boundary conditions (wrapping around 2π)
    if (phi_l < 0)
    {
      phi_l = two_pi - side_length;
      phi_u = two_pi;
    }
    if (phi_u > two_pi)
    {
      phi_l = 0;
      phi_u = side_length;
    }
    
    for (const auto& ns_branch : branch_ns_list)
    {
      double alpha_l = ns_branch[0];
      double alpha_u = ns_branch[1];
      
      // Add branch as [alpha_l, phi_l, alpha_u, phi_u]
      branch.push_back({alpha_l, phi_l, alpha_u, phi_u});
    }
  }
  
  return branch;
}
