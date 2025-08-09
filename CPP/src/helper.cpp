#include "helper.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

constexpr double PI = M_PI;

template <typename T>
std::vector<std::vector<T>> helper::readCSV(const std::string &filename)
{
  std::vector<std::vector<T>> data;
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open())
  {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return data;
  }

  // Skip header line
  std::getline(file, line);
  double count = 0;
  while (std::getline(file, line))
  {
    std::vector<T> row;
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
        count++;
        // std::cerr << "Error parsing value: " << cell << std::endl;
        std::cerr << count <<" 2D line(s) without corresponding 3D line in the map"<< std::endl;
        row.push_back(0.0);
      }
    }
    data.push_back(row);
  }

  file.close();
  return data;
}

void helper::readTestData(const std::string &data_folder, helper::ImageData &image_data)
{
  auto intrinsic_data = helper::readCSV<double>(data_folder + "camera_intrinsic.csv");
  auto gt_pose_data = helper::readCSV<double>(data_folder + "gt_pose.csv");
  auto lines2D_data = helper::readCSV<double>(data_folder + "2Dlines.csv");

  if (intrinsic_data.empty() || gt_pose_data.empty() || lines2D_data.empty())
  {
    std::cerr << "Error: Could not read all required CSV files" << std::endl;
    return;
  }

  // Extract camera intrinsics
  double fx = intrinsic_data[0][0];
  double cx = intrinsic_data[0][1];
  double fy = intrinsic_data[0][2];
  double cy = intrinsic_data[0][3];

  image_data.intrinsic_matrix << fx, 0, cx, 0, fy, cy, 0, 0, 1;
  image_data.R_gt << gt_pose_data[0][0], gt_pose_data[0][1], gt_pose_data[0][2],
      gt_pose_data[1][0], gt_pose_data[1][1], gt_pose_data[1][2],
      gt_pose_data[2][0], gt_pose_data[2][1], gt_pose_data[2][2];
  image_data.t_gt << gt_pose_data[0][3], gt_pose_data[1][3], gt_pose_data[2][3];

  // Process 2D lines: normalize normal vectors
  for (auto &line : lines2D_data)
  {
    // lines2D format: A, B, C, semantic_id, ua, va, ub, vb,
    Eigen::Vector3d normal(line[0], line[1], line[2]);
    normal = normal.transpose() * image_data.intrinsic_matrix;
    normal.normalize();
    line[0] = normal(0);
    line[1] = normal(1);
    line[2] = normal(2);
  }
  image_data.lines2D_data = lines2D_data;
  return;
}

void helper::readData(const std::string &data_folder, helper::ImageData &image_data, const std::string query_image_name)
{
  auto intrinsic_data = helper::readCSV<double>(data_folder + "intrinsics/" + query_image_name + ".csv");
  auto gt_pose_data = helper::readCSV<double>(data_folder + "poses/" + query_image_name + ".csv");
  auto lines2D_data = helper::readCSV<double>(data_folder + "lines2D/" + query_image_name + "_2Dlines.csv");
  auto retrived_3D_line_idx = helper::readCSV<int>(data_folder + "retrived_3D_line_idx/" + query_image_name + ".csv");
  auto retrived_closest_pose = helper::readCSV<double>(data_folder + "retrived_closest_pose/" + query_image_name + "_retrived_pose.csv");
  
  if (intrinsic_data.empty() || gt_pose_data.empty() || lines2D_data.empty())
  {
    std::cerr << "Error: Could not read all required CSV files" << std::endl;
    return;
  }

  // Extract camera intrinsics
  double fx = intrinsic_data[0][0];
  double cx = intrinsic_data[0][1];
  double fy = intrinsic_data[0][2];
  double cy = intrinsic_data[0][3];

  image_data.intrinsic_matrix << fx, 0, cx, 0, fy, cy, 0, 0, 1;
  image_data.R_gt << gt_pose_data[0][0], gt_pose_data[0][1], gt_pose_data[0][2],
      gt_pose_data[1][0], gt_pose_data[1][1], gt_pose_data[1][2],
      gt_pose_data[2][0], gt_pose_data[2][1], gt_pose_data[2][2];
  image_data.t_gt << gt_pose_data[0][3], gt_pose_data[1][3], gt_pose_data[2][3];

  // Process 2D lines: normalize normal vectors
  for (auto &line : lines2D_data)
  {
    // lines2D format: A, B, C, semantic_id, ua, va, ub, vb,
    Eigen::Vector3d normal(line[0], line[1], line[2]);
    normal = normal.transpose() * image_data.intrinsic_matrix;
    normal.normalize();
    line[0] = normal(0);
    line[1] = normal(1);
    line[2] = normal(2);
  }
  image_data.lines2D_data = lines2D_data;
  for (size_t i = 0; i < retrived_3D_line_idx.size(); i++)
      image_data.retrived_3D_line_idx.push_back(retrived_3D_line_idx[i][0]);
  image_data.R_retrived << retrived_closest_pose[0][0], retrived_closest_pose[0][1],
      retrived_closest_pose[0][2], retrived_closest_pose[1][0],
      retrived_closest_pose[1][1], retrived_closest_pose[1][2],
      retrived_closest_pose[2][0], retrived_closest_pose[2][1],
      retrived_closest_pose[2][2];
  image_data.t_retrived << retrived_closest_pose[0][3], retrived_closest_pose[1][3], retrived_closest_pose[2][3];
  return;
}



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

Eigen::Vector3d helper::polarToXyz(double alpha, double phi) noexcept
{
  double sin_alpha = std::sin(alpha);
  return Eigen::Vector3d(sin_alpha * std::cos(phi), sin_alpha * std::sin(phi),
                         std::cos(alpha));
}

std::pair<double, double> helper::xyzToPolar(const Eigen::Vector3d &axis) noexcept
{
  double length = axis.norm();
  if (length == 0)
    return {0.0, 0.0};

  Eigen::Vector3d unit_axis = axis / length;
  double alpha = std::atan2(
      std::sqrt(unit_axis(0) * unit_axis(0) + unit_axis(1) * unit_axis(1)),
      unit_axis(2));
  double phi;

  if (unit_axis(0) == 0 && unit_axis(1) == 0)
    phi = 0.0;
  else
    phi = std::atan2(unit_axis(1), unit_axis(0));

  if (phi < 0)
    phi += 2 * PI;

  return {alpha, phi};
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

Eigen::MatrixXd helper::createMLBuffer(const std::vector<int> &ids, double q_value, double epsilon, double u)
{
  int max_id = *std::max_element(ids.begin(), ids.end());
  int num_2d_lines = max_id + 1;
  std::vector<int> match_count(num_2d_lines, 0);
  // count the number of total associations for each 2D line
  for (size_t id : ids)
    match_count[id]++;
  int max_count = *std::max_element(match_count.begin(), match_count.end());
  Eigen::MatrixXd sat_buffer =
      Eigen::MatrixXd::Zero(num_2d_lines, max_count);
    // Saturated consensus maximization based on likelihood with hyperparameter q
    double C = q_value / (1 - q_value) * u / epsilon;
    for (int i = 0; i < num_2d_lines; i++)
    {
      if (match_count[i] == 0)
        continue;
      double d = -match_count[i] * std::log(u);
      for (int j = 0; j < match_count[i]; j++)
      {
        sat_buffer(i, j) = d + std::log(1.0 + C * (j+1) / match_count[i]) -
                               std::log(1.0 + C * j / match_count[i]);
      }
    }
  return sat_buffer;
}

Eigen::MatrixXd helper::createCMBuffer(const std::vector<int> &ids)
{
  int max_id = *std::max_element(ids.begin(), ids.end());
  int num_2d_lines = max_id + 1;
  std::vector<int> match_count(num_2d_lines, 0);
  // count the number of total associations for each 2D line
  for (size_t id : ids)
    match_count[id]++;
  int max_count = *std::max_element(match_count.begin(), match_count.end());
  Eigen::MatrixXd sat_buffer = Eigen::MatrixXd::Ones(num_2d_lines, max_count);
  
  return sat_buffer;
}


Eigen::MatrixXd helper::createTRBuffer(const std::vector<int> &ids)
{
  int max_id = *std::max_element(ids.begin(), ids.end());
  int num_2d_lines = max_id + 1;
  std::vector<int> match_count(num_2d_lines, 0);
  for (size_t id : ids)
    match_count[id]++;
  int max_count = *std::max_element(match_count.begin(), match_count.end());
  Eigen::MatrixXd sat_buffer =
      Eigen::MatrixXd::Zero(num_2d_lines, max_count);
  sat_buffer.col(0).setOnes();  // only count the first inlier association
  return sat_buffer;
}

double helper::calcScore(const std::vector<int> &inlier_ids,
                         const Eigen::MatrixXd &sat_buffer)
{
    double score = 0.0;
    int num_2D_lines = sat_buffer.rows();
    std::vector<int> inlier_count(num_2D_lines, 0);
    for (int id : inlier_ids)
      inlier_count[id]++;

    for (int i = 0; i < num_2D_lines; i++)
    {
      if (inlier_count[i] == 0)
        continue;
      for (int j = 0; j < inlier_count[i]; j++)
        score += sat_buffer(i, j);
    }

    return score;
} 