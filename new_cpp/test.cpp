#include "RotFGO.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Utility function to read CSV files
std::vector<std::vector<double>> readCSV(const std::string &filename)
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
void matchLines(const std::vector<std::vector<double>> &lines2D,
                const std::vector<std::vector<double>> &lines3D,
                std::vector<int> &ids, std::vector<Eigen::Vector3d> &n_2D,
                std::vector<Eigen::Vector3d> &v_3D,
                std::vector<Eigen::Vector3d> &endpoints_3D)
{

  // Clear output vectors
  ids.clear();
  n_2D.clear();
  v_3D.clear();
  endpoints_3D.clear();

  // Count total matches first
  int total_matches = 0;
  for (size_t i = 0; i < lines2D.size(); i++)
  {
    double semantic_id_2d = lines2D[i][3]; // semantic_id column
    for (size_t j = 0; j < lines3D.size(); j++)
    {
      double semantic_id_3d = lines3D[j][6]; // semantic_id column
      if (std::abs(semantic_id_2d - semantic_id_3d) < 0.1)
      {
        total_matches++;
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

  std::cout << "Found " << total_matches << " line associations" << std::endl;
}

int main()
{

  std::string data_folder = "D:\\Desktop\\SCORE\\matlab\\test_demo\\";

  std::cout << "Reading CSV files..." << std::endl;
  auto intrinsic_data = readCSV(data_folder + "camera_intrinsic.csv");
  auto gt_pose_data = readCSV(data_folder + "gt_pose.csv");
  auto lines2D_data = readCSV(data_folder + "2Dlines.csv");
  auto lines3D_data = readCSV(data_folder + "3Dlines.csv");

  if (intrinsic_data.empty() || gt_pose_data.empty() || lines2D_data.empty() ||
      lines3D_data.empty())
  {
    std::cerr << "Error: Could not read all required CSV files" << std::endl;
    return -1;
  }

  std::cout << "Loaded " << lines2D_data.size() << " 2D lines and "
            << lines3D_data.size() << " 3D lines" << std::endl;

  // Extract camera intrinsics
  double fx = intrinsic_data[0][0];
  double cx = intrinsic_data[0][1];
  double fy = intrinsic_data[0][2];
  double cy = intrinsic_data[0][3];

  Eigen::Matrix3d intrinsic_matrix;
  intrinsic_matrix << fx, 0, cx, 0, fy, cy, 0, 0, 1;
  std::cout << "Camera intrinsic matrix:\n"
            << intrinsic_matrix << std::endl;

  std::cout << "Camera intrinsics loaded: fx=" << fx << ", fy=" << fy
            << ", cx=" << cx << ", cy=" << cy << std::endl;

  Eigen::Matrix3d R_gt;
  Eigen::Vector3d t_gt;
  R_gt << gt_pose_data[0][0], gt_pose_data[0][1], gt_pose_data[0][2],
      gt_pose_data[1][0], gt_pose_data[1][1], gt_pose_data[1][2],
      gt_pose_data[2][0], gt_pose_data[2][1], gt_pose_data[2][2];
  t_gt << gt_pose_data[0][3], gt_pose_data[1][3], gt_pose_data[2][3];

  // Process 2D lines: normalize normal vectors
  std::vector<std::vector<double>> processed_lines2D = lines2D_data;
  for (auto &line : processed_lines2D)
  {
    // lines2D format: A, B, C, semantic_id, ua, va, ub, vb,
    Eigen::Vector3d normal(line[0], line[1], line[2]);
    normal = normal.transpose() * intrinsic_matrix;
    normal.normalize();
    line[0] = normal(0);
    line[1] = normal(1);
    line[2] = normal(2);
  }

  std::vector<int> ids;
  std::vector<Eigen::Vector3d> n_2D, v_3D, endpoints_3D;
  std::cout << "Performing semantic matching..." << std::endl;
  matchLines(processed_lines2D, lines3D_data, ids, n_2D, v_3D, endpoints_3D);

  if (ids.empty())
  {
    std::cerr << "No line associations found!" << std::endl;
    return -1;
  }

  double prox_thres_r = 1.0 * M_PI / 180.0;
  double branch_reso_r =
      M_PI / 256.0;                    // terminate bnb when branch size < resolution
  double sample_reso_r = M_PI / 256.0; // resolution for interval analysis
  double epsilon_r = 0.015;
  int west_east_flag = 2; // TODO: branch over whole sphere?

  for (int method = 0; method < 2; method++)
  {
    bool use_saturated = (method == 1);

    std::cout << std::endl;
    if (use_saturated)
    {
      std::cout
          << "=== Relocalization with Saturated Consensus Maximization ==="
          << std::endl;
    }
    else
    {
      std::cout << "=== Relocalization with Classic Consensus Maximization ==="
                << std::endl;
    }

    // Create kernel buffer
    Eigen::MatrixXd kernel_buffer =
        RotFGO::createKernelBuffer(ids, use_saturated);

    auto start_time = std::chrono::high_resolution_clock::now();
    RotFGO solver;
    std::vector<Eigen::Matrix3d> R_candidates =
        solver.solve(n_2D, v_3D, ids, kernel_buffer, branch_reso_r, epsilon_r,
                     sample_reso_r, prox_thres_r, west_east_flag);
    auto end_time = std::chrono::high_resolution_clock::now();

    double solve_time =
        std::chrono::duration<double>(end_time - start_time).count();

    if (R_candidates.empty())
    {
      std::cout << "No rotation candidates found!" << std::endl;
      continue;
    }

    Eigen::Matrix3d R_opt = R_candidates[0].transpose();
    Eigen::Matrix3d R_error = R_opt * R_gt.transpose();
    double trace_R = R_error.trace();
    double angle_error =
        std::acos(std::max(-1.0, std::min(1.0, (trace_R - 1.0) / 2.0)));

    std::cout << "Number of rotation candidates: " << R_candidates.size()
              << std::endl;
    std::cout << "Rotation error: " << angle_error << " radians ("
              << angle_error * 180.0 / M_PI << " degrees)" << std::endl;
    std::cout << "Solve time: " << solve_time << " seconds" << std::endl;

    std::cout << "Estimated rotation matrix:" << std::endl;
    std::cout << R_opt << std::endl;
  }
  return 0;
}
