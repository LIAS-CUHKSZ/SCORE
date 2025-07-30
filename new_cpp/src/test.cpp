#include "RotFGO.h"
#include "TransFGO.h"
#include "helper.h"
#include <chrono>
#include <iostream>

int main()
{

  std::string data_folder = "/home/leoj/Github_Repos/SCORE/matlab/test_demo/";

  std::cout << "Reading CSV files..." << std::endl;
  auto intrinsic_data = helper::readCSV(data_folder + "camera_intrinsic.csv");
  auto gt_pose_data = helper::readCSV(data_folder + "gt_pose.csv");
  auto lines2D_data = helper::readCSV(data_folder + "2Dlines.csv");
  auto lines3D_data = helper::readCSV(data_folder + "3Dlines.csv");

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
  helper::matchLines(processed_lines2D, lines3D_data, ids, n_2D, v_3D, endpoints_3D);

  if (ids.empty())
  {
    std::cerr << "No line associations found!" << std::endl;
    return -1;
  }

  double branch_reso_r =
      M_PI / 256.0;                    // terminate bnb when branch size < resolution
  double sample_reso_r = M_PI / 256.0; // resolution for interval analysis
  double prox_thres_r = branch_reso_r;
  double epsilon_r = 0.015;
  double q_value_r = 0.9;

  // Translation parameters
  double branch_reso_t = 0.01;                // terminate bnb when branch size <= resolution
  double prox_thres_t = 0.01;                 // proximity threshold for clustering
  double epsilon_t = 0.03;                    // error tolerance for translation
  Eigen::Vector3d space_size(10.5, 6.0, 3.0); // Scene bounding box for "69e5939669"

  // Create initial branches (both hemispheres)
  std::vector<RBranch> initial_branches;
  initial_branches.push_back(RBranch(0, 0, M_PI, M_PI));        // East hemisphere
  initial_branches.push_back(RBranch(0, M_PI, M_PI, 2 * M_PI)); // West hemisphere

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

    // --- Rot Estimation ---
    std::cout << "\n--- Rotation Estimation ---" << std::endl;

    // Create solver and solve
    auto start_time = std::chrono::high_resolution_clock::now();
    RotFGO solver(branch_reso_r, epsilon_r, sample_reso_r, prox_thres_r, use_saturated, q_value_r);
    std::vector<Eigen::Matrix3d> R_candidates =
        solver.solve(n_2D, v_3D, ids, initial_branches);
    auto end_time = std::chrono::high_resolution_clock::now();

    double solve_time =
        std::chrono::duration<double>(end_time - start_time).count();

    if (R_candidates.empty())
    {
      std::cout << "No rotation candidates found!" << std::endl;
      continue;
    }

    Eigen::Matrix3d R_opt = R_candidates[0];
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

    // --- Translation Estimation ---
    std::cout << "\n--- Translation Estimation ---" << std::endl;

    auto trans_start_time = std::chrono::high_resolution_clock::now();
    // Create and run translation solver with internal preprocessing
    TransFGO trans_solver(branch_reso_t, epsilon_t, prox_thres_t, space_size, use_saturated);
    Eigen::Vector3d t_fine_tuned = trans_solver.solve(ids, R_opt, v_3D, n_2D, endpoints_3D,
                                                      epsilon_r, intrinsic_matrix);
    auto trans_end_time = std::chrono::high_resolution_clock::now();

    double trans_solve_time =
        std::chrono::duration<double>(trans_end_time - trans_start_time).count();

    if (t_fine_tuned.isZero())
    {
      std::cout << "Translation estimation failed!" << std::endl;
      continue;
    }

    // Calculate translation error
    double t_err = (t_fine_tuned - t_gt).norm();
    double total_time = solve_time + trans_solve_time;

    std::cout << "Translation error: " << t_err << " meters" << std::endl;
    std::cout << "Translation solve time: " << trans_solve_time << " seconds" << std::endl;
    std::cout << "Total time: " << total_time << " seconds" << std::endl;

    std::cout << "Estimated translation vector:" << std::endl;
    std::cout << t_fine_tuned.transpose() << std::endl;
    std::cout << "Ground truth translation:" << std::endl;
    std::cout << t_gt.transpose() << std::endl;

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Rotation error: " << angle_error * 180.0 / M_PI << " degrees" << std::endl;
    std::cout << "Translation error: " << t_err << " meters" << std::endl;
    std::cout << "Total pipeline time: " << total_time << " seconds" << std::endl;
  }
  return 0;
}
