/*
test code of one query image (Sat-CM v.s. CM)
for simplification, we exclude code for:
1. image retriveal 
2. observability check
3. select among multiple rotation candidates

Author:  Qingcheng Zeng <qzeng450@connect.hkust-gz.edu.cn>
         Haodong Jiang  <221049033@link.cuhk.edu.cn>
License: MIT
*/

#include "RotFGO.h"
#include "TransFGO.h"
#include "helper.h"
#include <chrono>
#include <iostream>

using pF_Buffer_Gen = std::function<Eigen::MatrixXd(const std::vector<int>&)>;

int main()
{
  // Read data
  std::string data_folder = "/home/leoj/Github_Repos/SCORE/csv_dataset/test_data/";
  std::cout << "Reading CSV files..." << std::endl;
  helper::ImageData image_data;
  helper::readTestData(data_folder, image_data);
  auto lines3D_data = helper::readCSV<double>(data_folder + "3Dlines.csv");
  std::cout << "Loaded " << image_data.lines2D_data.size() << " 2D lines and "
            << lines3D_data.size() << " 3D lines" << std::endl;

  // Semantic matching
  std::vector<int> ids;
  std::vector<Eigen::Vector3d> n_2D, v_3D, endpoints_3D;
  std::cout << "Performing semantic matching..." << std::endl;
  int num_associated_lines = helper::matchLines(image_data.lines2D_data, lines3D_data, ids, n_2D, v_3D, endpoints_3D);
  if (num_associated_lines<5)
  {
    std::cerr << "Query image has less than 5 associated 2D lines, skip." << std::endl;
    return -1;
  }

  // Rotation parameters
  double branch_reso_r = M_PI / 512.0;   // terminate branching a branch if its size < resolution
  double sample_reso_r = M_PI / 256.0; // resolution for interval analysis
  double prox_thres_r = branch_reso_r; // proximity threshold for clustering stabbers
  double epsilon_r = 0.015;            // rotation error tolerance
  double u_r = 1;

  // Create initial branches (both hemispheres)
  std::vector<RBranch> initial_branches;
  initial_branches.push_back(RBranch(0, 0, M_PI, M_PI));        // East hemisphere
  initial_branches.push_back(RBranch(0, M_PI, M_PI, 2 * M_PI)); // West hemisphere

  // saturation function buffer generator
  double q_value = 0.9;
  pF_Buffer_Gen pF_MLBuffer = [q_value, epsilon_r,u_r](const std::vector<int>& ids) {
    return helper::createMLBuffer(ids, q_value, epsilon_r, u_r);
  };
  pF_Buffer_Gen pF_CMBuffer = helper::createCMBuffer;
 
  // Translation parameters
  double branch_reso_t = 0.02;                // terminate branching a branch if its size <= resolution
  double prox_thres_t = branch_reso_t;        // proximity threshold for clustering stabbers
  double epsilon_t = 0.03;                    // error tolerance for translation
  Eigen::Vector3d space_size(10.5, 6.0, 3.0); // Scene bounding box for "69e5939669"
  pF_Buffer_Gen pF_TRBuffer = helper::createTRBuffer;

  for (int method = 0; method < 2; method++)
  {
    // Create rotation solver
    RotFGO rot_solver(branch_reso_r, epsilon_r, sample_reso_r, prox_thres_r);
    if (method == 1)
    {
      std::cout
          << "=== Relocalization with Saturated Consensus Maximization ==="
          << std::endl;
      rot_solver.setSatBufferFunc(pF_MLBuffer);
    }
    else
    {
      std::cout << "=== Relocalization with Classic Consensus Maximization ==="
                << std::endl;
      rot_solver.setSatBufferFunc(pF_CMBuffer);
    }
    // --- Rot Estimation ---
    std::cout << "\n--- Rotation Estimation ---" << std::endl;
    // solve rotation
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<Eigen::Matrix3d> R_candidates =
        rot_solver.solve(n_2D, v_3D, ids, initial_branches);
    auto end_time = std::chrono::high_resolution_clock::now();

    double solve_time =
        std::chrono::duration<double>(end_time - start_time).count();

    if (R_candidates.empty())
    {
      std::cout << "No rotation candidates found!" << std::endl;
      continue;
    }

    Eigen::Matrix3d R_opt = R_candidates[0];
    Eigen::Matrix3d R_error = R_opt * image_data.R_gt.transpose();
    double trace_R = R_error.trace();
    double angle_error =
        std::acos(std::max(-1.0, std::min(1.0, (trace_R - 1.0) / 2.0)));

    std::cout << "Number of BnB rotation candidates: " << R_candidates.size()
              << std::endl;

    // --- Translation Estimation ---
    std::cout << "\n--- Translation Estimation ---" << std::endl;
    // Create and run translation solver with internal preprocessing
    TransFGO trans_solver(branch_reso_t, epsilon_t, prox_thres_t, space_size);
    trans_solver.setSatBufferFunc(pF_TRBuffer);

    auto trans_start_time = std::chrono::high_resolution_clock::now();
    auto [best_score, t_fine_tuned] = trans_solver.solve(ids, R_opt, v_3D, n_2D, endpoints_3D, epsilon_r, image_data.intrinsic_matrix);
    auto trans_end_time = std::chrono::high_resolution_clock::now();

    double trans_solve_time =
        std::chrono::duration<double>(trans_end_time - trans_start_time).count();

    if (t_fine_tuned.isZero())
    {
      std::cout << "Translation estimation failed!" << std::endl;
      continue;
    }

    // Calculate translation error
    double t_err = (t_fine_tuned - image_data.t_gt).norm();
    double total_time = solve_time + trans_solve_time;

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Ground truth rotation matrix:" << std::endl;
    std::cout << image_data.R_gt << std::endl;
    std::cout << "Estimated rotation matrix:" << std::endl;
    std::cout << R_opt << std::endl;
    std::cout << "Ground truth translation:" << std::endl;
    std::cout << image_data.t_gt.transpose() << std::endl;
    std::cout << "Estimated translation vector:" << std::endl;
    std::cout << t_fine_tuned.transpose() << std::endl;
    std::cout << "Rotation solve time: " << solve_time << " seconds" << std::endl;
    std::cout << "Translation solve time: " << trans_solve_time << " seconds" << std::endl;
    std::cout << "Total Pipeline time: " << total_time << " seconds" << std::endl;
    std::cout << "Rotation error: " << angle_error * 180.0 / M_PI << " degrees" << std::endl;
    std::cout << "Translation error: " << t_err << " meters" << std::endl;
  }
  return 0;
}
