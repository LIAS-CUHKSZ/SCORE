#include "RotFGO.h"
#include "TransFGO.h"
#include "helper.h"
#include <chrono>
#include <fstream>
#include <iostream>

using namespace std;
vector<string> scene_names = {"S1(workstation)", "S2(office)", "S3(game bar)",
                              "S4(art room)"};
vector<Eigen::Vector3d> room_sizes = {Eigen::Vector3d(8, 6, 4),
                                      Eigen::Vector3d(7, 7, 3),
                                      Eigen::Vector3d(10.5, 5, 3.5),
                                      Eigen::Vector3d(10.5, 6, 3.0)};
int main(int argc, char **argv) {
  // load arguments
  if (argc != 5) {
    std::cerr << "Usage: 1 y 2 4 (choose S1, use gt labels, divide side length pi by 2, image index 2)" << std::endl;
    return -1;
  }
  int scene_id = std::stoi(argv[1]);
  bool use_gt_labels = (std::string(argv[2]) == "y");
  int side_length_divide = std::stoi(argv[3]);
  int image_index = std::stoi(argv[4]);

  std::cout << "Chosen scene: " << scene_names[scene_id - 1] << std::endl;
  std::cout << "Use gt labels: " << use_gt_labels << std::endl;

  // load data
  string data_folder;
  if (use_gt_labels)
    data_folder = "/home/leoj/Github_Repos/SCORE/csv_dataset/S" +
                  std::to_string(scene_id) + "/";
  else
    data_folder = "/home/leoj/Github_Repos/SCORE/csv_dataset/S" +
                  std::to_string(scene_id) + "_pred/";
  vector<string> query_image_list;
  // read text file and push to query_image_list
  ifstream query_txt(data_folder + "query.txt");
  string line;
  while (getline(query_txt, line)) {
    // remove .jpg
    line = line.substr(0, line.size() - 4);
    query_image_list.push_back(line);
  }
  query_txt.close();
  std::cout << "Loaded " << query_image_list.size() << " query images"
            << std::endl;

  // rotation params
  double branch_reso_r = M_PI / 512;
  double epsilon_r = 0.015;
  double sample_reso = M_PI / 256;
  double prox_thres_r = branch_reso_r;
  bool use_saturated = true;
  double q_value_r = 0.9;
  if (use_gt_labels)
    q_value_r = 0.9;
  else
    q_value_r = 0.5;
  double side_length = M_PI / side_length_divide;
  RotFGO solver_r(branch_reso_r, epsilon_r, sample_reso, prox_thres_r,
                use_saturated, q_value_r);
  
  // translation params
  double branch_reso_t = 0.02;                // terminate bnb when branch size <= resolution
  double prox_thres_t = 0.02;                 // proximity threshold for clustering
  double epsilon_t = 0.03;                    // error tolerance for translation
  Eigen::Vector3d space_size = room_sizes[scene_id - 1];
  TransFGO solver_t(branch_reso_t, epsilon_t, prox_thres_t, space_size);

  // Relocalize one query image
  auto lines3D_data = helper::readCSV(data_folder + "3Dlines.csv");

  string query_image = query_image_list[image_index];
  // read csv data into vector<vector<double>>
  auto intrinsic_data =
      helper::readCSV(data_folder + "intrinsics/" + query_image + ".csv");
  auto gt_pose_data =
      helper::readCSV(data_folder + "poses/" + query_image + ".csv");
  auto lines2D_data =
      helper::readCSV(data_folder + "lines2D/" + query_image + "_2Dlines.csv");
  
  //remove lines without semantic label
  for (int i = 0; i < lines2D_data.size(); i++)
    if (lines2D_data[i][3] == 0)
    {
      lines2D_data.erase(lines2D_data.begin() + i);
      i--;
    }

  auto retrived_3D_line_idx = helper::readCSV(
      data_folder + "retrived_3D_line_idx/" + query_image + ".csv");
  auto retrived_closest_pose =
      helper::readCSV(data_folder + "retrived_closest_pose/" + query_image +
                      "_retrived_pose.csv");

  // image retrieval
  vector<vector<double>> sub_lines3D_data;
  for (int i = 0; i < retrived_3D_line_idx.size(); i++)
    sub_lines3D_data.push_back(lines3D_data[retrived_3D_line_idx[i][0]]);

  // retrived rotation
  Eigen::Matrix3d R_retrived;
  R_retrived << retrived_closest_pose[0][0], retrived_closest_pose[0][1],
      retrived_closest_pose[0][2], retrived_closest_pose[1][0],
      retrived_closest_pose[1][1], retrived_closest_pose[1][2],
      retrived_closest_pose[2][0], retrived_closest_pose[2][1],
      retrived_closest_pose[2][2];
  auto angles_retrived = helper::rot2angle(R_retrived.transpose());
  vector<vector<double>> branches = helper::confine_sphere(
      angles_retrived[0], angles_retrived[1], side_length, 3 * M_PI / 180);
  vector<RBranch> initial_branches;
  for (auto branch : branches)
    initial_branches.push_back(
        RBranch(branch[0], branch[1], branch[2], branch[3], -1, -1));

  // extract camera intrinsics
  Eigen::Matrix3d intrinsic_matrix; // fx, 0, cx, 0, fy, cy, 0, 0, 1
  intrinsic_matrix << intrinsic_data[0][0], 0, intrinsic_data[0][1], 0,
      intrinsic_data[0][2], intrinsic_data[0][3], 0, 0, 1;
  // extract ground true pose
  Eigen::Matrix3d R_gt; // rotation matrix
  Eigen::Vector3d t_gt; // translation vector
  R_gt << gt_pose_data[0][0], gt_pose_data[0][1], gt_pose_data[0][2],
      gt_pose_data[1][0], gt_pose_data[1][1], gt_pose_data[1][2],
      gt_pose_data[2][0], gt_pose_data[2][1], gt_pose_data[2][2];
  t_gt << gt_pose_data[0][3], gt_pose_data[1][3], gt_pose_data[2][3];

  // Process 2D lines: normalize normal vectors
  std::vector<std::vector<double>> processed_lines2D = lines2D_data;
  for (auto &line : processed_lines2D) {
    Eigen::Vector3d normal(line[0], line[1], line[2]);
    normal = normal.transpose() * intrinsic_matrix;
    normal.normalize();
    line[0] = normal(0);
    line[1] = normal(1);
    line[2] = normal(2);
  }

  // semantic matching
  std::vector<int> ids;
  std::vector<Eigen::Vector3d> n_2D, v_3D, endpoints_3D;
  std::cout << "Performing semantic matching..." << std::endl;
  int associated_2D_line_num = helper::matchLines(processed_lines2D, sub_lines3D_data, ids, n_2D, v_3D,
                     endpoints_3D);
  if (associated_2D_line_num < 5)
  {
    std::cout << "Query image " << query_image << " has less than 5 associated 2D lines, skip." << std::endl;
    return 0;
  }

  // --- Rot Estimation ---
  std::cout << "\n--- Rotation Estimation ---" << std::endl;

  // Create solver_r and solve
  auto start_time = std::chrono::high_resolution_clock::now();
  std::vector<Eigen::Matrix3d> R_candidates =
      solver_r.solve(n_2D, v_3D, ids, initial_branches);
  auto end_time = std::chrono::high_resolution_clock::now();

  double solve_time =
      std::chrono::duration<double>(end_time - start_time).count();

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

Eigen::Vector3d t_fine_tuned = solver_t.solve(ids, R_opt, v_3D, n_2D, endpoints_3D,
                                                    epsilon_r, intrinsic_matrix);
auto trans_end_time = std::chrono::high_resolution_clock::now();

double trans_solve_time =
    std::chrono::duration<double>(trans_end_time - trans_start_time).count();

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

  return 0;
}