/*
Full relocalization pipeline for one query image (Sat-CM v.s. CM)

Author:  Haodong Jiang  <221049033@link.cuhk.edu.cn>
License: MIT
*/

#include "RotFGO.h"
#include "TransFGO.h"
#include "helper.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <ostream>

using namespace std;
vector<string> scene_names = {"S1(workstation)", "S2(office)", "S3(game bar)", "S4(art room)"};
vector<Eigen::Vector3d> room_sizes = {Eigen::Vector3d(8, 6, 4),
                                      Eigen::Vector3d(7, 7, 3),
                                      Eigen::Vector3d(10.5, 5, 3.5),
                                      Eigen::Vector3d(10.5, 6, 3.0)};
int parse_args(int argc, char **argv, int &scene_id, bool &use_gt_labels, int &side_length_divide, int &image_index);
int main(int argc, char **argv) {
  // parse arguments
  int scene_id, side_length_divide, image_index;
  bool use_gt_labels;
  auto ret = parse_args(argc, argv, scene_id, use_gt_labels, side_length_divide, image_index);
  if (ret != 0) // parse arguments error
    return ret;

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
  
  string query_image_name = query_image_list[image_index];
  std::cout << "Chosen image: scene " << scene_names[scene_id - 1] << " "<< query_image_name << std::endl;
  if (side_length_divide > 0)
    std::cout << "Rotation axis search space: subcube with side length: PI/" << side_length_divide << std::endl;  
  else
    std::cout << "Rotation axis search space: whole sphere" << std::endl;
  if (use_gt_labels)
    std::cout << "Use gt labels" << std::endl;
  else
    std::cout << "Use predicted labels" << std::endl;

  // rotation configuation
  double branch_reso_r = M_PI / 512;   // terminate branching a branch if its size <= resolution        
  double sample_reso = M_PI / 256;     // resolution for interval analysis
  double prox_thres_r = branch_reso_r; // proximity threshold for clustering stabbers
    double epsilon_r = 0.015;          // rotation error tolerance
  double u_r = 1;
  double q_value_r;
    // saturation function buffer generator
  if (use_gt_labels)
      q_value_r = 0.9;
  else
    q_value_r = 0.5;
  pF_Buffer_Gen pF_MLBuffer = [q_value_r, epsilon_r,u_r](const std::vector<int>& ids) {
  return helper::createMLBuffer(ids, q_value_r, epsilon_r, u_r);};
  RotFGO solver_r(branch_reso_r, epsilon_r, sample_reso, prox_thres_r);
  solver_r.setSatBufferFunc(pF_MLBuffer);

  // translation configuration
  double branch_reso_t = 0.02;                // terminate branching a branch if its size <= resolution   
  double prox_thres_t = branch_reso_t;        // proximity threshold for clustering stabbers
  double epsilon_t = 0.03;                    // error tolerance for translation
  Eigen::Vector3d space_size = room_sizes[scene_id - 1];
  pF_Buffer_Gen pF_TRBuffer = helper::createTRBuffer;
  TransFGO solver_t(branch_reso_t, epsilon_t, prox_thres_t, space_size);
  solver_t.setSatBufferFunc(pF_TRBuffer);

  // load data for this query image
  auto lines3D_data = helper::readCSV<double>(data_folder + "3Dlines.csv");
  helper::ImageData image_data;
  helper::readData(data_folder, image_data, query_image_name);

  //remove lines without semantic label
  for (size_t i = 0; i < image_data.lines2D_data.size(); i++)
    if (image_data.lines2D_data[i][3] == 0)
    {
      image_data.lines2D_data.erase(image_data.lines2D_data.begin() + i);
      i--;
    }

  vector<vector<double>> sub_lines3D_data;
  for (size_t i = 0; i < image_data.retrived_3D_line_idx.size(); i++)
    sub_lines3D_data.push_back(lines3D_data[image_data.retrived_3D_line_idx[i]]);

  // retrived rotation
  auto angles_retrived = helper::rot2angle(image_data.R_retrived.transpose());
  vector<RBranch> initial_branches;
  if (side_length_divide > 0)
  {
    double side_length = M_PI / side_length_divide;
    vector<vector<double>>  branches = helper::confine_sphere(
      angles_retrived[0], angles_retrived[1], side_length, 3 * M_PI / 180);
    for (auto branch : branches)
    initial_branches.push_back(
        RBranch(branch[0], branch[1], branch[2], branch[3], -1, -1));
  }
  else // search rotation axis on the whole sphere
  {
    initial_branches.push_back(RBranch(0, 0, M_PI, M_PI));        // East hemisphere
    initial_branches.push_back(RBranch(0, M_PI, M_PI, 2 * M_PI)); // West hemisphere
  }

  // semantic matching
  std::vector<int> ids;
  std::vector<Eigen::Vector3d> n_2D, v_3D, endpoints_3D;
  std::cout << "Performing semantic matching..." << std::endl;
  int associated_2D_line_num = helper::matchLines(image_data.lines2D_data, sub_lines3D_data, ids, n_2D, v_3D,
                     endpoints_3D);
  if (associated_2D_line_num < 5)
  {
    std::cout << "Query image " << query_image_name << " has less than 5 associated 2D lines, skip." << std::endl;
    return 0;
  }
  std::cout << "\n--- Rotation Estimation ---" << std::endl;

  // Create solver_r and solve
  auto start_time = std::chrono::high_resolution_clock::now();
  std::vector<Eigen::Matrix3d> R_candidates =
      solver_r.solve(n_2D, v_3D, ids, initial_branches);
  auto end_time = std::chrono::high_resolution_clock::now();

  double solve_time =
      std::chrono::duration<double>(end_time - start_time).count();
  std::cout << "Number of BnB rotation candidates: " << R_candidates.size()
            << std::endl;
  std::cout << "Rotation Solve time: " << solve_time << " seconds" << std::endl;

  // translation estimation based on each rotation candidate
  // select the best one according to achieved score
  std::cout << "\n--- Translation Estimation ---" << std::endl;
  auto trans_start_time = std::chrono::high_resolution_clock::now();
  double best_trans_score = -1;
  std::vector<Eigen::Matrix3d> best_R;
  std::vector<Eigen::Vector3d> best_t;
  for (size_t i = 0; i < R_candidates.size(); i++)
  {
    if (R_candidates.size()>1)
      std::cout<<"estimate based on rot candidaties No. "<<i+1 <<std::endl;
    auto [score, t_fine_tuned] = solver_t.solve(ids, R_candidates[i], v_3D, n_2D, endpoints_3D,
                                                    epsilon_r, image_data.intrinsic_matrix);
    if (score > best_trans_score+1e-8)
    {
      best_trans_score = score;
      best_R.clear();
      best_t.clear();
      best_R.push_back(R_candidates[i]);
      best_t.push_back(t_fine_tuned);
    }
    else if (std::abs(score - best_trans_score) < 1e-8)
    {
      best_R.push_back(R_candidates[i]);
      best_t.push_back(t_fine_tuned);
    }
  }
auto trans_end_time = std::chrono::high_resolution_clock::now();
double trans_solve_time =
    std::chrono::duration<double>(trans_end_time - trans_start_time).count();
std::cout << "Translation solve time: " << trans_solve_time << " seconds" << std::endl;


// Summary output
double angle_error = -std::numeric_limits<double>::infinity();
double trans_error = -std::numeric_limits<double>::infinity();
for (size_t i = 0; i < best_R.size(); i++)
{
  Eigen::Matrix3d R_error = best_R[i] * image_data.R_gt.transpose();
  double trace_R = R_error.trace();
  double angle_error_i =
      std::acos(std::max(-1.0, std::min(1.0, (trace_R - 1.0) / 2.0)));
  if (angle_error_i > angle_error)
    angle_error = angle_error_i;
}

for (size_t i = 0; i < best_t.size(); i++)
{
  double trans_error_i = (best_t[i] - image_data.t_gt).norm();
  if (trans_error_i > trans_error)
    trans_error = trans_error_i;
}
//
double total_time = solve_time + trans_solve_time;
std::cout << "\n=== Summary ===" << std::endl;
std::cout << "Total pipeline time: " << total_time << " seconds" << std::endl;
std::cout << "# final rot candidates: " << best_R.size() << ", with maximum err: " 
          << angle_error * 180.0 / M_PI << " degrees" << std::endl;
std::cout << "# final trans candidates: " << best_t.size() << ", with maximum err: " 
          << trans_error << " meters" << std::endl;
return 0;
}

int parse_args(int argc, char **argv, int &scene_id, bool &use_gt_labels, int &side_length_divide, int &image_index)
{
  if (argc != 5) {
    std::cerr << "Usage:" << argv[0] << " 1 y 2 4 (choose S1, use gt labels, divide side length pi by 2, image index 2)" << std::endl;
    return -1;
  }
  try {
    scene_id = std::stoi(argv[1]);
    if (scene_id < 1 || scene_id > 4) {
      std::cerr << "Error: scene_id must be between 1 and 4" << std::endl;
      return -1;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error parsing scene_id: " << argv[1] << std::endl;
    return -1;
  }
  
  std::string gt_arg = std::string(argv[2]);
  if (gt_arg == "y" || gt_arg == "Y") {
    use_gt_labels = true;
  } else if (gt_arg == "n" || gt_arg == "N") {
    use_gt_labels = false;
  } else {
    std::cerr << "Error: second argument must be 'y' or 'n', got: " << argv[2] << std::endl;
    return -1;
  }
  
  try {
    side_length_divide = std::stoi(argv[3]);
    if (side_length_divide < 0) {
      std::cerr << "Error: side_length_divide must be non-negative" << std::endl;
      return -1;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error parsing side_length_divide: " << argv[3] << std::endl;
    return -1;
  }
  try {
    image_index = std::stoi(argv[4]);
    if (image_index < 0) {
      std::cerr << "Error: image_index must be non-negative" << std::endl;
      return -1;
    }
    if (scene_id == 1 && image_index>103){
      std::cerr << "Error: image_index must be <=103 for scene S1" << std::endl;
      return -1;
    }
    if (scene_id == 2 && image_index>68){
      std::cerr << "Error: image_index must be <=68 for scene S2" << std::endl;
      return -1;
    }
    if (scene_id == 3 && image_index>80){
      std::cerr << "Error: image_index must be <=80 for scene S3" << std::endl;
      return -1;
    }
    if (scene_id == 4 && image_index>148){
      std::cerr << "Error: image_index must be <=148 for scene S4" << std::endl;
      return -1;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error parsing image_index: " << argv[4] << std::endl;
    return -1;
  }
  return 0;
}
