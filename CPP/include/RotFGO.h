/*
FGO-PnL Rotation Solver

Author:  Qingcheng Zeng <qzeng450@connect.hkust-gz.edu.cn>
         Haodong Jiang  <221049033@link.cuhk.edu.cn>
License: MIT
*/

#ifndef ROTFGO_H
#define ROTFGO_H

#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <queue>
#include "helper.h"

// store data for a pair of associated 2D/3D lines
struct LinePairData
{
  std::vector<Eigen::Vector3d> vector_n;
  std::vector<Eigen::Vector3d> vector_v;
  std::vector<Eigen::Vector3d> outer_product;
  std::vector<double> inner_product;
  std::vector<Eigen::Vector3d> vector_outer_east;
  std::vector<Eigen::Vector3d> vector_outer_west;
  std::vector<bool> outer_product_belong;
  std::vector<Eigen::Vector2d> outer_east;
  std::vector<Eigen::Vector2d> outer_west;
  std::vector<Eigen::Vector3d> vector_normal_east;
  std::vector<Eigen::Vector3d> vector_normal_west;
  std::vector<Eigen::Vector3d> vector_o_normal_east;
  std::vector<Eigen::Vector3d> vector_o_normal_west;
  std::vector<Eigen::Vector2d> normal_east;
  std::vector<Eigen::Vector2d> normal_west;
  std::vector<Eigen::Vector2d> o_normal_east;
  std::vector<Eigen::Vector2d> o_normal_west;
  std::vector<double> outer_norm;
  int size;
};

// Comparator for priority queue
// store data for a branch during rotation BnB
struct RBranch
{
  double alpha_min, phi_min, alpha_max, phi_max;
  double upper_bound, lower_bound;

  RBranch()
      : alpha_min(0), phi_min(0), alpha_max(0), phi_max(0), upper_bound(-1),
        lower_bound(-1) {}
  RBranch(double a_min, double p_min, double a_max, double p_max)
      : alpha_min(a_min), phi_min(p_min), alpha_max(a_max), phi_max(p_max),
        upper_bound(-1), lower_bound(-1) {}
  RBranch(double a_min, double p_min, double a_max, double p_max,
          double u_bound, double l_bound)
      : alpha_min(a_min), phi_min(p_min), alpha_max(a_max), phi_max(p_max),
        upper_bound(u_bound), lower_bound(l_bound) {}

  // Get the size of the branch (for tie-breaking)
  constexpr double size() const noexcept
  {
    return alpha_max - alpha_min;
  }

  std::vector<RBranch> subDivide() const
  {
    // Compute midpoints for alpha and phi
    double alpha_mid = 0.5 * (alpha_min + alpha_max);
    double phi_mid = 0.5 * (phi_min + phi_max);
    return {
        {alpha_mid, phi_mid, alpha_max, phi_max}, // upper-right
        {alpha_min, phi_mid, alpha_mid, phi_max}, // upper-left
        {alpha_mid, phi_min, alpha_max, phi_mid}, // lower-right
        {alpha_min, phi_min, alpha_mid, phi_mid}  // lower-left
    };
  }

  // by upper_bound (descending), then by size (descending)
  struct Comparator
  {
    inline bool operator()(const RBranch &a, const RBranch &b) const noexcept
    {
      const double diff = a.upper_bound - b.upper_bound;
      if (std::abs(diff) < 1e-10)
      {
        return a.size() < b.size(); // larger branch size first
      }
      return diff < 0; // higher upper bound first
    }
  };
};

class RotFGO
{
public:
  using BranchQueue = std::priority_queue<RBranch, std::vector<RBranch>, RBranch::Comparator>;
  // configuration structure for RotFGO parameters
  struct RotFGOConfig
  {
    double branch_resolution;  // stop splitting a branch when its size is less than this
    double epsilon_r;          // error tolerance
    double sample_resolution;  // control granuity of interval analysis
    double prox_threshold;     // cluster similar stabbers
    RotFGOConfig(double branch_res, double eps, double sample_res,
                 double prox_thresh)
        : branch_resolution(branch_res), epsilon_r(eps), sample_resolution(sample_res),
          prox_threshold(prox_thresh){}
  };

  // struct SolverStats
  // {
  //   int total_iterations = 0;
  //   int branches_processed = 0;
  //   int branches_pruned = 0;
  //   double bound_calculation_time = 0.0;
  //   double pruning_time = 0.0;
  //   double total_time = 0.0;

  //   void reset()
  //   {
  //     total_iterations = branches_processed = branches_pruned = 0;
  //     bound_calculation_time = pruning_time = total_time = 0.0;
  //   }
  // };

RotFGO(const RotFGOConfig &config)
      : branch_resolution_(config.branch_resolution), epsilon_r_(config.epsilon_r),
        sample_resolution_(config.sample_resolution), prox_threshold_(config.prox_threshold), 
        pF_buffer(nullptr) {}

  // Legacy constructor for backward compatibility
  RotFGO(double branch_resolution, double epsilon_r,
         double sample_resolution, double prox_threshold)
      : branch_resolution_(branch_resolution), epsilon_r_(epsilon_r),
        sample_resolution_(sample_resolution), prox_threshold_(prox_threshold),
        pF_buffer(nullptr) {}
  
  ~RotFGO() {}

  std::vector<Eigen::Matrix3d>
  solve(const std::vector<Eigen::Vector3d> &vector_n,
        const std::vector<Eigen::Vector3d> &vector_v,
        const std::vector<int> &ids,
        const std::vector<RBranch> &initial_branches);

  void setSatBufferFunc(pF_Buffer_Gen pF_buffer_gen)
  {
    pF_buffer = pF_buffer_gen;
  }

private:
  LinePairData dataProcess(const std::vector<Eigen::Vector3d> &vector_n,
                           const std::vector<Eigen::Vector3d> &vector_v);
  std::tuple<Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector2d>
  calculateNormals(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2);

  void updateBestSolution(const RBranch &branch,
                          const std::vector<double> &theta_candidates,
                          double &best_lower_bound,
                          std::vector<Eigen::Vector3d> &best_axes,
                          std::vector<double> &best_angles);

  void pruneBranchQueue(BranchQueue &branch_queue,
                        double best_lower_bound);

  std::vector<double> calcBounds(const LinePairData &line_pair_data, RBranch &branch,
                                 const std::vector<int> &ids,
                                 const Eigen::MatrixXd &kernel_buffer);

  std::vector<double> lowerInterval(double A, double phi, double constant);
  std::vector<double> upperInterval(double A_1, double phi_1,
                                    double const_1, double A_2,
                                    double phi_2, double const_2);

  std::pair<std::vector<double>, std::vector<double>>
  h1IntervalMapping(const LinePairData &line_pair_data, const RBranch &branch);

  std::pair<std::vector<double>, std::vector<double>>
  h2IntervalMapping(const LinePairData &line_pair_data, const RBranch &branch);

  std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
  calcIntervalParams(const std::vector<double> &inner_product,
                     const std::vector<double> &h1, const std::vector<double> &h2);

  // hyperparameters
  double branch_resolution_;
  double epsilon_r_;
  double sample_resolution_;
  double prox_threshold_;
  double eps = 1e-10;
  // function pointer to saturation function buffer generator
  pF_Buffer_Gen pF_buffer;
};

#endif // ROTFGO_H
