#ifndef ROTFGO_H
#define ROTFGO_H

#include <cmath>
#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <queue>

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

struct Branch
{
  double alpha_min, phi_min, alpha_max, phi_max;
  double upper_bound, lower_bound;

  Branch()
      : alpha_min(0), phi_min(0), alpha_max(0), phi_max(0), upper_bound(-1),
        lower_bound(-1) {}
  Branch(double a_min, double p_min, double a_max, double p_max)
      : alpha_min(a_min), phi_min(p_min), alpha_max(a_max), phi_max(p_max),
        upper_bound(-1), lower_bound(-1) {}

  // Get the size of the branch (for tie-breaking)
  double size() const
  {
    return alpha_max - alpha_min;
  }

  // Subdivide this branch into four sub-branches
  std::vector<Branch> subdivide() const
  {
    // Compute midpoints for alpha and phi
    double alpha_mid = 0.5 * (alpha_min + alpha_max);
    double phi_mid = 0.5 * (phi_min + phi_max);

    std::vector<Branch> sub_branches;
    sub_branches.emplace_back(alpha_mid, phi_mid, alpha_max, phi_max); // upper-right
    sub_branches.emplace_back(alpha_min, phi_mid, alpha_mid, phi_max); // upper-left
    sub_branches.emplace_back(alpha_mid, phi_min, alpha_max, phi_mid); // lower-right
    sub_branches.emplace_back(alpha_min, phi_min, alpha_mid, phi_mid); // lower-left

    return sub_branches;
  }

  // Embedded comparator for priority queue
  // Prioritize by upper_bound (descending), then by size (descending)
  struct Comparator
  {
    bool operator()(const Branch &a, const Branch &b) const
    {
      if (std::abs(a.upper_bound - b.upper_bound) < 1e-10)
      {
        // If upper bounds are equal, prioritize larger branch size
        return a.size() < b.size();
      }
      // Prioritize higher upper bound
      return a.upper_bound < b.upper_bound;
    }
  };
};

class RotFGO
{
  using BranchQueue = std::priority_queue<Branch, std::vector<Branch>, Branch::Comparator>;

public:
  RotFGO(double branch_resolution = M_PI / 256.0, double epsilon_r = 0.015,
         double sample_resolution = M_PI / 256.0, double prox_threshold = M_PI / 180.0);
  ~RotFGO();

  // Main solving function
  std::vector<Eigen::Matrix3d>
  solve(const std::vector<Eigen::Vector3d> &vector_n,
        const std::vector<Eigen::Vector3d> &vector_v,
        const std::vector<int> &ids, const Eigen::MatrixXd &kernel_buffer,
        int west_or_east = 2);

  // Kernel buffer creation
  static Eigen::MatrixXd createKernelBuffer(const std::vector<int> &ids,
                                             bool use_saturated = true);

private:
  // Data preprocessing
  LinePairData dataProcess(const std::vector<Eigen::Vector3d> &vector_n,
                           const std::vector<Eigen::Vector3d> &vector_v);

  // Bound calculation
  std::tuple<double, double, std::vector<double>>
  calculateBounds(const LinePairData &line_pair_data, const Branch &branch,
                  const std::vector<int> &ids,
                  const Eigen::MatrixXd &kernel_buffer);

  // Interval stabbing
  std::pair<double, std::vector<double>> saturatedIntervalStabbing(
      const std::vector<double> &intervals, const std::vector<int> &ids,
      const Eigen::MatrixXd &kernel_buffer);

  // Helper functions for coordinate conversion
  Eigen::Vector3d polarToXyz(double alpha, double phi);
  std::pair<double, double> xyzToPolar(const Eigen::Vector3d &axis);

  // Helper functions for normal calculation
  std::tuple<Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector2d>
  calculateNormals(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2);

  // Interval calculation functions
  std::vector<double> lowerInterval(double A, double phi, double constant);
  std::vector<double> upperInterval(double A_1, double phi_1,
                                    double const_1, double A_2,
                                    double phi_2, double const_2);

  // H1 and H2 interval mapping
  std::pair<std::vector<double>, std::vector<double>>
  h1IntervalMapping(const LinePairData &line_pair_data, const Branch &branch);
  std::pair<std::vector<double>, std::vector<double>>
  h2IntervalMapping(const LinePairData &line_pair_data, const Branch &branch);

  // Parameter calculation
  std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
  calculateParams(const std::vector<double> &inner_product,
                  const std::vector<double> &h1, const std::vector<double> &h2);

  // Clustering
  std::vector<double> clusterStabber(const std::vector<double> &theta);

  // Algorithm parameters stored as member variables
  double branch_resolution_;
  double epsilon_r_;
  double sample_resolution_;
  double prox_threshold_;
};

#endif // ROTFGO_H
