#ifndef ROTFGO_H
#define ROTFGO_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>
#include <chrono>
#include <vector>

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
};

class RotFGO
{
public:
  RotFGO();
  ~RotFGO();

  // Main solving function
  std::vector<Eigen::Matrix3d>
  solve(const std::vector<Eigen::Vector3d> &vector_n,
        const std::vector<Eigen::Vector3d> &vector_v,
        const std::vector<int> &ids, const Eigen::MatrixXd &kernel_buffer,
        double branch_resolution = M_PI / 256.0, double epsilon_r = 0.015,
        double sample_resolution = M_PI / 256.0,
        double prox_threshold = M_PI / 180.0, int west_or_east = 2);

  // Get timing information
  double getLastSolveTime() const { return last_solve_time_; }
  int getNumCandidates() const { return num_candidates_; }

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
                  double epsilon, double sample_resolution,
                  const std::vector<int> &ids,
                  const Eigen::MatrixXd &kernel_buffer, double prox_threshold);

  // Interval stabbing
  std::pair<double, std::vector<double>> saturatedIntervalStabbing(
      const std::vector<double> &intervals, const std::vector<int> &ids,
      const Eigen::MatrixXd &kernel_buffer, double prox_threshold);

  // Helper functions for coordinate conversion
  Eigen::Vector3d polarToXyz(double alpha, double phi);
  std::pair<double, double> xyzToPolar(const Eigen::Vector3d &axis);

  // Helper functions for normal calculation
  std::tuple<Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector2d>
  calculateNormals(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2);

  // Branch subdivision
  std::vector<Branch> subBranch(const Branch &branch);

  // Interval calculation functions
  std::vector<double> lowerInterval(double A, double phi, double constant,
                                    double epsilon);
  std::vector<double> upperInterval(double A_1, double phi_1,
                                    double const_1, double A_2,
                                    double phi_2, double const_2,
                                    double epsilon);

  // H1 and H2 interval mapping
  std::pair<std::vector<double>, std::vector<double>>
  h1IntervalMapping(const LinePairData &line_pair_data, const Branch &branch,
                    double sample_resolution);
  std::pair<std::vector<double>, std::vector<double>>
  h2IntervalMapping(const LinePairData &line_pair_data, const Branch &branch,
                    double sample_resolution);

  // Parameter calculation
  std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
  calculateParams(const std::vector<double> &inner_product,
                  const std::vector<double> &h1, const std::vector<double> &h2);

  // Clustering
  std::vector<double> clusterStabber(const std::vector<double> &theta,
                                     double prox_threshold);

  // Rotation matrix from axis-angle
  Eigen::Matrix3d axisAngleToRotMatrix(const Eigen::Vector3d &axis,
                                       double angle);

  // Member variables
  double last_solve_time_;
  int num_candidates_;
  bool verbose_;
};

#endif // ROTFGO_H
