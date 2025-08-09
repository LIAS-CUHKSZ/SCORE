/*
FGO-PnL Rotation Solver

Author:  Qingcheng Zeng <qzeng450@connect.hkust-gz.edu.cn>
         Haodong Jiang  <221049033@link.cuhk.edu.cn>
License: MIT
*/
#include "RotFGO.h"
#include "SatIS.h"
#include <algorithm>
#include <cmath>
#include <future>
#include <array>
#include <iostream>

constexpr double PI = M_PI;

std::vector<Eigen::Matrix3d>
RotFGO::solve(const std::vector<Eigen::Vector3d> &vector_n,
              const std::vector<Eigen::Vector3d> &vector_v,
              const std::vector<int> &ids,
              const std::vector<RBranch> &initial_branches)
{
  if (pF_buffer == nullptr)
  {
    std::cerr << "Error: Saturation Function for rotation is not set" << std::endl;
    return std::vector<Eigen::Matrix3d>();
  }
  Eigen::MatrixXd sat_buffer = pF_buffer(ids);
  // Step 1: Pre-compute line pair data
  LinePairData line_pair_data = dataProcess(vector_n, vector_v);

  // Step 2: Initialize branch-and-bound process
  BranchQueue bq;

  double best_lb = -1.0;
  std::vector<Eigen::Vector3d> u_best;
  std::vector<double> theta_best;
  int iter = 0;

  // Step 3: Initialize with user-provided branches
  for (auto branch : initial_branches)
  {
    auto theta_candidates = calcBounds(line_pair_data, branch, ids, sat_buffer);
    updateBestSolution(branch, theta_candidates, best_lb, u_best, theta_best);
    bq.push(branch);
  }

  // Step 4: Main branch-and-bound loop
  while (!bq.empty())
  {
    // Get up to 2 branches from the queue for asynchronous processing
    std::vector<RBranch> cur_bs;

    // Pop first branch
    cur_bs.emplace_back(std::move(const_cast<RBranch &>(bq.top())));
    bq.pop();

    // Pop second branch if available and if first branch is large enough
    // In order to process at most 8 sub-branches asynchronously and speed up
    if (!bq.empty() && cur_bs[0].size() + this->eps>= branch_resolution_)
    {
      cur_bs.emplace_back(std::move(const_cast<RBranch &>(bq.top())));
      bq.pop();
    }

    // Process all branches and their sub-branches asynchronously
    std::vector<std::future<std::vector<double>>> futures;
    std::vector<RBranch> sub_b;

    for (const auto &b : cur_bs)
    {
      if (b.size() + this->eps>= branch_resolution_)
      {
        std::vector<RBranch> sub_branches = b.subDivide();
        sub_b.insert(sub_b.end(), sub_branches.begin(), sub_branches.end());
      }
    }

    // Create async tasks for all sub-branches
    for (size_t i = 0; i < sub_b.size(); ++i)
    {
      futures.push_back(std::async(std::launch::async,
                                   [this, &line_pair_data, &sub_b, &ids, &sat_buffer, i]()
                                   {
                                     return calcBounds(line_pair_data, sub_b[i], ids, sat_buffer);
                                   }));
    }

    // Process results from all async tasks
    for (size_t i = 0; i < futures.size(); ++i)
    {
      ++iter;
      auto theta_candidates = futures[i].get();

      updateBestSolution(sub_b[i], theta_candidates, best_lb,
                         u_best, theta_best);

      //stop further splitting if U/L bounds meet, and current cube attains the best lower bound.
      if (sub_b[i].lower_bound+this->eps > best_lb && sub_b[i].upper_bound < best_lb + this->eps) 
      // numerically more stable than sub_b[i].lower_bound == best_lb && sub_b[i].upper_bound == best_lb
      {
        continue;
      }
      // continue splitting if the current cube has potential to improve best_lb
      if (sub_b[i].upper_bound - this->eps > best_lb)
      {
        bq.push(sub_b[i]);
      }
    }
    pruneBranchQueue(bq, best_lb);
  }

  // Step 5: Generate rotation matrices
  size_t res_sz = u_best.size();
  std::vector<Eigen::Matrix3d> R_opt(res_sz);
  for (size_t i = 0; i < res_sz; i++)
  {
    Eigen::AngleAxisd axis_angle(theta_best[i], u_best[i].normalized());
    R_opt[i] = axis_angle.toRotationMatrix().transpose();
  }
  return R_opt;
}

LinePairData RotFGO::dataProcess(const std::vector<Eigen::Vector3d> &vector_n,
                                 const std::vector<Eigen::Vector3d> &vector_v)
{
  LinePairData data;
  size_t N = vector_n.size();
  data.size = N;
  data.vector_n = vector_n;
  data.vector_v = vector_v;
  data.outer_product.resize(N);
  data.inner_product.resize(N);
  data.vector_outer_east.resize(N);
  data.vector_outer_west.resize(N);
  data.outer_product_belong.resize(N);
  data.outer_east.resize(N);
  data.outer_west.resize(N);
  data.vector_normal_east.resize(N);
  data.vector_normal_west.resize(N);
  data.vector_o_normal_east.resize(N);
  data.vector_o_normal_west.resize(N);
  data.normal_east.resize(N);
  data.normal_west.resize(N);
  data.o_normal_east.resize(N);
  data.o_normal_west.resize(N);
  data.outer_norm.resize(N);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(4)
#endif
  for (size_t i = 0; i < N; i++)
  {
    const Eigen::Vector3d &n = vector_n[i];
    const Eigen::Vector3d &v = vector_v[i];

    // Outer product: cross(v, n) as in MATLAB
    data.outer_product[i] = v.cross(n);
    data.outer_product_belong[i] = (data.outer_product[i](1) >= 0);

    if (data.outer_product_belong[i])
    {
      data.vector_outer_east[i] = data.outer_product[i];
      data.vector_outer_west[i] = -data.outer_product[i];
    }
    else
    {
      data.vector_outer_east[i] = -data.outer_product[i];
      data.vector_outer_west[i] = data.outer_product[i];
    }

    auto [alpha, phi] = helper::xyzToPolar(data.outer_product[i]);
    if (phi > PI)
    {
      data.outer_east[i] = Eigen::Vector2d(PI - alpha, phi - PI);
      data.outer_west[i] = Eigen::Vector2d(alpha, phi);
    }
    else
    {
      data.outer_east[i] = Eigen::Vector2d(alpha, phi);
      data.outer_west[i] = Eigen::Vector2d(PI - alpha, phi + PI);
    }

    data.inner_product[i] = v.dot(n);

    auto [normal_east, normal_west, o_normal_east, o_normal_west] =
        calculateNormals(n, v);
    data.normal_east[i] = normal_east;
    data.normal_west[i] = normal_west;
    data.o_normal_east[i] = o_normal_east;
    data.o_normal_west[i] = o_normal_west;

    data.vector_normal_east[i] = helper::polarToXyz(normal_east(0), normal_east(1));
    data.vector_normal_west[i] = helper::polarToXyz(normal_west(0), normal_west(1));
    data.vector_o_normal_east[i] =
        helper::polarToXyz(o_normal_east(0), o_normal_east(1));
    data.vector_o_normal_west[i] =
        helper::polarToXyz(o_normal_west(0), o_normal_west(1));
    data.outer_norm[i] = data.outer_product[i].norm();
  }
  return data;
}

std::tuple<Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector2d>
RotFGO::calculateNormals(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2)
{
  Eigen::Vector3d mid = (v1 + v2) / 2.0;
  Eigen::Vector2d normal_east, normal_west, o_normal_east, o_normal_west;

  if (mid.norm() < 1e-4)
  {
    normal_east = Eigen::Vector2d(0.0, 0.0);
    normal_west = Eigen::Vector2d(0.0, 0.0);

    auto [alpha_v1, phi_v1] = helper::xyzToPolar(v1);
    if (phi_v1 > PI)
    {
      o_normal_east = Eigen::Vector2d(PI - alpha_v1, phi_v1 - PI);
      o_normal_west = Eigen::Vector2d(alpha_v1, phi_v1);
    }
    else
    {
      o_normal_east = Eigen::Vector2d(alpha_v1, phi_v1);
      o_normal_west = Eigen::Vector2d(PI - alpha_v1, phi_v1 + PI);
    }
    return std::make_tuple(normal_east, normal_west, o_normal_east, o_normal_west);
  }

  mid.normalize();
  Eigen::Vector3d cross_nv = (v1.cross(v2)).normalized();
  Eigen::Vector3d orthogonal = mid.cross(cross_nv);
  orthogonal.normalize();
  auto [alpha_mid, phi_mid] = helper::xyzToPolar(mid);
  auto [alpha_orthogonal, phi_orthogonal] = helper::xyzToPolar(orthogonal);

  if (phi_mid > PI)
  {
    normal_east = Eigen::Vector2d(PI - alpha_mid, phi_mid - PI);
    normal_west = Eigen::Vector2d(alpha_mid, phi_mid);
  }
  else
  {
    normal_east = Eigen::Vector2d(alpha_mid, phi_mid);
    normal_west = Eigen::Vector2d(PI - alpha_mid, phi_mid + PI);
  }

  if (phi_orthogonal > PI)
  {
    o_normal_east = Eigen::Vector2d(PI - alpha_orthogonal, phi_orthogonal - PI);
    o_normal_west = Eigen::Vector2d(alpha_orthogonal, phi_orthogonal);
  }
  else
  {
    o_normal_east = Eigen::Vector2d(alpha_orthogonal, phi_orthogonal);
    o_normal_west = Eigen::Vector2d(PI - alpha_orthogonal, phi_orthogonal + PI);
  }
  return std::make_tuple(normal_east, normal_west, o_normal_east, o_normal_west);
}

std::vector<double> RotFGO::lowerInterval(double A, double phi, double constant)
{
  std::vector<double> interval;
  if (A == 0)
    return {};
  double c_up = -constant + epsilon_r_;
  double c_lo = -constant - epsilon_r_;
  if (c_up <= -A)
  {
    return {};
  }
  else if (c_up <= 0)
  {
    if (c_lo <= -A)
    {
      double m = std::asin(c_up / A);
      double m_l = PI - m;
      double m_r = 2 * PI + m;
      if (phi <= -m || phi >= m_r)
      {
        return {};
      }
      else
      {
        interval.emplace_back(std::max(0.0, m_l - phi));
        interval.emplace_back(std::min(PI, m_r - phi));
      }
    }
    else
    {
      double m = std::asin(c_up / A);
      double n = std::asin(c_lo / A);
      double m_l = PI - m;
      double n_l = PI - n;
      double m_r = 2 * PI + n;
      double n_r = 2 * PI + m;
      if (phi <= -m || phi >= n_r)
      {
        return {};
      }
      else if (phi <= PI + n)
      {
        interval.emplace_back(m_l - phi);
        interval.emplace_back(std::min(PI, n_l - phi));
      }
      else if (phi <= n_l)
      {
        interval.emplace_back(std::max(m_l - phi, 0.0));
        interval.emplace_back(n_l - phi);
        interval.emplace_back(m_r - phi);
        interval.emplace_back(std::min(PI, n_r - phi));
      }
      else
      {
        interval.emplace_back(std::max(m_r - phi, 0.0));
        interval.emplace_back(std::min(PI, n_r - phi));
      }
    }
  }
  else if (c_up <= A)
  {
    if (c_lo <= -A)
    {
      double m = std::asin(c_up / A);
      if (phi <= m)
      {
        interval.emplace_back(0.0);
        interval.emplace_back(m - phi);
        interval.emplace_back(PI - m - phi);
        interval.emplace_back(PI);
      }
      else if (phi <= 2 * PI - m)
      {
        interval.emplace_back(std::max(0.0, PI - m - phi));
        interval.emplace_back(std::min(PI, 2 * PI + m - phi));
      }
      else
      {
        interval.emplace_back(0.0);
        interval.emplace_back(2 * PI + m - phi);
        interval.emplace_back(3 * PI - m - phi);
        interval.emplace_back(PI);
      }
    }
    else if (c_lo <= 0)
    {
      double m = std::asin(c_up / A);
      double n = std::asin(c_lo / A);
      double m_r = PI - m;
      double n_l = PI - n;
      double n_r = 2 * PI + n;
      if (phi < m)
      {
        interval.emplace_back(0.0);
        interval.emplace_back(m - phi);
        interval.emplace_back(m_r - phi);
        interval.emplace_back(std::min(PI, n_l - phi));
      }
      else if (phi <= n_r - PI)
      {
        interval.emplace_back(std::max(0.0, m_r - phi));
        interval.emplace_back(std::min(PI, n_l - phi));
      }
      else if (phi <= n_l)
      {
        interval.emplace_back(std::max(0.0, m_r - phi));
        interval.emplace_back(n_l - phi);
        interval.emplace_back(n_r - phi);
        interval.emplace_back(std::min(PI, 2 * PI + m - phi));
      }
      else if (phi <= m_r + PI)
      {
        interval.emplace_back(std::max(0.0, n_r - phi));
        interval.emplace_back(std::min(PI, 2 * PI + m - phi));
      }
      else
      {
        interval.emplace_back(std::max(0.0, n_r - phi));
        interval.emplace_back(2 * PI + m - phi);
        interval.emplace_back(3 * PI - m - phi);
        interval.emplace_back(PI);
      }
    }
    else
    {
      double m = std::asin(c_up / A);
      double n = std::asin(c_lo / A);
      double m_r_1 = PI - m;
      double n_r_1 = PI - n;
      if (phi <= m)
      {
        interval.emplace_back(std::max(0.0, n - phi));
        interval.emplace_back(m - phi);
        interval.emplace_back(m_r_1 - phi);
        interval.emplace_back(n_r_1 - phi);
      }
      else if (phi <= PI + n && phi >= n_r_1)
      {
        return {};
      }
      else if (phi <= n_r_1)
      {
        interval.emplace_back(std::max(0.0, m_r_1 - phi));
        interval.emplace_back(n_r_1 - phi);
      }
      else if (phi <= m_r_1 + PI)
      {
        interval.emplace_back(2 * PI + n - phi);
        interval.emplace_back(std::min(PI, 2 * PI + m - phi));
      }
      else
      {
        interval.emplace_back(2 * PI + n - phi);
        interval.emplace_back(2 * PI + m - phi);
        interval.emplace_back(m_r_1 + 2 * PI - phi);
        interval.emplace_back(std::min(PI, n_r_1 + 2 * PI - phi));
      }
    }
  }
  else
  {
    if (c_lo <= -A)
    {
      interval.emplace_back(0.0);
      interval.emplace_back(PI);
    }
    else if (c_lo <= 0)
    {
      double m = std::asin(c_lo / A);
      double m_l = PI - m;
      double m_r = 2 * PI + m;
      if (phi <= m_r - PI)
      {
        interval.emplace_back(0.0);
        interval.emplace_back(std::min(m_l - phi, PI));
      }
      else if (phi >= m_l)
      {
        interval.emplace_back(std::max(0.0, m_r - phi));
        interval.emplace_back(PI);
      }
      else
      {
        interval.emplace_back(0.0);
        interval.emplace_back(m_l - phi);
        interval.emplace_back(m_r - phi);
        interval.emplace_back(PI);
      }
    }
    else if (c_lo < A)
    {
      double m = std::asin(c_lo / A);
      if (phi <= PI - m)
      {
        interval.emplace_back(std::max(0.0, m - phi));
        interval.emplace_back(PI - m - phi);
      }
      else if (phi >= PI + m)
      {
        interval.emplace_back(2 * PI + m - phi);
        interval.emplace_back(std::min(PI, 3 * PI - m - phi));
      }
      else
      {
        return {};
      }
    }
    else
    {
      return {};
    }
  }

  // filter out invalid intervals and normalize
  std::vector<double> result;
  for (size_t i = 0; i < interval.size(); i += 2)
  {
    if (i + 1 < interval.size())
    {
      double start = std::max(0.0, std::min(PI, interval[i]));
      double end = std::max(0.0, std::min(PI, interval[i + 1]));
      if (start < end)
      {
        result.emplace_back(start);
        result.emplace_back(end);
      }
    }
  }
  return result;
}

std::vector<double> RotFGO::upperInterval(double A_1, double phi_1,
                                          double const_1, double A_2,
                                          double phi_2, double const_2)
{
  // This function finds intervals satisfying both:
  // A_1*sin(theta + phi_1) + const_1 >= -epsilon
  // A_2*sin(theta + phi_2) + const_2 <= epsilon
  // Where A_1,phi_1,const_1 are upper bounds, A_2,phi_2,const_2 are lower bounds
  std::vector<double> upper_intervals, lower_intervals;

  auto intersectInterval = [](double a1, double a2, double b1, double b2) -> std::vector<double>
  {
    if (a2 < b1 || b2 < a1)
      return {};
    else
      return {std::max(a1, b1), std::min(a2, b2)};
  };

  // First constraint: A_1*sin(theta + phi_1) + const_1 >= -epsilon
  double c_lo = -const_1 - epsilon_r_;
  if (A_1 < c_lo)
  {
    return {};
  }
  else if (c_lo >= 0)
  {
    double x_l = std::asin(c_lo / A_1);
    if (phi_1 <= PI - x_l)
    {
      upper_intervals.emplace_back(std::max(0.0, x_l - phi_1));
      upper_intervals.emplace_back(PI - x_l - phi_1);
    }
    else if (phi_1 >= PI + x_l)
    {
      upper_intervals.emplace_back(2 * PI + x_l - phi_1);
      upper_intervals.emplace_back(std::min(PI, 3 * PI - x_l - phi_1));
    }
    else
    {
      return {};
    }
  }
  else if (c_lo >= -A_1)
  {
    double x = std::asin(c_lo / A_1);
    double x_l = PI - x;
    double x_r = 2 * PI + x;
    if (phi_1 <= x_r - PI)
    {
      upper_intervals.emplace_back(0.0);
      upper_intervals.emplace_back(std::min(x_l - phi_1, PI));
    }
    else if (phi_1 >= x_l)
    {
      upper_intervals.emplace_back(std::max(0.0, x_r - phi_1));
      upper_intervals.emplace_back(PI);
    }
    else
    {
      upper_intervals.emplace_back(0.0);
      upper_intervals.emplace_back(x_l - phi_1);
      upper_intervals.emplace_back(x_r - phi_1);
      upper_intervals.emplace_back(PI);
    }
  }
  else
  {
    upper_intervals.emplace_back(0.0);
    upper_intervals.emplace_back(PI);
  }

  // Second constraint: A_2*sin(theta + phi_2) + const_2 <= epsilon
  double c_up = epsilon_r_ - const_2;
  if (A_2 <= c_up)
  {
    lower_intervals.emplace_back(0.0);
    lower_intervals.emplace_back(PI);
  }
  else if (c_up >= 0)
  {
    double x_l = std::asin(c_up / A_2);
    if (phi_2 <= x_l)
    {
      lower_intervals.emplace_back(0.0);
      lower_intervals.emplace_back(x_l - phi_2);
      lower_intervals.emplace_back(PI - x_l - phi_2);
      lower_intervals.emplace_back(PI);
    }
    else if (phi_2 <= 2 * PI - x_l)
    {
      // Fixed: should be PI - x_l - phi_2, not max(0.0, PI - x_l - phi_2)
      // as shown in MATLAB: lower_interval = [ max(0,pi-x_l-phi_2);min(pi,2*pi+x_l-phi_2)];
      lower_intervals.emplace_back(std::max(0.0, PI - x_l - phi_2));
      lower_intervals.emplace_back(std::min(PI, 2 * PI + x_l - phi_2));
    }
    else
    {
      lower_intervals.emplace_back(0.0);
      lower_intervals.emplace_back(2 * PI + x_l - phi_2);
      lower_intervals.emplace_back(3 * PI - x_l - phi_2);
      lower_intervals.emplace_back(PI);
    }
  }
  else if (c_up >= -A_2)
  {
    double x = std::asin(c_up / A_2);
    double x_l = PI - x;
    double x_r = 2 * PI + x;
    if (phi_2 <= -x || phi_2 >= x_r)
    {
      return {};
    }
    else
    {
      lower_intervals.emplace_back(std::max(0.0, x_l - phi_2));
      lower_intervals.emplace_back(std::min(PI, x_r - phi_2));
    }
  }
  else
  {
    return {};
  }

  std::vector<double> result;
  int num_upper = upper_intervals.size() / 2;
  int num_lower = lower_intervals.size() / 2;

  for (int i = 0; i < num_upper; i++)
  {
    for (int j = 0; j < num_lower; j++)
    {
      auto intersection = intersectInterval(
          upper_intervals[2 * i], upper_intervals[2 * i + 1],
          lower_intervals[2 * j], lower_intervals[2 * j + 1]);
      if (!intersection.empty())
      {
        result.insert(result.end(), intersection.begin(), intersection.end());
      }
    }
  }

  // Filter out invalid intervals and ensure they're in [0, PI]
  std::vector<double> filtered_result;
  for (size_t i = 0; i < result.size(); i += 2)
  {
    if (i + 1 < result.size())
    {
      double start = std::max(0.0, std::min(PI, result[i]));
      double end = std::max(0.0, std::min(PI, result[i + 1]));
      if (start < end)
      {
        filtered_result.emplace_back(start);
        filtered_result.emplace_back(end);
      }
    }
  }

  return filtered_result;
}

std::pair<std::vector<double>, std::vector<double>>
RotFGO::h1IntervalMapping(const LinePairData &line_pair_data,
                          const RBranch &branch)
{
  size_t N = line_pair_data.size;
  std::vector<double> h1_upper(N), h1_lower(N);
  double cube_width = branch.alpha_max - branch.alpha_min;
  std::vector<double> range_alpha = {branch.alpha_min, branch.alpha_max};
  std::vector<double> range_phi = {branch.phi_min, branch.phi_max};

  auto intervalProjection = [](double a, const std::vector<double> &interval) -> std::pair<double, double>
  {
    double far, near;
    if (a < interval[0])
    {
      far = interval[1];
      near = interval[0];
    }
    else if (a <= (interval[0] + interval[1]) / 2)
    {
      far = interval[1];
      near = a;
    }
    else if (a <= interval[1])
    {
      far = interval[0];
      near = a;
    }
    else
    {
      far = interval[0];
      near = interval[1];
    }
    return {far, near};
  };

  if (cube_width <= sample_resolution_)
  {
    for (size_t i = 0; i < N; i++)
    {
      bool east = line_pair_data.outer_product_belong[i];
      int flag = ((range_phi[1] > PI && !east) || (range_phi[1] <= PI && east)) ? 1 : -1;

      Eigen::Vector2d outer_angle;
      Eigen::Vector3d x;

      if (range_phi[0] <= PI && range_phi[1] <= PI)
      {
        outer_angle = line_pair_data.outer_east[i];
        x = line_pair_data.vector_outer_east[i];
      }
      else
      {
        outer_angle = line_pair_data.outer_west[i];
        x = line_pair_data.vector_outer_west[i];
      }

      auto [phi_far, phi_near] = intervalProjection(outer_angle(1), range_phi);
      auto [alpha_far, alpha_near] = intervalProjection(outer_angle(0), range_alpha);

      // Find maximum
      double delta_phi_near = std::abs(phi_near - outer_angle(1));
      double maximum;
      if (delta_phi_near == 0)
      {
        maximum = x.dot(helper::polarToXyz(alpha_near, phi_near));
      }
      else
      {
        maximum = std::max(x.dot(helper::polarToXyz(range_alpha[0], phi_near)),
                           x.dot(helper::polarToXyz(range_alpha[1], phi_near)));
      }

      // Find minimum
      double minimum = std::min(x.dot(helper::polarToXyz(range_alpha[0], phi_far)),
                                x.dot(helper::polarToXyz(range_alpha[1], phi_far)));
      if (flag == 1)
      {
        h1_upper[i] = maximum;
        h1_lower[i] = minimum;
      }
      else
      {
        h1_upper[i] = -minimum;
        h1_lower[i] = -maximum;
      }
    }
  }
  else
  {
    // large cube case
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(4)
#endif
    for (size_t i = 0; i < N; i++)
    {
      bool east = line_pair_data.outer_product_belong[i];
      int flag = ((range_phi[1] > PI && !east) || (range_phi[1] <= PI && east)) ? 1 : -1;

      Eigen::Vector2d outer_angle;
      Eigen::Vector3d x;
      double outer_alpha, outer_phi;

      if (range_phi[0] <= PI && range_phi[1] <= PI)
      {
        outer_angle = line_pair_data.outer_east[i];
        x = line_pair_data.vector_outer_east[i];
      }
      else
      {
        outer_angle = line_pair_data.outer_west[i];
        x = line_pair_data.vector_outer_west[i];
      }

      outer_alpha = outer_angle(0);
      outer_phi = outer_angle(1);
      auto [phi_far, phi_near] = intervalProjection(outer_phi, range_phi);
      auto [alpha_far, alpha_near] = intervalProjection(outer_alpha, range_alpha);
      bool is_north = range_alpha[0] <= PI / 2 && range_alpha[1] <= PI / 2;
      bool is_south = !is_north;

      // Find maximum (complex logic from MATLAB)
      double maximum;
      double delta_phi_near = std::abs(phi_near - outer_phi);
      if (std::abs(outer_alpha - PI / 2) < 1e-5 && (range_alpha[0] >= PI / 2 || range_alpha[1] <= PI / 2))
      {
        if ((delta_phi_near <= PI / 2 && is_north) || (delta_phi_near > PI / 2 && is_south))
        {
          maximum = x.dot(helper::polarToXyz(range_alpha[1], phi_near));
        }
        else
        {
          maximum = x.dot(helper::polarToXyz(range_alpha[0], phi_near));
        }
      }
      else if (delta_phi_near == 0)
      {
        maximum = x.dot(helper::polarToXyz(alpha_near, phi_near));
      }
      else if (delta_phi_near > PI / 2)
      {
        double tangent = std::tan(outer_alpha) * std::cos(delta_phi_near);
        double max_alpha;
        if (tangent > 1e8)
        {
          max_alpha = PI / 2;
        }
        else
        {
          max_alpha = std::atan(tangent);
          if (max_alpha < 0)
            max_alpha += PI;
        }

        if (max_alpha <= (range_alpha[0] + range_alpha[1]) / 2)
        {
          maximum = x.dot(helper::polarToXyz(range_alpha[1], phi_near));
        }
        else
        {
          maximum = x.dot(helper::polarToXyz(range_alpha[0], phi_near));
        }
      }
      else if (delta_phi_near < PI / 2 && outer_alpha < PI / 2 && range_alpha[0] >= outer_alpha)
      {
        maximum = x.dot(helper::polarToXyz(range_alpha[0], phi_near));
      }
      else if (delta_phi_near < PI / 2 && outer_alpha > PI / 2 && range_alpha[1] <= PI - outer_alpha)
      {
        maximum = x.dot(helper::polarToXyz(range_alpha[1], phi_near));
      }
      else if (std::abs(delta_phi_near - PI / 2) < this->eps)
      {
        if (outer_alpha <= PI / 2)
        {
          maximum = x.dot(helper::polarToXyz(range_alpha[0], phi_near));
        }
        else
        {
          maximum = x.dot(helper::polarToXyz(range_alpha[1], phi_near));
        }
      }
      else
      {
        double tangent = std::tan(outer_alpha) * std::cos(delta_phi_near);
        double max_alpha;
        if (tangent > 1e8)
        {
          max_alpha = PI / 2;
        }
        else
        {
          max_alpha = std::atan(tangent);
          if (max_alpha < 0)
            max_alpha += PI;
        }

        if (max_alpha <= range_alpha[0])
        {
          maximum = x.dot(helper::polarToXyz(range_alpha[0], phi_near));
        }
        else if (max_alpha <= range_alpha[1])
        {
          maximum = x.dot(helper::polarToXyz(max_alpha, phi_near));
        }
        else
        {
          maximum = x.dot(helper::polarToXyz(range_alpha[1], phi_near));
        }
      }

      // Find minimum (complex logic from MATLAB)
      double minimum;
      double delta_phi_far = std::abs(phi_far - outer_phi);
      if (std::abs(outer_alpha - PI / 2) < 1e-5 && (range_alpha[0] >= PI / 2 || range_alpha[1] <= PI / 2))
      {
        if ((delta_phi_far <= PI / 2 && is_north) || (delta_phi_far > PI / 2 && is_south))
        {
          minimum = x.dot(helper::polarToXyz(range_alpha[0], phi_far));
        }
        else
        {
          minimum = x.dot(helper::polarToXyz(range_alpha[1], phi_far));
        }
      }
      else if (delta_phi_far < PI / 2)
      {
        double tangent = std::tan(outer_alpha) * std::cos(delta_phi_far);
        double min_alpha;
        if (tangent > 1e8)
        {
          min_alpha = PI / 2;
        }
        else
        {
          min_alpha = std::atan(tangent);
          if (min_alpha < 0)
            min_alpha += PI;
        }

        if (min_alpha <= (range_alpha[0] + range_alpha[1]) / 2)
        {
          minimum = x.dot(helper::polarToXyz(range_alpha[1], phi_far));
        }
        else
        {
          minimum = x.dot(helper::polarToXyz(range_alpha[0], phi_far));
        }
      }
      else if (delta_phi_far > PI / 2 && outer_alpha < PI / 2 && range_alpha[1] <= PI - outer_alpha)
      {
        minimum = x.dot(helper::polarToXyz(range_alpha[1], phi_far));
      }
      else if (delta_phi_far > PI / 2 && outer_alpha > PI / 2 && range_alpha[0] >= PI - outer_alpha)
      {
        minimum = x.dot(helper::polarToXyz(range_alpha[0], phi_far));
      }
      else if (std::abs(delta_phi_far - PI / 2) < this->eps)
      {
        if (outer_alpha <= PI / 2)
        {
          minimum = x.dot(helper::polarToXyz(range_alpha[1], phi_far));
        }
        else
        {
          minimum = x.dot(helper::polarToXyz(range_alpha[0], phi_far));
        }
      }
      else
      {
        double tangent = std::tan(outer_alpha) * std::cos(delta_phi_far);
        double min_alpha;
        if (tangent > 1e8)
        {
          min_alpha = PI / 2;
        }
        else
        {
          min_alpha = std::atan(tangent);
          if (min_alpha < 0)
            min_alpha += PI;
        }

        if (min_alpha <= range_alpha[0])
        {
          minimum = x.dot(helper::polarToXyz(range_alpha[0], phi_far));
        }
        else if (min_alpha <= range_alpha[1])
        {
          minimum = x.dot(helper::polarToXyz(min_alpha, phi_far));
        }
        else
        {
          minimum = x.dot(helper::polarToXyz(range_alpha[1], phi_far));
        }
      }

      if (flag == 1)
      {
        h1_upper[i] = maximum;
        h1_lower[i] = minimum;
      }
      else
      {
        h1_upper[i] = -minimum;
        h1_lower[i] = -maximum;
      }
    }
  }
  return {h1_upper, h1_lower};
}

std::pair<std::vector<double>, std::vector<double>>
RotFGO::h2IntervalMapping(const LinePairData &line_pair_data,
                          const RBranch &branch)
{
  size_t N = line_pair_data.size;
  std::vector<double> h2_upper(N), h2_lower(N);
  std::vector<Eigen::Vector3d> vertex_cache;
  if (branch.alpha_max - branch.alpha_min <= sample_resolution_)
  {
    // Small cube - use 4 vertices
    vertex_cache.resize(4);
    vertex_cache[0] = helper::polarToXyz(branch.alpha_min, branch.phi_min);
    vertex_cache[1] = helper::polarToXyz(branch.alpha_min, branch.phi_max);
    vertex_cache[2] = helper::polarToXyz(branch.alpha_max, branch.phi_min);
    vertex_cache[3] = helper::polarToXyz(branch.alpha_max, branch.phi_max);
  }
  else
  {
    // large cube - sample boundaries following MATLAB pattern exactly
    // MATLAB: alpha = branch(1):sample_resolution:branch(3);
    // MATLAB: phi = branch(2):sample_resolution:branch(4);
    std::vector<double> alpha_range, phi_range;
    for (double a = branch.alpha_min; a <= branch.alpha_max + this->eps; a += sample_resolution_)
    {
      alpha_range.emplace_back(std::min(a, branch.alpha_max));
    }
    for (double p = branch.phi_min; p <= branch.phi_max + this->eps; p += sample_resolution_)
    {
      phi_range.emplace_back(std::min(p, branch.phi_max));
    }

    if (alpha_range.empty())
      alpha_range.emplace_back(branch.alpha_min);
    if (phi_range.empty())
      phi_range.emplace_back(branch.phi_min);

    size_t temp = alpha_range.size() - 1;
    if (temp <= 0)
      temp = 1; // Ensure at least 1

    vertex_cache.reserve(temp * 4);
    // MATLAB: vertex_cache(1:temp,:)=vec_polar2xyz(alpha(1:temp),phi(1));
    for (size_t a = 0; a < temp; a++)
    {
      vertex_cache.emplace_back(helper::polarToXyz(alpha_range[a], phi_range[0]));
    }
    // MATLAB: vertex_cache(temp+1:2*temp,:) = vec_polar2xyz(alpha(end),phi(1:temp));
    for (size_t p = 0; p < temp; p++)
    {
      vertex_cache.emplace_back(helper::polarToXyz(alpha_range.back(), phi_range[p]));
    }
    // MATLAB: vertex_cache(2*temp+1:3*temp,:) = vec_polar2xyz(alpha(2:end),phi(end));
    for (size_t a = 1; a < alpha_range.size(); a++)
    {
      vertex_cache.emplace_back(helper::polarToXyz(alpha_range[a], phi_range.back()));
    }
    // MATLAB: vertex_cache(3*temp+1:4*temp,:)=vec_polar2xyz(alpha(1),phi(2:end));
    for (size_t p = 1; p < phi_range.size(); p++)
    {
      vertex_cache.emplace_back(helper::polarToXyz(alpha_range[0], phi_range[p]));
    }
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(4)
#endif
  for (size_t i = 0; i < N; i++)
  {
    const Eigen::Vector3d &n_i = line_pair_data.vector_n[i];
    const Eigen::Vector3d &v_i = line_pair_data.vector_v[i];
    double inner_product = line_pair_data.inner_product[i];
    Eigen::Vector2d normal_angle, o_normal_angle;
    Eigen::Vector3d normal_vector, o_normal_vector;

    if (branch.phi_min < PI)
    {
      normal_angle = line_pair_data.normal_east[i];
      normal_vector = line_pair_data.vector_normal_east[i];
      o_normal_angle = line_pair_data.o_normal_east[i];
      o_normal_vector = line_pair_data.vector_o_normal_east[i];
    }
    else
    {
      normal_angle = line_pair_data.normal_west[i];
      normal_vector = line_pair_data.vector_normal_west[i];
      o_normal_angle = line_pair_data.o_normal_west[i];
      o_normal_vector = line_pair_data.vector_o_normal_west[i];
    }

    bool normal_in_branch = (normal_angle(0) >= branch.alpha_min &&
                             normal_angle(0) <= branch.alpha_max &&
                             normal_angle(1) >= branch.phi_min &&
                             normal_angle(1) <= branch.phi_max);
    bool o_normal_in_branch = (o_normal_angle(0) >= branch.alpha_min &&
                               o_normal_angle(0) <= branch.alpha_max &&
                               o_normal_angle(1) >= branch.phi_min &&
                               o_normal_angle(1) <= branch.phi_max);
    int flag = (normal_in_branch ? 2 : 0) + (o_normal_in_branch ? 1 : 0);

    double maximum = std::numeric_limits<double>::lowest();
    double minimum = std::numeric_limits<double>::max();

    switch (flag)
    {
    case 3: // Both normals in branch
      maximum = normal_vector.dot(n_i) * normal_vector.dot(v_i);
      minimum = o_normal_vector.dot(n_i) * o_normal_vector.dot(v_i);
      break;

    case 2: // Only normal in branch
      maximum = normal_vector.dot(n_i) * normal_vector.dot(v_i);
      // Find minimum from vertices
      for (const auto &vertex : vertex_cache)
      {
        double val = vertex.dot(n_i) * vertex.dot(v_i);
        minimum = std::min(minimum, val);
      }
      break;

    case 1: // Only o_normal in branch
      minimum = o_normal_vector.dot(n_i) * o_normal_vector.dot(v_i);
      // Find maximum from vertices
      for (const auto &vertex : vertex_cache)
      {
        double val = vertex.dot(n_i) * vertex.dot(v_i);
        maximum = std::max(maximum, val);
      }
      break;

    default: // Neither normal in branch
      for (const auto &vertex : vertex_cache)
      {
        double val = vertex.dot(n_i) * vertex.dot(v_i);
        maximum = std::max(maximum, val);
        minimum = std::min(minimum, val);
      }
      break;
    }
    h2_upper[i] = maximum - inner_product;
    h2_lower[i] = minimum - inner_product;
  }

  return {h2_upper, h2_lower};
}


std::vector<double> RotFGO::calcBounds(
    const LinePairData &line_pair_data, RBranch &branch,
    const std::vector<int> &ids,
    const Eigen::MatrixXd &sat_buffer)
{
  size_t N = line_pair_data.size;
  Eigen::Vector3d u_center =
      helper::polarToXyz(0.5 * (branch.alpha_min + branch.alpha_max),
                 0.5 * (branch.phi_min + branch.phi_max));
  std::vector<double> h1_center(N), h2_center(N);

  // #ifdef _OPENMP // TODO: check optimum num or use SIMD
  // #pragma omp parallel for schedule(static) num_threads(4)
  // #endif
  for (size_t i = 0; i < N; i++)
  {
    h1_center[i] = u_center.dot(line_pair_data.outer_product[i]);
    h2_center[i] = u_center.dot(line_pair_data.vector_n[i]) *
                       u_center.dot(line_pair_data.vector_v[i]) -
                   line_pair_data.inner_product[i];
  }

  auto [A_center, phi_center, const_center] =
      calcIntervalParams(line_pair_data.inner_product, h1_center, h2_center);

  std::vector<double> intervals_lower;
  std::vector<int> ids_lower;
  // Pre-estimate capacity to reduce reallocations
  intervals_lower.reserve(N * 2); // Estimate 2 intervals per line on average
  ids_lower.reserve(N);

  for (size_t i = 0; i < N; i++)
  {
    std::vector<double> tmp_interval =
        lowerInterval(A_center[i], phi_center[i], const_center[i]);
    intervals_lower.insert(intervals_lower.end(), tmp_interval.begin(),
                           tmp_interval.end());

    int num_intervals = tmp_interval.size() / 2;
    for (int j = 0; j < num_intervals; j++)
    {
      ids_lower.emplace_back(ids[i]);
    }
  }

  std::vector<double> theta_lower;
  if (!ids_lower.empty())
  {
    auto [score, stabbers] = SatIS::saturatedIntervalStabbing(
        intervals_lower, ids_lower, sat_buffer, prox_threshold_);
    branch.lower_bound = score;
    theta_lower = stabbers;
  }

  // Calculate upper bound using interval analysis
  auto [h1_upper, h1_lower] =
      h1IntervalMapping(line_pair_data, branch);
  auto [h2_upper, h2_lower] =
      h2IntervalMapping(line_pair_data, branch);
  auto [A_lower, phi_lower, const_lower] =
      calcIntervalParams(line_pair_data.inner_product, h1_lower, h2_lower);
  auto [A_upper, phi_upper, const_upper] =
      calcIntervalParams(line_pair_data.inner_product, h1_upper, h2_upper);

  // Prepare intervals for upper bound
  std::vector<double> intervals_upper;
  std::vector<int> ids_upper;
  intervals_upper.reserve(N * 2);
  ids_upper.reserve(N);
  for (size_t i = 0; i < N; i++)
  {
    std::vector<double> tmp_interval =
        upperInterval(A_upper[i], phi_upper[i], const_upper[i], A_lower[i],
                      phi_lower[i], const_lower[i]);
    intervals_upper.insert(intervals_upper.end(), tmp_interval.begin(),
                           tmp_interval.end());

    int num_intervals = tmp_interval.size() / 2;
    for (int j = 0; j < num_intervals; j++)
    {
      ids_upper.emplace_back(ids[i]);
    }
  }

  if (!ids_upper.empty())
  {
    auto [score, _] = SatIS::saturatedIntervalStabbing(intervals_upper, ids_upper,
                                                       sat_buffer, prox_threshold_);
    branch.upper_bound = score;
  }
  return theta_lower;
}


std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
RotFGO::calcIntervalParams(const std::vector<double> &inner_product,
                           const std::vector<double> &h1,
                           const std::vector<double> &h2)
{
  size_t N = inner_product.size();
  std::vector<double> A(N), phi(N), constant(N);
  for (size_t i = 0; i < N; i++)
  {
    A[i] = std::sqrt(h1[i] * h1[i] + h2[i] * h2[i]);
    phi[i] = std::atan2(-h2[i], h1[i]);
    if (phi[i] < 0)
      phi[i] += 2 * PI;
    constant[i] = inner_product[i] + h2[i];
  }
  return std::make_tuple(A, phi, constant);
}



void RotFGO::updateBestSolution(const RBranch &branch,
                                const std::vector<double> &theta_candidates,
                                double &best_lb,
                                std::vector<Eigen::Vector3d> &best_axes,
                                std::vector<double> &best_angles)
{
  if (branch.lower_bound - this->eps > best_lb)
  {
    best_lb = branch.lower_bound;
    best_axes.clear();
    best_angles.clear();
    std::vector<double> clustered_theta = SatIS::clusterStabber(theta_candidates, prox_threshold_);
    Eigen::Vector3d rotation_axis = helper::polarToXyz(
        0.5 * (branch.alpha_min + branch.alpha_max),
        0.5 * (branch.phi_min + branch.phi_max));
    for (double theta : clustered_theta)
    {
      best_axes.emplace_back(rotation_axis);
      best_angles.emplace_back(theta);
    }
  }
  else if (std::abs(branch.lower_bound - best_lb) < this->eps)
  {
    // Found an equally good solution - add to existing candidates
    std::vector<double> clustered_theta = SatIS::clusterStabber(theta_candidates, prox_threshold_);
    Eigen::Vector3d rotation_axis = helper::polarToXyz(
        0.5 * (branch.alpha_min + branch.alpha_max),
        0.5 * (branch.phi_min + branch.phi_max));
    for (double theta : clustered_theta)
    {
      best_axes.emplace_back(rotation_axis);
      best_angles.emplace_back(theta);
    }
  }
}


void RotFGO::pruneBranchQueue(BranchQueue &bq, double best_lb)
{
  BranchQueue promising;
  while (!bq.empty())
  {
    const RBranch &branch = bq.top();
    if (branch.upper_bound + this->eps < best_lb)
      break;
    promising.push(std::move(const_cast<RBranch &>(branch)));
    bq.pop();
  }
  bq = std::move(promising);
}
