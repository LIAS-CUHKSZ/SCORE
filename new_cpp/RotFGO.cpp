#include "RotFGO.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <queue>

const double PI = M_PI;

RotFGO::RotFGO() : last_solve_time_(0.0), num_candidates_(0), verbose_(false) {}

RotFGO::~RotFGO() {}

std::vector<Eigen::Matrix3d>
RotFGO::solve(const std::vector<Eigen::Vector3d> &vector_n,
              const std::vector<Eigen::Vector3d> &vector_v,
              const std::vector<int> &ids, const Eigen::MatrixXd &kernel_buffer,
              double branch_resolution, double epsilon_r,
              double sample_resolution, double prox_threshold,
              int west_or_east)
{
  auto start_time = std::chrono::high_resolution_clock::now();

  // Step 1: pre compute line pair data
  LinePairData line_pair_data = dataProcess(vector_n, vector_v);

  // Step 2: init BnB process
  double best_lower = -1.0;
  double best_upper = -1.0;
  std::vector<Eigen::Vector3d> u_best;
  std::vector<double> theta_best;
  std::vector<Branch> branches; // Store all branches for pruning

  Branch next_branch;
  int iteration = 0;

  if (west_or_east == 2)
  {
    // Branch over the whole sphere, start with east hemisphere
    Branch east_branch(0, 0, PI, PI);
    auto [upper, lower, theta] =
        calculateBounds(line_pair_data, east_branch, epsilon_r,
                        sample_resolution, ids, kernel_buffer, prox_threshold);
    iteration++;
    // Update best bounds
    best_lower = lower;
    best_upper = upper;

    // Add east_branch to branches (as done in MATLAB: branch = [east_branch;upper;lower])
    east_branch.upper_bound = upper;
    east_branch.lower_bound = lower;
    branches.push_back(east_branch);

    // Cluster stabbers and get rotation candidates
    std::vector<double> clustered_theta = clusterStabber(theta, prox_threshold);
    Eigen::Vector3d u_center =
        polarToXyz(0.5 * (east_branch.alpha_min + east_branch.alpha_max),
                   0.5 * (east_branch.phi_min + east_branch.phi_max));

    // Initialize u_best and theta_best with east hemisphere results
    for (double t : clustered_theta)
    {
      u_best.push_back(u_center);
      theta_best.push_back(t);
    }

    // Set west hemisphere as next branch
    next_branch = Branch(0, PI, PI, 2 * PI);
  }
  else
  {
    if (west_or_east == 1)
    {
      // Only west hemisphere
      next_branch = Branch(0, PI, PI, 2 * PI);
    }
    else
    {
      // Only east hemisphere
      next_branch = Branch(0, 0, PI, PI);
    }
  }

  // Step 3: BnB process
  while (true)
  {
    // Subdivide the branch into four
    std::vector<Branch> sub_branches = subBranch(next_branch);

    std::vector<double> new_upper(4), new_lower(4);
    std::vector<std::vector<double>> new_theta_lower(4);

    // Calculate bounds for each sub-branch
    for (int i = 0; i < 4; i++)
    {
      auto [upper, lower, theta] = calculateBounds(
          line_pair_data, sub_branches[i], epsilon_r, sample_resolution, ids,
          kernel_buffer, prox_threshold);
      iteration++;
      new_upper[i] = upper;
      new_lower[i] = lower;
      new_theta_lower[i] = theta;

      sub_branches[i].upper_bound = upper;
      sub_branches[i].lower_bound = lower;
      branches.push_back(sub_branches[i]);
      std::cout << "Iteration: " << iteration
                << ", Sub-branch " << i
                << ", Upper Bound: " << upper
                << ", Lower Bound: " << lower
                << ", Best Lower: " << best_lower << std::endl;
    }

    // Update best lower bound and rotation candidates
    double sub_max_lower =
        *std::max_element(new_lower.begin(), new_lower.end());

    // Find all indices with maximum lower bound
    std::vector<int> idx_sub_best;
    for (int i = 0; i < 4; i++)
    {
      if (new_lower[i] == sub_max_lower)
      {
        idx_sub_best.push_back(i);
      }
    }

    // Process each best sub-branch
    for (int idx : idx_sub_best)
    {
      if (sub_max_lower > best_lower)
      {
        best_lower = sub_max_lower;

        // Clear previous best candidates and add new ones
        u_best.clear();
        theta_best.clear();

        std::vector<double> clustered_theta =
            clusterStabber(new_theta_lower[idx], prox_threshold);
        Eigen::Vector3d u_center = polarToXyz(
            0.5 * (sub_branches[idx].alpha_min + sub_branches[idx].alpha_max),
            0.5 * (sub_branches[idx].phi_min + sub_branches[idx].phi_max));

        for (double t : clustered_theta)
        {
          u_best.push_back(u_center);
          theta_best.push_back(t);
        }
      }
      else if (sub_max_lower == best_lower)
      {
        // Add to existing best candidates
        std::vector<double> clustered_theta =
            clusterStabber(new_theta_lower[idx], prox_threshold);
        Eigen::Vector3d u_center = polarToXyz(
            0.5 * (sub_branches[idx].alpha_min + sub_branches[idx].alpha_max),
            0.5 * (sub_branches[idx].phi_min + sub_branches[idx].phi_max));

        for (double t : clustered_theta)
        {
          u_best.push_back(u_center);
          theta_best.push_back(t);
        }
      }
    }

    // Prune branches with lower bound less than best_lower (as in MATLAB)
    branches.erase(std::remove_if(branches.begin(), branches.end(),
                                  [best_lower](const Branch &b)
                                  {
                                    return b.upper_bound < best_lower;
                                  }),
                   branches.end());

    // Find next branch with best upper bound and largest size
    if (branches.empty())
      break;

    best_upper = -1.0;
    int best_idx = -1;
    double largest_size = -1.0;

    // Find branches with best upper bound
    for (size_t i = 0; i < branches.size(); i++)
    {
      best_upper = std::max(best_upper, branches[i].upper_bound);
    }

    // Among branches with best upper bound, find the one with largest size
    for (size_t i = 0; i < branches.size(); i++)
    {
      if (branches[i].upper_bound == best_upper)
      {
        double branch_size = branches[i].alpha_max - branches[i].alpha_min;
        if (branch_size > largest_size)
        {
          largest_size = branch_size;
          best_idx = i;
        }
      }
    }

    next_branch = branches[best_idx];
    branches.erase(branches.begin() + best_idx);

    // Stop condition
    if ((next_branch.alpha_max - next_branch.alpha_min) < branch_resolution)
    {
      break;
    }
  }

  // Step 4: Generate rotation matrices
  std::vector<Eigen::Matrix3d> R_opt;
  num_candidates_ = u_best.size();

  for (int i = 0; i < num_candidates_; i++)
  {
    R_opt.push_back(axisAngleToRotMatrix(u_best[i], theta_best[i]));
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  last_solve_time_ =
      std::chrono::duration<double>(end_time - start_time).count();

  return R_opt;
}

LinePairData RotFGO::dataProcess(const std::vector<Eigen::Vector3d> &vector_n,
                                 const std::vector<Eigen::Vector3d> &vector_v)
{
  LinePairData data;
  int N = vector_n.size();
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

  for (int i = 0; i < N; i++)
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

    // Convert to polar coordinates
    auto [alpha, phi] = xyzToPolar(data.outer_product[i]);
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

    // Inner product
    data.inner_product[i] = v.dot(n);

    // Calculate normals
    auto [normal_east, normal_west, o_normal_east, o_normal_west] =
        calculateNormals(n, v);
    data.normal_east[i] = normal_east;
    data.normal_west[i] = normal_west;
    data.o_normal_east[i] = o_normal_east;
    data.o_normal_west[i] = o_normal_west;

    // Convert normals to xyz
    data.vector_normal_east[i] = polarToXyz(normal_east(0), normal_east(1));
    data.vector_normal_west[i] = polarToXyz(normal_west(0), normal_west(1));
    data.vector_o_normal_east[i] =
        polarToXyz(o_normal_east(0), o_normal_east(1));
    data.vector_o_normal_west[i] =
        polarToXyz(o_normal_west(0), o_normal_west(1));

    data.outer_norm[i] = data.outer_product[i].norm();
  }

  return data;
}

std::pair<double, std::vector<double>> RotFGO::saturatedIntervalStabbing(
    const std::vector<double> &intervals, const std::vector<int> &ids,
    const Eigen::MatrixXd &kernel_buffer, double prox_threshold)
{
  int L = ids.size();

  // Create endpoint events
  std::vector<std::pair<double, std::pair<int, int>>>
      events; // (value, (mask, id_index))

  for (int i = 0; i < L; i++)
  {
    events.push_back({intervals[2 * i], {0, i}});     // entering interval
    events.push_back({intervals[2 * i + 1], {1, i}}); // exiting interval
  }

  // Sort events by value
  std::sort(events.begin(), events.end());

  int max_id = *std::max_element(ids.begin(), ids.end());
  std::vector<int> count_buffer(max_id + 1, 0);

  double score = 0.0;
  double best_score = 0.0;
  std::vector<double> stabbers;

  for (int i = 0; i < events.size() - 1; i++)
  {
    double current_pos = events[i].first;
    int mask = events[i].second.first;
    int id_idx = events[i].second.second;
    int line_id = ids[id_idx];

    if (mask == 0)
    { // entering interval
      count_buffer[line_id]++;
      score += kernel_buffer(line_id, count_buffer[line_id] - 1); // 0-indexed access

      if (score >= best_score)
      {
        double next_pos = events[i + 1].first;

        // Generate stabbers in this interval using the MATLAB approach
        std::vector<double> new_stabbers;

        // MATLAB: [Intervals(sidx(i)):prox_thres:Intervals(sidx(i+1)),Intervals(sidx(i+1))]
        double pos = current_pos;
        while (pos <= next_pos)
        {
          new_stabbers.push_back(pos);
          pos += prox_threshold;
        }
        // Always include the endpoint
        if (new_stabbers.empty() || new_stabbers.back() != next_pos)
        {
          new_stabbers.push_back(next_pos);
        }

        if (score > best_score)
        {
          stabbers = new_stabbers;
          best_score = score;
        }
        else if (score == best_score)
        {
          stabbers.insert(stabbers.end(), new_stabbers.begin(),
                          new_stabbers.end());
        }
      }
    }
    else
    { // exiting interval
      score -= kernel_buffer(line_id, count_buffer[line_id] - 1);
      count_buffer[line_id]--;
    }
  }

  return {best_score, stabbers};
}

Eigen::Vector3d RotFGO::polarToXyz(double alpha, double phi)
{
  double sin_alpha = std::sin(alpha);
  return Eigen::Vector3d(sin_alpha * std::cos(phi), sin_alpha * std::sin(phi),
                         std::cos(alpha));
}

std::pair<double, double> RotFGO::xyzToPolar(const Eigen::Vector3d &axis)
{
  double length = axis.norm();
  if (length == 0)
  {
    return {0.0, 0.0};
  }

  Eigen::Vector3d unit_axis = axis / length;

  double alpha = std::atan2(
      std::sqrt(unit_axis(0) * unit_axis(0) + unit_axis(1) * unit_axis(1)),
      unit_axis(2));
  double phi;

  if (unit_axis(0) == 0 && unit_axis(1) == 0)
  {
    phi = 0.0;
  }
  else
  {
    phi = std::atan2(unit_axis(1), unit_axis(0));
  }

  if (phi < 0)
  {
    phi += 2 * PI;
  }

  return {alpha, phi};
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

    auto [alpha_v1, phi_v1] = xyzToPolar(v1);
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
  Eigen::Vector3d cross_nv = v1.cross(v2);
  cross_nv.normalize();
  Eigen::Vector3d orthogonal = mid.cross(cross_nv);
  orthogonal.normalize();

  auto [alpha_mid, phi_mid] = xyzToPolar(mid);
  auto [alpha_orthogonal, phi_orthogonal] = xyzToPolar(orthogonal);

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

std::vector<Branch> RotFGO::subBranch(const Branch &branch)
{
  std::vector<double> a = {branch.alpha_min, branch.phi_min};
  std::vector<double> b = {branch.alpha_max, branch.phi_max};
  std::vector<double> c = {0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])};

  std::vector<std::vector<double>> M = {{a[0], c[0], b[0]}, {a[1], c[1], b[1]}};

  std::vector<Branch> out;
  for (int i = 1; i <= 4; i++)
  { // MATLAB uses 1-based indexing
    // bitget in MATLAB: bitget(1,1)=1, bitget(2,1)=0, bitget(3,1)=1, bitget(4,1)=0
    // bitget in MATLAB: bitget(1,2)=0, bitget(2,2)=1, bitget(3,2)=1, bitget(4,2)=0
    int bit1 = (i & 1) ? 0 : 1;        // bitget(i,1)
    int bit2 = (i & 2) ? 1 : 0;        // bitget(i,2)
    double alpha_min = M[0][bit1];     // M(1, bitget(i,1)+1) in MATLAB
    double phi_min = M[1][bit2];       // M(2, bitget(i,2)+1) in MATLAB
    double alpha_max = M[0][bit1 + 1]; // M(1, bitget(i,1)+2) in MATLAB
    double phi_max = M[1][bit2 + 1];   // M(2, bitget(i,2)+2) in MATLAB

    out.push_back(Branch(alpha_min, phi_min, alpha_max, phi_max));
  }
  // inverse out
  std::reverse(out.begin(), out.end());
  return out;
}

std::vector<double> RotFGO::lowerInterval(double A, double phi, double constant,
                                          double epsilon)
{
  std::vector<double> interval;

  if (A == 0)
    return {};

  double c_up = -constant + epsilon;
  double c_lo = -constant - epsilon;

  // Helper function to normalize angle to [0, 2*PI]
  auto normalizeAngle = [](double angle)
  {
    while (angle < 0)
      angle += 2 * PI;
    while (angle >= 2 * PI)
      angle -= 2 * PI;
    return angle;
  };

  // Helper function to clamp to [0, PI]
  auto clampToPi = [](double val)
  {
    return std::max(0.0, std::min(PI, val));
  };

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
        interval.push_back(std::max(0.0, m_l - phi));
        interval.push_back(std::min(PI, m_r - phi));
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
        interval.push_back(m_l - phi);
        interval.push_back(std::min(PI, n_l - phi));
      }
      else if (phi <= n_l)
      {
        interval.push_back(std::max(m_l - phi, 0.0));
        interval.push_back(n_l - phi);
        interval.push_back(m_r - phi);
        interval.push_back(std::min(PI, n_r - phi));
      }
      else
      {
        interval.push_back(std::max(m_r - phi, 0.0));
        interval.push_back(std::min(PI, n_r - phi));
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
        interval.push_back(0.0);
        interval.push_back(m - phi);
        interval.push_back(PI - m - phi);
        interval.push_back(PI);
      }
      else if (phi <= 2 * PI - m)
      {
        interval.push_back(std::max(0.0, PI - m - phi));
        interval.push_back(std::min(PI, 2 * PI + m - phi));
      }
      else
      {
        interval.push_back(0.0);
        interval.push_back(2 * PI + m - phi);
        interval.push_back(3 * PI - m - phi);
        interval.push_back(PI);
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
        interval.push_back(0.0);
        interval.push_back(m - phi);
        interval.push_back(m_r - phi);
        interval.push_back(std::min(PI, n_l - phi));
      }
      else if (phi <= n_r - PI)
      {
        interval.push_back(std::max(0.0, m_r - phi));
        interval.push_back(std::min(PI, n_l - phi));
      }
      else if (phi <= n_l)
      {
        interval.push_back(std::max(0.0, m_r - phi));
        interval.push_back(n_l - phi);
        interval.push_back(n_r - phi);
        interval.push_back(std::min(PI, 2 * PI + m - phi));
      }
      else if (phi <= m_r + PI)
      {
        interval.push_back(std::max(0.0, n_r - phi));
        interval.push_back(std::min(PI, 2 * PI + m - phi));
      }
      else
      {
        interval.push_back(std::max(0.0, n_r - phi));
        interval.push_back(2 * PI + m - phi);
        interval.push_back(3 * PI - m - phi);
        interval.push_back(PI);
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
        interval.push_back(std::max(0.0, n - phi));
        interval.push_back(m - phi);
        interval.push_back(m_r_1 - phi);
        interval.push_back(n_r_1 - phi);
      }
      else if (phi <= PI + n && phi >= n_r_1)
      {
        return {};
      }
      else if (phi <= n_r_1)
      {
        interval.push_back(std::max(0.0, m_r_1 - phi));
        interval.push_back(n_r_1 - phi);
      }
      else if (phi <= m_r_1 + PI)
      {
        interval.push_back(2 * PI + n - phi);
        interval.push_back(std::min(PI, 2 * PI + m - phi));
      }
      else
      {
        interval.push_back(2 * PI + n - phi);
        interval.push_back(2 * PI + m - phi);
        interval.push_back(m_r_1 + 2 * PI - phi);
        interval.push_back(std::min(PI, n_r_1 + 2 * PI - phi));
      }
    }
  }
  else
  {
    if (c_lo <= -A)
    {
      interval.push_back(0.0);
      interval.push_back(PI);
    }
    else if (c_lo <= 0)
    {
      double m = std::asin(c_lo / A);
      double m_l = PI - m;
      double m_r = 2 * PI + m;
      if (phi <= m_r - PI)
      {
        interval.push_back(0.0);
        interval.push_back(std::min(m_l - phi, PI));
      }
      else if (phi >= m_l)
      {
        interval.push_back(std::max(0.0, m_r - phi));
        interval.push_back(PI);
      }
      else
      {
        interval.push_back(0.0);
        interval.push_back(m_l - phi);
        interval.push_back(m_r - phi);
        interval.push_back(PI);
      }
    }
    else if (c_lo < A)
    {
      double m = std::asin(c_lo / A);
      if (phi <= PI - m)
      {
        interval.push_back(std::max(0.0, m - phi));
        interval.push_back(PI - m - phi);
      }
      else if (phi >= PI + m)
      {
        interval.push_back(2 * PI + m - phi);
        interval.push_back(std::min(PI, 3 * PI - m - phi));
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

  // Filter out invalid intervals and normalize
  std::vector<double> result;
  for (int i = 0; i < interval.size(); i += 2)
  {
    if (i + 1 < interval.size())
    {
      double start = std::max(0.0, std::min(PI, interval[i]));
      double end = std::max(0.0, std::min(PI, interval[i + 1]));
      if (start < end)
      {
        result.push_back(start);
        result.push_back(end);
      }
    }
  }

  return result;
}

std::vector<double> RotFGO::upperInterval(double A_1, double phi_1,
                                          double const_1, double A_2,
                                          double phi_2, double const_2,
                                          double epsilon)
{
  // This function finds intervals satisfying both:
  // A_1*sin(theta + phi_1) + const_1 >= -epsilon
  // A_2*sin(theta + phi_2) + const_2 <= epsilon
  // Where A_1,phi_1,const_1 are upper bounds, A_2,phi_2,const_2 are lower bounds

  std::vector<double> upper_intervals, lower_intervals;

  // Helper function to intersect two intervals [a1, a2] and [b1, b2]
  auto intersectInterval = [](double a1, double a2, double b1, double b2) -> std::vector<double>
  {
    if (a2 < b1 || b2 < a1)
    {
      return {};
    }
    else
    {
      return {std::max(a1, b1), std::min(a2, b2)};
    }
  };

  // First constraint: A_1*sin(theta + phi_1) + const_1 >= -epsilon
  double c_lo = -const_1 - epsilon;
  if (A_1 < c_lo)
  {
    return {};
  }
  else if (c_lo >= 0)
  {
    double x_l = std::asin(c_lo / A_1);
    if (phi_1 <= PI - x_l)
    {
      upper_intervals.push_back(std::max(0.0, x_l - phi_1));
      upper_intervals.push_back(PI - x_l - phi_1);
    }
    else if (phi_1 >= PI + x_l)
    {
      upper_intervals.push_back(2 * PI + x_l - phi_1);
      upper_intervals.push_back(std::min(PI, 3 * PI - x_l - phi_1));
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
      upper_intervals.push_back(0.0);
      upper_intervals.push_back(std::min(x_l - phi_1, PI));
    }
    else if (phi_1 >= x_l)
    {
      upper_intervals.push_back(std::max(0.0, x_r - phi_1));
      upper_intervals.push_back(PI);
    }
    else
    {
      upper_intervals.push_back(0.0);
      upper_intervals.push_back(x_l - phi_1);
      upper_intervals.push_back(x_r - phi_1);
      upper_intervals.push_back(PI);
    }
  }
  else
  {
    upper_intervals.push_back(0.0);
    upper_intervals.push_back(PI);
  }

  // Second constraint: A_2*sin(theta + phi_2) + const_2 <= epsilon
  double c_up = epsilon - const_2;
  if (A_2 <= c_up)
  {
    lower_intervals.push_back(0.0);
    lower_intervals.push_back(PI);
  }
  else if (c_up >= 0)
  {
    double x_l = std::asin(c_up / A_2);
    if (phi_2 <= x_l)
    {
      lower_intervals.push_back(0.0);
      lower_intervals.push_back(x_l - phi_2);
      lower_intervals.push_back(PI - x_l - phi_2);
      lower_intervals.push_back(PI);
    }
    else if (phi_2 <= 2 * PI - x_l)
    {
      // Fixed: should be PI - x_l - phi_2, not max(0.0, PI - x_l - phi_2)
      // as shown in MATLAB: lower_interval = [ max(0,pi-x_l-phi_2);min(pi,2*pi+x_l-phi_2)];
      lower_intervals.push_back(std::max(0.0, PI - x_l - phi_2));
      lower_intervals.push_back(std::min(PI, 2 * PI + x_l - phi_2));
    }
    else
    {
      lower_intervals.push_back(0.0);
      lower_intervals.push_back(2 * PI + x_l - phi_2);
      lower_intervals.push_back(3 * PI - x_l - phi_2);
      lower_intervals.push_back(PI);
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
      lower_intervals.push_back(std::max(0.0, x_l - phi_2));
      lower_intervals.push_back(std::min(PI, x_r - phi_2));
    }
  }
  else
  {
    return {};
  }

  // Intersect all pairs of intervals
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
  for (int i = 0; i < result.size(); i += 2)
  {
    if (i + 1 < result.size())
    {
      double start = std::max(0.0, std::min(PI, result[i]));
      double end = std::max(0.0, std::min(PI, result[i + 1]));
      if (start < end)
      {
        filtered_result.push_back(start);
        filtered_result.push_back(end);
      }
    }
  }

  return filtered_result;
}

std::pair<std::vector<double>, std::vector<double>>
RotFGO::h1IntervalMapping(const LinePairData &line_pair_data,
                          const Branch &branch, double sample_resolution)
{
  int N = line_pair_data.size;
  std::vector<double> h1_upper(N), h1_lower(N);

  double cube_width = branch.alpha_max - branch.alpha_min;
  std::vector<double> range_alpha = {branch.alpha_min, branch.alpha_max};
  std::vector<double> range_phi = {branch.phi_min, branch.phi_max};

  // Helper function for interval projection
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

  if (cube_width <= sample_resolution)
  {
    // Small cube case
    for (int i = 0; i < N; i++)
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
        maximum = x.dot(polarToXyz(alpha_near, phi_near));
      }
      else
      {
        maximum = std::max(x.dot(polarToXyz(range_alpha[0], phi_near)),
                           x.dot(polarToXyz(range_alpha[1], phi_near)));
      }

      // Find minimum
      double minimum = std::min(x.dot(polarToXyz(range_alpha[0], phi_far)),
                                x.dot(polarToXyz(range_alpha[1], phi_far)));

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
    // Large cube case - implement the complex analytical method from MATLAB
    for (int i = 0; i < N; i++)
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
          maximum = x.dot(polarToXyz(range_alpha[1], phi_near));
        }
        else
        {
          maximum = x.dot(polarToXyz(range_alpha[0], phi_near));
        }
      }
      else if (delta_phi_near == 0)
      {
        maximum = x.dot(polarToXyz(alpha_near, phi_near));
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
          maximum = x.dot(polarToXyz(range_alpha[1], phi_near));
        }
        else
        {
          maximum = x.dot(polarToXyz(range_alpha[0], phi_near));
        }
      }
      else if (delta_phi_near < PI / 2 && outer_alpha < PI / 2 && range_alpha[0] >= outer_alpha)
      {
        maximum = x.dot(polarToXyz(range_alpha[0], phi_near));
      }
      else if (delta_phi_near < PI / 2 && outer_alpha > PI / 2 && range_alpha[1] <= PI - outer_alpha)
      {
        maximum = x.dot(polarToXyz(range_alpha[1], phi_near));
      }
      else if (std::abs(delta_phi_near - PI / 2) < 1e-10)
      {
        if (outer_alpha <= PI / 2)
        {
          maximum = x.dot(polarToXyz(range_alpha[0], phi_near));
        }
        else
        {
          maximum = x.dot(polarToXyz(range_alpha[1], phi_near));
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
          maximum = x.dot(polarToXyz(range_alpha[0], phi_near));
        }
        else if (max_alpha <= range_alpha[1])
        {
          maximum = x.dot(polarToXyz(max_alpha, phi_near));
        }
        else
        {
          maximum = x.dot(polarToXyz(range_alpha[1], phi_near));
        }
      }

      // Find minimum (complex logic from MATLAB)
      double minimum;
      double delta_phi_far = std::abs(phi_far - outer_phi);

      if (std::abs(outer_alpha - PI / 2) < 1e-5 && (range_alpha[0] >= PI / 2 || range_alpha[1] <= PI / 2))
      {
        if ((delta_phi_far <= PI / 2 && is_north) || (delta_phi_far > PI / 2 && is_south))
        {
          minimum = x.dot(polarToXyz(range_alpha[0], phi_far));
        }
        else
        {
          minimum = x.dot(polarToXyz(range_alpha[1], phi_far));
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
          minimum = x.dot(polarToXyz(range_alpha[1], phi_far));
        }
        else
        {
          minimum = x.dot(polarToXyz(range_alpha[0], phi_far));
        }
      }
      else if (delta_phi_far > PI / 2 && outer_alpha < PI / 2 && range_alpha[1] <= PI - outer_alpha)
      {
        minimum = x.dot(polarToXyz(range_alpha[1], phi_far));
      }
      else if (delta_phi_far > PI / 2 && outer_alpha > PI / 2 && range_alpha[0] >= PI - outer_alpha)
      {
        minimum = x.dot(polarToXyz(range_alpha[0], phi_far));
      }
      else if (std::abs(delta_phi_far - PI / 2) < 1e-10)
      {
        if (outer_alpha <= PI / 2)
        {
          minimum = x.dot(polarToXyz(range_alpha[1], phi_far));
        }
        else
        {
          minimum = x.dot(polarToXyz(range_alpha[0], phi_far));
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
          minimum = x.dot(polarToXyz(range_alpha[0], phi_far));
        }
        else if (min_alpha <= range_alpha[1])
        {
          minimum = x.dot(polarToXyz(min_alpha, phi_far));
        }
        else
        {
          minimum = x.dot(polarToXyz(range_alpha[1], phi_far));
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
                          const Branch &branch, double sample_resolution)
{
  int N = line_pair_data.size;
  std::vector<double> h2_upper(N), h2_lower(N);

  std::vector<Eigen::Vector3d> vertex_cache;

  if (branch.alpha_max - branch.alpha_min <= sample_resolution)
  {
    // Small cube - use 4 vertices
    vertex_cache.resize(4);
    vertex_cache[0] = polarToXyz(branch.alpha_min, branch.phi_min);
    vertex_cache[1] = polarToXyz(branch.alpha_min, branch.phi_max);
    vertex_cache[2] = polarToXyz(branch.alpha_max, branch.phi_min);
    vertex_cache[3] = polarToXyz(branch.alpha_max, branch.phi_max);
  }
  else
  {
    // Large cube - sample boundaries following MATLAB pattern exactly
    // MATLAB: alpha = branch(1):sample_resolution:branch(3);
    // MATLAB: phi = branch(2):sample_resolution:branch(4);
    std::vector<double> alpha_range, phi_range;

    for (double a = branch.alpha_min; a <= branch.alpha_max + 1e-10; a += sample_resolution)
    {
      alpha_range.push_back(std::min(a, branch.alpha_max));
    }
    for (double p = branch.phi_min; p <= branch.phi_max + 1e-10; p += sample_resolution)
    {
      phi_range.push_back(std::min(p, branch.phi_max));
    }

    if (alpha_range.empty())
      alpha_range.push_back(branch.alpha_min);
    if (phi_range.empty())
      phi_range.push_back(branch.phi_min);

    int temp = alpha_range.size() - 1;
    if (temp <= 0)
      temp = 1; // Ensure at least 1

    vertex_cache.reserve(temp * 4);

    // MATLAB: vertex_cache(1:temp,:)=vec_polar2xyz(alpha(1:temp),phi(1));
    for (int a = 0; a < temp; a++)
    {
      vertex_cache.push_back(polarToXyz(alpha_range[a], phi_range[0]));
    }

    // MATLAB: vertex_cache(temp+1:2*temp,:) = vec_polar2xyz(alpha(end),phi(1:temp));
    for (int p = 0; p < temp; p++)
    {
      vertex_cache.push_back(polarToXyz(alpha_range.back(), phi_range[p]));
    }

    // MATLAB: vertex_cache(2*temp+1:3*temp,:) = vec_polar2xyz(alpha(2:end),phi(end));
    for (int a = 1; a < alpha_range.size(); a++)
    {
      vertex_cache.push_back(polarToXyz(alpha_range[a], phi_range.back()));
    }

    // MATLAB: vertex_cache(3*temp+1:4*temp,:)=vec_polar2xyz(alpha(1),phi(2:end));
    for (int p = 1; p < phi_range.size(); p++)
    {
      vertex_cache.push_back(polarToXyz(alpha_range[0], phi_range[p]));
    }
  }

  for (int i = 0; i < N; i++)
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

std::tuple<double, double, std::vector<double>> RotFGO::calculateBounds(
    const LinePairData &line_pair_data, const Branch &branch, double epsilon,
    double sample_resolution, const std::vector<int> &ids,
    const Eigen::MatrixXd &kernel_buffer, double prox_threshold)
{
  int N = line_pair_data.size;

  // Calculate lower bound using center point
  Eigen::Vector3d u_center =
      polarToXyz(0.5 * (branch.alpha_min + branch.alpha_max),
                 0.5 * (branch.phi_min + branch.phi_max));

  std::vector<double> h1_center(N), h2_center(N);
  for (int i = 0; i < N; i++)
  {
    h1_center[i] = u_center.dot(line_pair_data.outer_product[i]);
    h2_center[i] = u_center.dot(line_pair_data.vector_n[i]) *
                       u_center.dot(line_pair_data.vector_v[i]) -
                   line_pair_data.inner_product[i];
  }

  auto [A_center, phi_center, const_center] =
      calculateParams(line_pair_data.inner_product, h1_center, h2_center);

  // Prepare intervals for lower bound
  std::vector<double> intervals_lower;
  std::vector<int> ids_lower;

  for (int i = 0; i < N; i++)
  {
    std::vector<double> tmp_interval =
        lowerInterval(A_center[i], phi_center[i], const_center[i], epsilon);
    intervals_lower.insert(intervals_lower.end(), tmp_interval.begin(),
                           tmp_interval.end());

    int num_intervals = tmp_interval.size() / 2;
    for (int j = 0; j < num_intervals; j++)
    {
      ids_lower.push_back(ids[i]);
    }
  }

  double Q_lower = 0.0;
  std::vector<double> theta_lower;

  if (!ids_lower.empty())
  {
    auto [score, stabbers] = saturatedIntervalStabbing(
        intervals_lower, ids_lower, kernel_buffer, prox_threshold);
    Q_lower = score;
    theta_lower = stabbers;
  }

  // Calculate upper bound using interval analysis
  auto [h1_upper, h1_lower] =
      h1IntervalMapping(line_pair_data, branch, sample_resolution);
  auto [h2_upper, h2_lower] =
      h2IntervalMapping(line_pair_data, branch, sample_resolution);

  auto [A_lower, phi_lower, const_lower] =
      calculateParams(line_pair_data.inner_product, h1_lower, h2_lower);
  auto [A_upper, phi_upper, const_upper] =
      calculateParams(line_pair_data.inner_product, h1_upper, h2_upper);

  // Prepare intervals for upper bound
  std::vector<double> intervals_upper;
  std::vector<int> ids_upper;

  for (int i = 0; i < N; i++)
  {
    std::vector<double> tmp_interval =
        upperInterval(A_upper[i], phi_upper[i], const_upper[i], A_lower[i],
                      phi_lower[i], const_lower[i], epsilon);
    intervals_upper.insert(intervals_upper.end(), tmp_interval.begin(),
                           tmp_interval.end());

    int num_intervals = tmp_interval.size() / 2;
    for (int j = 0; j < num_intervals; j++)
    {
      ids_upper.push_back(ids[i]);
    }
  }

  double Q_upper = 0.0;
  if (!ids_upper.empty())
  {
    auto [score, _] = saturatedIntervalStabbing(intervals_upper, ids_upper,
                                                kernel_buffer, prox_threshold);
    Q_upper = score;
  }

  return std::make_tuple(Q_upper, Q_lower, theta_lower);
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
RotFGO::calculateParams(const std::vector<double> &inner_product,
                        const std::vector<double> &h1,
                        const std::vector<double> &h2)
{
  int N = inner_product.size();
  std::vector<double> A(N), phi(N), constant(N);

  for (int i = 0; i < N; i++)
  {
    A[i] = std::sqrt(h1[i] * h1[i] + h2[i] * h2[i]);
    phi[i] = std::atan2(-h2[i], h1[i]);
    if (phi[i] < 0)
      phi[i] += 2 * PI;
    constant[i] = inner_product[i] + h2[i];
  }

  return std::make_tuple(A, phi, constant);
}

std::vector<double> RotFGO::clusterStabber(const std::vector<double> &theta,
                                           double prox_threshold)
{
  if (theta.empty())
    return {};

  if (theta.size() == 1)
    return theta;

  std::vector<double> sorted_theta = theta;
  std::sort(sorted_theta.begin(), sorted_theta.end());

  std::vector<double> stabber_buffer;
  std::vector<double> stabber_clustered;

  stabber_buffer.push_back(sorted_theta[0]);

  for (int n = 1; n < sorted_theta.size(); n++)
  {
    double new_stabber = sorted_theta[n];

    if (new_stabber - stabber_buffer[0] > prox_threshold)
    {
      // Case 1: difference with current stabber head is too large
      // Get median stabber in the buffer
      int temp_idx = stabber_buffer.size() - 1 + (stabber_buffer.size() % 2);
      std::vector<double> temp_buffer(stabber_buffer.begin(),
                                      stabber_buffer.begin() + temp_idx);
      std::sort(temp_buffer.begin(), temp_buffer.end());

      double median_stabber;
      if (temp_buffer.size() % 2 == 1)
      {
        median_stabber = temp_buffer[temp_buffer.size() / 2];
      }
      else
      {
        median_stabber = (temp_buffer[temp_buffer.size() / 2 - 1] +
                          temp_buffer[temp_buffer.size() / 2]) /
                         2.0;
      }

      // Push new cluster into cluster buffer
      stabber_clustered.push_back(median_stabber);

      // Clear the stabber buffer
      stabber_buffer.clear();
    }

    // Push in the current stabber
    stabber_buffer.push_back(new_stabber);
  }

  // Handle the last cluster
  if (!stabber_buffer.empty())
  {
    int temp_idx = stabber_buffer.size() - 1 + (stabber_buffer.size() % 2);
    std::vector<double> temp_buffer(stabber_buffer.begin(),
                                    stabber_buffer.begin() + temp_idx);
    std::sort(temp_buffer.begin(), temp_buffer.end());

    double median_stabber;
    if (temp_buffer.size() % 2 == 1)
    {
      median_stabber = temp_buffer[temp_buffer.size() / 2];
    }
    else
    {
      median_stabber = (temp_buffer[temp_buffer.size() / 2 - 1] +
                        temp_buffer[temp_buffer.size() / 2]) /
                       2.0;
    }

    stabber_clustered.push_back(median_stabber);
  }

  return stabber_clustered;
}

Eigen::Matrix3d RotFGO::axisAngleToRotMatrix(const Eigen::Vector3d &axis,
                                             double angle)
{
  Eigen::Vector3d unit_axis = axis.normalized();

  // Rodrigues' rotation formula
  Eigen::Matrix3d K; // skew-symmetric matrix
  K << 0, -unit_axis(2), unit_axis(1), unit_axis(2), 0, -unit_axis(0),
      -unit_axis(1), unit_axis(0), 0;

  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R = I + std::sin(angle) * K + (1 - std::cos(angle)) * K * K;

  return R;
}

Eigen::MatrixXd RotFGO::createKernelBuffer(const std::vector<int> &ids,
                                           bool use_saturated)
{
  int max_id = *std::max_element(ids.begin(), ids.end());
  int num_2d_lines = max_id + 1;

  // Count matches per 2D line
  std::vector<int> match_count(num_2d_lines, 0);
  for (int id : ids)
  {
    match_count[id]++;
  }

  int max_count = *std::max_element(match_count.begin(), match_count.end());
  Eigen::MatrixXd kernel_buffer =
      Eigen::MatrixXd::Zero(num_2d_lines, max_count);

  if (!use_saturated)
  {
    // Classic consensus maximization - all weights are 1
    kernel_buffer.setOnes();
  }
  else
  {
    // Saturated consensus maximization with entropy
    double L = 0.0;
    for (int count : match_count)
    {
      if (count > 0)
      {
        L += std::log(count);
      }
    }

    for (int i = 0; i < num_2d_lines; i++)
    {
      if (match_count[i] == 0)
        continue;

      kernel_buffer(i, 0) = 1.0 - std::log(match_count[i]) / L;
      for (int j = 1; j < match_count[i]; j++)
      {
        kernel_buffer(i, j) = (std::log(j + 1) - std::log(j)) / L;
      }
    }
  }

  return kernel_buffer;
}
