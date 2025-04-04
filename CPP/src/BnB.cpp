#include "BnB.hpp"
#include "helper.hpp"
#include "struct.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <numeric>
#include <queue>
#include <utility> // for std::pair
#include <vector>
using namespace std;
using namespace Eigen;

FGO_PnL::~FGO_PnL() {
  // Destructor implementation and Clean up resources if any
  cout << "FGO_PnL object is being deleted" << endl;
};
// data input
void FGO_PnL::data_load(vector<Vector3d> &line_2ds, vector<Vector3d> &line_3ds,
                        vector<Vector3d> &points_3d, vector<int> &line_tags) {
  int N = line_2ds.size();
  this->init_line_pairs.resize(N);
#pragma omp parallel for schedule(static) num_threads(4)
  for (auto i = 0; i < N; i++) {
    this->init_line_pairs[i].line_2d = line_2ds[i];
    this->init_line_pairs[i].line_3d = line_3ds[i];
    this->init_line_pairs[i].point_3d = points_3d[i];
    this->init_line_pairs[i].line_tag = line_tags[i];
    preprocess(this->init_line_pairs[i]);
  }

  return;
};
// data process
void FGO_PnL::preprocess(line_pair &lp) {
  Vector3d &vector_n = lp.line_2d;
  Vector3d &vector_v = lp.line_3d;
  lp.attribute->inner_product = vector_n.dot(vector_v);
  lp.attribute->outer_product = vector_v.cross(vector_n);
  lp.attribute->norm_of_outer_product = lp.attribute->outer_product.norm();
  Vector2d outer_angle = xyz_2_polar(lp.attribute->outer_product);

  if (outer_angle[1] > M_PI) {
    lp.attribute->east_or_not = false;
    lp.attribute->angle_west_outer_product = outer_angle;
    lp.attribute->angle_east_outer_product =
        Vector2d(M_PI - outer_angle[0], outer_angle[1] - M_PI);
    lp.attribute->east_outer_product = -lp.attribute->outer_product;
    lp.attribute->west_outer_product = lp.attribute->outer_product;
  } else {
    lp.attribute->east_or_not = true;
    lp.attribute->angle_west_outer_product =
        Vector2d(M_PI - outer_angle[0], outer_angle[1] + M_PI);
    lp.attribute->angle_east_outer_product = outer_angle;
    lp.attribute->east_outer_product = lp.attribute->outer_product;
    lp.attribute->west_outer_product = -lp.attribute->outer_product;
  }
  angle_bisectors(lp);
  return;
};
void FGO_PnL::rot_bnb_estimate() {
  range alpha(0, M_PI);
  range phi_1(0, M_PI);
  range phi_2(M_PI, 2 * M_PI);
  Square branch1(alpha, phi_1);
  Square branch2(alpha, phi_2);
  rot_bnb_epoch(branch1);
  rot_bnb_epoch(branch2);
  priority_queue<Square, vector<Square>> pq;
  pq.push(branch1);
  pq.push(branch2);
  lower_bound = max(branch1.lower_bound, branch2.lower_bound);
  // Vector3d I_find_u=Vector3d::Zero();
  upper_bound = pq.top().upper_bound;
  Vector3d I_find_u = Vector3d::Zero();
  while (!pq.empty()) {
    this->iter_count++;
    auto branch_i = pq.top();
    // cout<<"iteration: "<<this->iter_count;
    // cout<<" ["<<branch_i.alpha.lower/M_PI<<",
    // "<<branch_i.alpha.upper/M_PI<<", "<<branch_i.phi.lower/M_PI<<",
    // "<<branch_i.phi.upper/M_PI<<" ]"; cout<<"UB: "<<branch_i.upper_bound<<"
    // LB: "<<this->lower_bound<<endl;
    pq.pop();

    if (branch_i.upper_bound == lower_bound ||
        branch_i.alpha.width <= this->branch_resolution) {
      // vec_rot = I_find_u;
      this->R_hat =
          rodrigues(this->vec_rot[0], this->vec_rot[1], this->vec_rot[2]);
      cout << "output: alpha:" << this->vec_rot[0]
           << "  phi:" << this->vec_rot[1] << "  theta:" << this->vec_rot[2]
           << endl;
      cout << this->R_hat << endl;
      break;
    } else if (branch_i.upper_bound > lower_bound) {
      Square branch1(range(branch_i.alpha.lower, branch_i.alpha.center),
                     range(branch_i.phi.lower, branch_i.phi.center));
      Square branch2(range(branch_i.alpha.center, branch_i.alpha.upper),
                     range(branch_i.phi.lower, branch_i.phi.center));
      Square branch3(range(branch_i.alpha.lower, branch_i.alpha.center),
                     range(branch_i.phi.center, branch_i.phi.upper));
      Square branch4(range(branch_i.alpha.center, branch_i.alpha.upper),
                     range(branch_i.phi.center, branch_i.phi.upper));
      rot_bnb_epoch(branch1);
      rot_bnb_epoch(branch2);
      rot_bnb_epoch(branch3);
      rot_bnb_epoch(branch4);
      pq.push(branch1);
      pq.push(branch2);
      pq.push(branch3);
      pq.push(branch4);
    } else {
      continue;
    }
  };
  return;
};
void FGO_PnL::rot_bnb_epoch(Square &branch) {
  auto [h1_upper, h1_lower] = h1_bounds(branch);
  auto [h2_upper, h2_lower] = h2_bounds(branch);
  auto lower_params = generate_params(h1_lower, h2_lower);
  auto upper_params = generate_params(h1_upper, h2_upper);

  int N = init_line_pairs.size();
  VectorXd h1_center = VectorXd::Zero(N);
  VectorXd h2_center = VectorXd::Zero(N);
  VectorXd u_center =
      polar_2_xyz(Vector2d(branch.alpha.center, branch.phi.center));
#pragma omp parallel for schedule(static) num_threads(4)
  for (auto i = 0; i < N; i++) {
    h1_center[i] = u_center.dot(init_line_pairs[i].attribute->outer_product);
    h2_center[i] = u_center.dot(init_line_pairs[i].line_3d) *
                       u_center.dot(init_line_pairs[i].line_2d) -
                   init_line_pairs[i].attribute->inner_product;
  }
  auto center_params = generate_params(h1_center, h2_center);
  vector<double> upper_intervals;
  vector<double> lower_intervals;
  vector<int> line_tags_upper;
  vector<int> line_tags_lower;
#pragma omp parallel for schedule(static) num_threads(8)
  for (int i = 0; i < N; ++i) {
    // Upper intervals
    vector<double> tmp_interval;
    vector<double> tmp_interval2;
    tmp_interval = upper_interval(upper_params.row(i), lower_params.row(i));
    tmp_interval2 = lower_interval(center_params.row(i));
#pragma omp critical
    {
      upper_intervals.insert(upper_intervals.end(), tmp_interval.begin(),
                             tmp_interval.end());
      line_tags_upper.insert(line_tags_upper.end(), tmp_interval.size() / 2,
                             init_line_pairs[i].line_tag);
      // Lower intervals
      lower_intervals.insert(lower_intervals.end(), tmp_interval2.begin(),
                             tmp_interval2.end());
      line_tags_lower.insert(line_tags_lower.end(), tmp_interval2.size() / 2,
                             init_line_pairs[i].line_tag);
    }
  }
  auto [Q_upper, _] =
      satured_interval_stabbing(upper_intervals, line_tags_upper);
  auto [Q_lower, theta_lower] =
      satured_interval_stabbing(lower_intervals, line_tags_lower);

  branch.lower_bound = Q_lower;
  branch.upper_bound = Q_upper;
  branch.theta_hat = theta_lower;
  if (this->lower_bound < branch.lower_bound) {
    this->lower_bound = branch.lower_bound;
    this->vec_rot[0] = branch.center[0];
    this->vec_rot[1] = branch.center[1];
    this->vec_rot[2] = branch.theta_hat;
  }
  return;
};

pair<VectorXd, VectorXd> FGO_PnL::h1_bounds(Square &branch) {
  int N = init_line_pairs.size(); // 假设line_pair是类成员变量
  VectorXd upper = VectorXd::Zero(N);
  VectorXd lower = VectorXd::Zero(N);

  const double &cube_width = branch.alpha.width;

  const range &range_alpha = branch.alpha;
  const range &range_phi = branch.phi;

  if (cube_width <= sample_resolution) {
#pragma omp parallel for schedule(static) num_threads(8)
    for (int i = 0; i < N; ++i) {
      Vector2d outer_angle;
      Vector3d outer_product;
      int flag = 1;
      if ((range_phi.upper > M_PI &&
           !init_line_pairs[i].attribute->east_or_not) ||
          (range_phi.upper <= M_PI &&
           init_line_pairs[i].attribute->east_or_not)) {
        flag = 1;
      } else {
        flag = -1;
      }
      if (range_phi.lower <= M_PI) {
        outer_angle = init_line_pairs[i].attribute->angle_east_outer_product;
        outer_product = init_line_pairs[i].attribute->east_outer_product;
      } else {
        outer_angle = init_line_pairs[i].attribute->angle_west_outer_product;
        outer_product = init_line_pairs[i].attribute->west_outer_product;
      }

      auto [phi_far, phi_near] = interval_projection(outer_angle[1], range_phi);
      auto [alpha_far, alpha_near] =
          interval_projection(outer_angle[0], range_alpha);
      double delta_phi_near = abs(phi_near - outer_angle[1]);
      double maximum = 0;
      double minimum = 0;
      if (delta_phi_near == 0) {
        maximum = outer_product.dot(polar_2_xyz(alpha_near, phi_near));
      } else {
        maximum =
            max(outer_product.dot(polar_2_xyz(range_alpha.lower, phi_near)),
                outer_product.dot(polar_2_xyz(range_alpha.upper, phi_near)));
      }
      double delta_phi_far = abs(phi_far - outer_angle[1]);
      minimum = min(outer_product.dot(polar_2_xyz(range_alpha.lower, phi_far)),
                    outer_product.dot(polar_2_xyz(range_alpha.upper, phi_far)));
      if (flag == 1) {
        upper(i) = maximum;
        lower(i) = minimum;
      } else {
        upper(i) = -minimum;
        lower(i) = -maximum;
      }
    }
  } else {
#pragma omp parallel for schedule(static) num_threads(8)
    for (int i = 0; i < N; ++i) {
      Vector2d outer_angle;
      Vector3d outer_product;
      int flag = 1;
      if ((range_phi.upper > M_PI &&
           !init_line_pairs[i].attribute->east_or_not) ||
          (range_phi.upper <= M_PI &&
           init_line_pairs[i].attribute->east_or_not)) {
        flag = 1;
      } else {
        flag = -1;
      }
      if (range_phi.lower <= M_PI) {
        outer_angle = init_line_pairs[i].attribute->angle_east_outer_product;
        outer_product = init_line_pairs[i].attribute->east_outer_product;
      } else {
        outer_angle = init_line_pairs[i].attribute->angle_west_outer_product;
        outer_product = init_line_pairs[i].attribute->west_outer_product;
      }

      auto [phi_far, phi_near] = interval_projection(outer_angle[1], range_phi);
      auto [alpha_far, alpha_near] =
          interval_projection(outer_angle[0], range_alpha);

      // Maximum calculation
      double maximum = 0;
      double delta_phi_near = std::abs(phi_near - outer_angle[1]);
      if (delta_phi_near == 0) {
        maximum = outer_product.dot(polar_2_xyz(alpha_near, phi_near));
      } else if (delta_phi_near > M_PI / 2) {
        double tangent = std::tan(outer_angle[0]) * std::cos(delta_phi_near);
        double max_alpha = (tangent > 1e8) ? M_PI / 2 : std::atan(tangent);
        if (max_alpha < 0)
          max_alpha += M_PI;

        max_alpha = (max_alpha <= range_alpha.center) ? range_alpha.upper
                                                      : range_alpha.lower;
        maximum = outer_product.dot(polar_2_xyz(max_alpha, phi_near));
      } else if (delta_phi_near < M_PI / 2 && outer_angle[0] < M_PI / 2 &&
                 range_alpha.lower >= outer_angle[0]) {
        maximum = outer_product.dot(polar_2_xyz(range_alpha.lower, phi_near));
      } else if (delta_phi_near < M_PI / 2 && outer_angle[0] > M_PI / 2 &&
                 range_alpha.lower <= M_PI - outer_angle[0]) {
        maximum = outer_product.dot(polar_2_xyz(range_alpha.upper, phi_near));
      } else if (delta_phi_near == M_PI / 2) {
        maximum =
            (outer_angle[0] <= M_PI / 2)
                ? outer_product.dot(polar_2_xyz(range_alpha.lower, phi_near))
                : outer_product.dot(polar_2_xyz(range_alpha.upper, phi_near));
      } else {
        double tangent = std::tan(outer_angle[0]) * std::cos(delta_phi_near);
        double max_alpha = (tangent > 1e8) ? M_PI / 2 : std::atan(tangent);
        if (max_alpha < 0)
          max_alpha += M_PI;
        if (max_alpha <= range_alpha.lower) {
          maximum = outer_product.dot(polar_2_xyz(range_alpha.lower, phi_near));
        } else if (max_alpha <= range_alpha.upper) {
          maximum = outer_product.dot(polar_2_xyz(max_alpha, phi_near));
        } else {
          maximum = outer_product.dot(polar_2_xyz(range_alpha.upper, phi_near));
        }
      }

      // Minimum calculation
      double minimum = 0;
      double delta_phi_far = std::abs(phi_far - outer_angle[1]);
      // ... 类似maximum的计算逻辑
      if (delta_phi_far < M_PI / 2) {
        double tangent = std::tan(outer_angle[0]) * std::cos(delta_phi_far);
        double min_alpha = (tangent > 1e8) ? M_PI / 2 : std::atan(tangent);
        if (min_alpha < 0)
          min_alpha += M_PI;

        if (min_alpha <=
            range_alpha.center) { // sum(range_alpha)/2 用center代替
          minimum = outer_product.dot(polar_2_xyz(range_alpha.lower, phi_far));
        } else {
          minimum = outer_product.dot(polar_2_xyz(range_alpha.upper, phi_far));
        }
      } else if (delta_phi_far > M_PI / 2 && outer_angle[0] < M_PI / 2 &&
                 range_alpha.upper <= (M_PI - outer_angle[0])) {
        minimum = outer_product.dot(polar_2_xyz(range_alpha.upper, phi_far));
      } else if (delta_phi_far > M_PI / 2 && outer_angle[0] > M_PI / 2 &&
                 range_alpha.lower >= (M_PI - outer_angle[0])) {
        minimum = outer_product.dot(polar_2_xyz(range_alpha.lower, phi_far));
      } else if (delta_phi_far == M_PI / 2) {
        minimum =
            (outer_angle[0] <= M_PI / 2)
                ? outer_product.dot(polar_2_xyz(range_alpha.upper, phi_far))
                : outer_product.dot(polar_2_xyz(range_alpha.lower, phi_far));
      } else {
        double tangent = std::tan(outer_angle[0]) * std::cos(delta_phi_far);
        double min_alpha = (tangent > 1e8) ? M_PI / 2 : std::atan(tangent);

        if (min_alpha < 0)
          min_alpha += M_PI;

        if (min_alpha <= range_alpha.lower) {
          minimum = outer_product.dot(polar_2_xyz(range_alpha.lower, phi_far));
        } else if (min_alpha <= range_alpha.upper) {
          minimum = outer_product.dot(polar_2_xyz(min_alpha, phi_far));
        } else {
          minimum = outer_product.dot(polar_2_xyz(range_alpha.upper, phi_far));
        }
      }
      if (flag == 1) {
        upper(i) = maximum;
        lower(i) = minimum;
      } else {
        upper(i) = -minimum;
        lower(i) = -maximum;
      }
    }
  }

  return {upper, lower};
};

pair<VectorXd, VectorXd> FGO_PnL::h2_bounds(Square &branch) {
  int N = this->init_line_pairs.size();
  VectorXd upper = VectorXd::Zero(N);
  VectorXd lower = VectorXd::Zero(N);
  double cube_width = branch.alpha.upper - branch.alpha.lower;
  MatrixXd vertex_cache;

  // Generate vertex cache
  if (cube_width <= sample_resolution) {
    // Add four vertices
    vertex_cache.resize(3, 4);
    vertex_cache.col(0) = polar_2_xyz(branch.alpha.lower, branch.phi.lower);
    vertex_cache.col(1) = polar_2_xyz(branch.alpha.upper, branch.phi.lower);
    vertex_cache.col(2) = polar_2_xyz(branch.alpha.lower, branch.phi.upper);
    vertex_cache.col(3) = polar_2_xyz(branch.alpha.upper, branch.phi.upper);
  } else {
    // Generate alpha and phi ranges
    auto generate_range = [](double start, double end, double step) {
      vector<double> res;
      for (double v = start; v <= end + 1e-9; v += step)
        res.push_back(v);
      return res;
    };

    vector<double> alpha = generate_range(
        branch.alpha.lower, branch.alpha.upper, sample_resolution);
    vector<double> phi =
        generate_range(branch.phi.lower, branch.phi.upper, sample_resolution);
    int temp = alpha.size() - 1;
    vertex_cache.resize(3, 4 * temp);

    // Part 1: alpha[0..temp-1], phi[0]
    for (int i = 0; i < temp; ++i)
      vertex_cache.col(i) = polar_2_xyz(alpha[i], phi[0]);

    // Part 2: alpha[end], phi[0..temp-1]
    double &a_end = alpha.back();
    for (int i = 0; i < temp; ++i)
      vertex_cache.col(temp + i) = polar_2_xyz(a_end, phi[i]);

    // Part 3: alpha[1..end], phi[end]
    double &p_end = phi.back();
    for (int i = 1; i <= temp; ++i)
      vertex_cache.col(2 * temp + i - 1) = polar_2_xyz(alpha[i], p_end);

    // Part 4: alpha[0], phi[1..end]
    double &a_start = alpha[0];
    for (int i = 1; i <= temp; ++i)
      vertex_cache.col(3 * temp + i - 1) = polar_2_xyz(a_start, phi[i]);
  }

  // Process each line pair
#pragma omp parallel for schedule(static) num_threads(12)
  for (int i = 0; i < N; ++i) {
    double maximum, minimum;
    Vector2d normal_angle, o_normal_angle;
    Vector3d normal_vector, o_normal_vector;
    Vector3d &n_i = init_line_pairs[i].line_2d;
    Vector3d &v_i = init_line_pairs[i].line_3d;
    double &inner_product = init_line_pairs[i].attribute->inner_product;
    if (branch.phi.lower < M_PI) {
      normal_angle = init_line_pairs[i].attribute->angle_east_angle_bisector;
      normal_vector = init_line_pairs[i].attribute->east_angle_bisector;
      o_normal_angle =
          init_line_pairs[i].attribute->angle_east_othogonal_angle_bisector;
      o_normal_vector =
          init_line_pairs[i].attribute->east_othogonal_angle_bisector;
    } else {
      normal_angle = init_line_pairs[i].attribute->angle_west_angle_bisector;
      normal_vector = init_line_pairs[i].attribute->west_angle_bisector;
      o_normal_angle =
          init_line_pairs[i].attribute->angle_west_othogonal_angle_bisector;
      o_normal_vector =
          init_line_pairs[i].attribute->west_othogonal_angle_bisector;
    }

    // Check bounds
    bool normal_in = (normal_angle[0] >= branch.alpha.lower) &&
                     (normal_angle[0] <= branch.alpha.upper) &&
                     (normal_angle[1] >= branch.phi.lower) &&
                     (normal_angle[1] <= branch.phi.upper);

    bool o_normal_in = (o_normal_angle[0] >= branch.alpha.lower) &&
                       (o_normal_angle[0] <= branch.alpha.upper) &&
                       (o_normal_angle[1] >= branch.phi.lower) &&
                       (o_normal_angle[1] <= branch.phi.upper);

    int flag = (normal_in ? 2 : 0) + (o_normal_in ? 1 : 0);

    switch (flag) {
    case 3: {
      double max_val = normal_vector.dot(n_i) * normal_vector.dot(v_i);
      double min_val = o_normal_vector.dot(n_i) * o_normal_vector.dot(v_i);
      upper[i] = max_val - inner_product;
      lower[i] = min_val - inner_product;
      continue; // Skip to next iteration
    }
    case 2:
      maximum = normal_vector.dot(n_i) * normal_vector.dot(v_i);
      break;
    case 1:
      minimum = o_normal_vector.dot(n_i) * o_normal_vector.dot(v_i);
      break;
    }

    // Find min/max in vertex cache
    double tmp_max, tmp_min;
    RowVectorXd tmp = (n_i.transpose() * vertex_cache).array() *
                      (v_i.transpose() * vertex_cache).array();
    tmp_max = tmp.maxCoeff();
    tmp_min = tmp.minCoeff();
    // Determine bounds
    switch (flag) {
    case 2:
      upper[i] = maximum;
      lower[i] = tmp_min;
      break;
    case 1:
      upper[i] = tmp_max;
      lower[i] = minimum;
      break;
    default:
      upper[i] = tmp_max;
      lower[i] = tmp_min;
    }

    upper[i] -= inner_product;
    lower[i] -= inner_product;
  }
  return {upper, lower};
};
vector<double> FGO_PnL::upper_interval(RowVector3d coef_1, RowVector3d coef_2) {
  double A_1 = coef_1[0];
  double phi_1 = coef_1[1];
  double const_1 = coef_1[2];
  double A_2 = coef_2[0];
  double phi_2 = coef_2[1];
  double const_2 = coef_2[2];
  vector<double> interval;

  // Upper intervals calculation (f_max >= -rotation_epsilon)
  vector<double> upper_intervals;
  double c_lo = -const_1 - this->rotation_epsilon;

  if (A_1 < c_lo) {
    return interval;
  } else if (c_lo >= 0) {
    double x_l = asin(c_lo / A_1);
    if (phi_1 <= M_PI - x_l) {
      upper_intervals.push_back(max(0.0, x_l - phi_1));
      upper_intervals.push_back(M_PI - x_l - phi_1);
    } else if (phi_1 >= M_PI + x_l) {
      upper_intervals.push_back(2 * M_PI + x_l - phi_1);
      upper_intervals.push_back(min(M_PI, 3 * M_PI - x_l - phi_1));
    } else {
      return interval;
    }
  } else if (c_lo >= -A_1) {
    double x = asin(c_lo / A_1);
    double x_l = M_PI - x;
    double x_r = 2 * M_PI + x;
    if (phi_1 <= x_r - M_PI) {
      upper_intervals.push_back(0.0);
      upper_intervals.push_back(min(x_l - phi_1, M_PI));
    } else if (phi_1 >= x_l) {
      upper_intervals.push_back(max(0.0, x_r - phi_1));
      upper_intervals.push_back(M_PI);
    } else {
      upper_intervals.push_back(0.0);
      upper_intervals.push_back(x_l - phi_1);
      upper_intervals.push_back(x_r - phi_1);
      upper_intervals.push_back(M_PI);
    }
  } else {
    upper_intervals.push_back(0.0);
    upper_intervals.push_back(M_PI);
  }

  // Lower intervals calculation (f_min <= rotation_epsilon)
  vector<double> lower_intervals;
  double c_up = this->rotation_epsilon - const_2;

  if (A_2 <= c_up) {
    lower_intervals.push_back(0.0);
    lower_intervals.push_back(M_PI);
  } else if (c_up >= 0) {
    double x_l = asin(c_up / A_2);
    if (phi_2 <= x_l) {
      lower_intervals.push_back(0.0);
      lower_intervals.push_back(x_l - phi_2);
      lower_intervals.push_back(M_PI - x_l - phi_2);
      lower_intervals.push_back(M_PI);
    } else if (phi_2 <= 2 * M_PI - x_l) {
      lower_intervals.push_back(max(0.0, M_PI - x_l - phi_2));
      lower_intervals.push_back(min(M_PI, 2 * M_PI + x_l - phi_2));
    } else {
      lower_intervals.push_back(0.0);
      lower_intervals.push_back(2 * M_PI + x_l - phi_2);
      lower_intervals.push_back(3 * M_PI - x_l - phi_2);
      lower_intervals.push_back(M_PI);
    }
  } else if (c_up >= -A_2) {
    double x = asin(c_up / A_2);
    double x_l = M_PI - x;
    double x_r = 2 * M_PI + x;
    if (phi_2 <= -x || phi_2 >= x_r) {
      return interval;
    } else {
      lower_intervals.push_back(max(0.0, x_l - phi_2));
      lower_intervals.push_back(min(M_PI, x_r - phi_2));
    }
  } else {
    return interval;
  }

  // TODO
  for (size_t i = 0; i < upper_intervals.size(); i += 2) {
    if (i + 1 >= upper_intervals.size())
      break;
    for (size_t j = 0; j < lower_intervals.size(); j += 2) {
      if (j + 1 >= lower_intervals.size())
        break;
      vector<double> intersect =
          interval_intersection(upper_intervals[i], upper_intervals[i + 1],
                                lower_intervals[j], lower_intervals[j + 1]);
      interval.insert(interval.end(), intersect.begin(), intersect.end());
    }
  }
  return interval;
};
vector<double> FGO_PnL::lower_interval(RowVector3d coef) {
  double A = coef[0];
  double phi = coef[1];
  double const_ = coef[2];
  vector<double> interval;

  double c_up = -const_ + this->rotation_epsilon;
  double c_lo = -const_ - this->rotation_epsilon;

  if (c_up <= -A) {
    return interval;
  } else if (c_up <= 0) {
    if (c_lo <= -A) {
      double m = asin(c_up / A);
      double m_l = M_PI - m;
      double m_r = 2 * M_PI + m;
      if (phi <= -m || phi >= m_r) {
        return interval;
      } else {
        interval.push_back(max(0.0, m_l - phi));
        interval.push_back(min(M_PI, m_r - phi));
      }
    } else {
      double m = asin(c_up / A);
      double n = asin(c_lo / A);
      double m_l = M_PI - m;
      double n_l = M_PI - n;
      double m_r = 2 * M_PI + n;
      double n_r = 2 * M_PI + m;
      if (phi <= -m || phi >= n_r) {
        return interval;
      } else if (phi <= M_PI + n) {
        interval.push_back(m_l - phi);
        interval.push_back(min(M_PI, n_l - phi));
      } else if (phi <= n_l) {
        interval.push_back(max(m_l - phi, 0.0));
        interval.push_back(n_l - phi);
        interval.push_back(m_r - phi);
        interval.push_back(min(M_PI, n_r - phi));
      } else {
        interval.push_back(max(m_r - phi, 0.0));
        interval.push_back(min(M_PI, n_r - phi));
      }
    }
  } else if (c_up <= A) {
    if (c_lo <= -A) {
      double m = asin(c_up / A);
      if (phi <= m) {
        interval.push_back(0.0);
        interval.push_back(m - phi);
        interval.push_back(M_PI - m - phi);
        interval.push_back(M_PI);
      } else if (phi <= 2 * M_PI - m) {
        interval.push_back(max(0.0, M_PI - m - phi));
        interval.push_back(min(M_PI, 2 * M_PI + m - phi));
      } else {
        interval.push_back(0.0);
        interval.push_back(2 * M_PI + m - phi);
        interval.push_back(3 * M_PI - m - phi);
        interval.push_back(M_PI);
      }
    } else if (c_lo <= 0) {
      double m = asin(c_up / A);
      double n = asin(c_lo / A);
      double m_r = M_PI - m;
      double n_l = M_PI - n;
      double n_r = 2 * M_PI + n;
      if (phi < m) {
        interval.push_back(0.0);
        interval.push_back(m - phi);
        interval.push_back(m_r - phi);
        interval.push_back(min(M_PI, n_l - phi));
      } else if (phi <= n_r - M_PI) {
        interval.push_back(max(0.0, m_r - phi));
        interval.push_back(min(M_PI, n_l - phi));
      } else if (phi <= n_l) {
        interval.push_back(max(0.0, m_r - phi));
        interval.push_back(n_l - phi);
        interval.push_back(n_r - phi);
        interval.push_back(min(M_PI, 2 * M_PI + m - phi));
      } else if (phi <= m_r + M_PI) {
        interval.push_back(max(0.0, n_r - phi));
        interval.push_back(min(M_PI, 2 * M_PI + m - phi));
      } else {
        interval.push_back(max(0.0, n_r - phi));
        interval.push_back(2 * M_PI + m - phi);
        interval.push_back(3 * M_PI - m - phi);
        interval.push_back(M_PI);
      }
    } else {
      double m = asin(c_up / A);
      double n = asin(c_lo / A);
      double m_r_1 = M_PI - m;
      double n_r_1 = M_PI - n;
      if (phi <= m) {
        interval.push_back(max(0.0, n - phi));
        interval.push_back(m - phi);
        interval.push_back(m_r_1 - phi);
        interval.push_back(n_r_1 - phi);
      } else if (phi <= M_PI + n && phi >= n_r_1) {
        return interval;
      } else if (phi <= n_r_1) {
        interval.push_back(max(0.0, m_r_1 - phi));
        interval.push_back(n_r_1 - phi);
      } else if (phi <= m_r_1 + M_PI) {
        interval.push_back(2 * M_PI + n - phi);
        interval.push_back(min(M_PI, 2 * M_PI + m - phi));
      } else {
        interval.push_back(2 * M_PI + n - phi);
        interval.push_back(2 * M_PI + m - phi);
        interval.push_back(m_r_1 + 2 * M_PI - phi);
        interval.push_back(min(M_PI, n_r_1 + 2 * M_PI - phi));
      }
    }
  } else {
    if (c_lo <= -A) {
      interval.push_back(0.0);
      interval.push_back(M_PI);
    } else if (c_lo <= 0) {
      double m = asin(c_lo / A);
      double m_l = M_PI - m;
      double m_r = 2 * M_PI + m;
      if (phi <= m_r - M_PI) {
        interval.push_back(0.0);
        interval.push_back(min(m_l - phi, M_PI));
      } else if (phi >= m_l) {
        interval.push_back(max(0.0, m_r - phi));
        interval.push_back(M_PI);
      } else {
        interval.push_back(0.0);
        interval.push_back(m_l - phi);
        interval.push_back(m_r - phi);
        interval.push_back(M_PI);
      }
    } else if (c_lo < A) {
      double m = asin(c_lo / A);
      if (phi <= M_PI - m) {
        interval.push_back(max(0.0, m - phi));
        interval.push_back(M_PI - m - phi);
      } else if (phi >= M_PI + m) {
        interval.push_back(2 * M_PI + m - phi);
        interval.push_back(min(M_PI, 3 * M_PI - m - phi));
      } else {
        return interval;
      }
    } else {
      return interval;
    }
  }
  return interval;
};
pair<double, double>
FGO_PnL::satured_interval_stabbing(vector<double> &interval,
                                   vector<int> line_tags) {
  if (line_tags.empty() || interval.empty()) {
    return {0.0, 0.0};
  }
  int L = interval.size() / 2;
  vector<int> masks;
  masks.reserve(interval.size());
  for (int i = 0; i < L; ++i) {
    masks.push_back(0); // Start point
    masks.push_back(1); // End point
  }

  vector<int> sidx(interval.size());
  iota(sidx.begin(), sidx.end(), 0);
  sort(sidx.begin(), sidx.end(),
       [&](double a, double b) { return interval[a] < interval[b]; });
  unordered_map<int, int> count_buffer; // 改用哈希表存储计数

  double weighted_count = 0.0;
  double weighted_num_stabbed = 0.0;
  double stabber = 0.0;

  for (int i = 0; i < sidx.size(); ++i) {
    int pos = sidx[i];
    if (masks[pos] == 0) { // Start point
      int interval_idx = pos / 2;
      int temp_id = line_tags[interval_idx];
      int &cnt = count_buffer[temp_id];
      ++cnt;
      weighted_count += this->satured_kernel_cache[min(cnt, 15) - 1];

      // 合并后（1行）
      if (weighted_count > weighted_num_stabbed) {
        weighted_num_stabbed = weighted_count;
        stabber = interval[pos] + 1e-12;
      }
    } else { // End point
      int interval_idx = (pos - 1) / 2;
      int temp_id = line_tags[interval_idx];
      int &cnt = count_buffer[temp_id];
      weighted_count -= this->satured_kernel_cache[min(cnt, 15) - 1];
      cnt--;
    }
  }

  return {weighted_num_stabbed, stabber};
};
// Transition function
void FGO_PnL::angle_bisectors(line_pair &lp) {
  const Eigen::Vector3d &v1 = lp.line_2d;
  const Eigen::Vector3d &v2 = lp.line_3d;

  Vector2d normal = Vector2d::Zero();
  Vector2d o_normal = Vector2d::Zero();
  if (lp.attribute->inner_product >= 0.99999) {
    normal = xyz_2_polar(v1);
    o_normal = Vector2d(M_PI_2, normal(1) + M_PI_2);
  } else if (lp.attribute->inner_product <= -0.99999) {
    normal = Vector2d(M_PI_2, normal(1) + M_PI_2);
    o_normal = xyz_2_polar(v1);
  } else {
    Eigen::Vector3d mid = (v1 + v2) / 2.0;
    mid.normalize();
    Eigen::Vector3d n = v1.cross(v2);
    Eigen::Vector3d orthogonal = mid.cross(n);
    orthogonal.normalize();
    normal = xyz_2_polar(mid);
    o_normal = xyz_2_polar(orthogonal);
  }
  // Set normal directions
  if (normal[1] > M_PI) {
    lp.attribute->angle_west_angle_bisector = normal;
    lp.attribute->angle_east_angle_bisector =
        Vector2d(M_PI - normal[0], normal[1] - M_PI);
    lp.attribute->east_angle_bisector =
        polar_2_xyz(lp.attribute->angle_east_angle_bisector);
    lp.attribute->west_angle_bisector =
        polar_2_xyz(lp.attribute->angle_west_angle_bisector);
  } else {
    lp.attribute->angle_west_angle_bisector =
        Vector2d(M_PI - normal[0], normal[1] + M_PI);
    lp.attribute->angle_east_angle_bisector = normal;
    lp.attribute->east_angle_bisector =
        polar_2_xyz(lp.attribute->angle_east_angle_bisector);
    lp.attribute->west_angle_bisector =
        polar_2_xyz(lp.attribute->angle_west_angle_bisector);
  }
  if (o_normal[1] > M_PI) {
    lp.attribute->angle_west_othogonal_angle_bisector = o_normal;
    lp.attribute->angle_east_othogonal_angle_bisector =
        Vector2d(M_PI - o_normal[0], o_normal[1] - M_PI);
    lp.attribute->east_othogonal_angle_bisector =
        polar_2_xyz(lp.attribute->angle_east_othogonal_angle_bisector);
    lp.attribute->west_othogonal_angle_bisector =
        polar_2_xyz(lp.attribute->angle_west_othogonal_angle_bisector);
  } else {
    lp.attribute->angle_west_othogonal_angle_bisector =
        Vector2d(M_PI - o_normal[0], o_normal[1] + M_PI);
    lp.attribute->angle_east_othogonal_angle_bisector = o_normal;
    lp.attribute->east_othogonal_angle_bisector =
        polar_2_xyz(lp.attribute->angle_east_othogonal_angle_bisector);
    lp.attribute->west_othogonal_angle_bisector =
        polar_2_xyz(lp.attribute->angle_west_othogonal_angle_bisector);
  }
  return;
};
MatrixXd FGO_PnL::generate_params(VectorXd &lower, VectorXd &upper) {
  int N = this->init_line_pairs.size();
  VectorXd inner_product = VectorXd::Zero(N);
  MatrixXd params = MatrixXd::Zero(N, 3);

#pragma omp parallel for schedule(static) num_threads(4)
  for (int i = 0; i < N; i++) {
    inner_product(i) = this->init_line_pairs[i].attribute->inner_product;
  }

  VectorXd h1 = lower.array();
  VectorXd h2 = upper.array();
  VectorXd A = (h1.array().square() + h2.array().square()).sqrt();
  VectorXd phi =
      (-h2).binaryExpr(h1, [](double a, double b) { return atan2(a, b); });
  phi = phi.unaryExpr([](double x) { return x < 0 ? x + 2 * M_PI : x; });
  VectorXd constant = inner_product + h2;

  params.col(0) = A;
  params.col(1) = phi;
  params.col(2) = constant;
  return params;
};