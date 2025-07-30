#include "TransFGO.h"
#include "SatIS.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <queue>
#include <future>
#include <array>

constexpr double PI = M_PI;

// Solve for translation candidates using branch-and-bound
std::vector<Eigen::Vector3d>
TransFGO::getCandidates(const std::vector<Eigen::Vector3d> &pert_rot_n_2D,
                        const std::vector<Eigen::Vector3d> &endpoints_3D,
                        const std::vector<int> &ids,
                        const Eigen::MatrixXd &kernel_buffer)
{
    // Step 1: Prepare data
    const size_t num_lines = endpoints_3D.size() / 2;
    std::vector<Eigen::Vector3d> p_3D(num_lines);
    for (size_t i = 0; i < num_lines; ++i)
    {
        p_3D[i] = endpoints_3D[i * 2]; // Take every first endpoint
    }

    // Step 2: Initialize branch queue with 1m x 1m cubes
    BranchQueue q;
    double best_lb = -1.0;
    int ceil_y = static_cast<int>(std::ceil(space_size_(1)));
    int ceil_z = static_cast<int>(std::ceil(space_size_(2)));

    // Initialize branches with 1m x 1m cubes
    for (int i = 0; i < ceil_y; i++)
    {
        for (int j = 0; j < ceil_z; j++)
        {
            TBranch b(i, j, i + 1, j + 1);
            double upper_bound = calcUB(pert_rot_n_2D, p_3D, ids, b, kernel_buffer);
            if (upper_bound > 0)
            {
                b.upper_bound = upper_bound;
                q.push(b);
            }
        }
    }

    // Step 3: Main branch-and-bound loop
    std::vector<TBranch> best_branches;
    const int max_iter = 500;
    const double eps = 1e-8;

    int iter = 0;
    while (!q.empty() && iter++ < max_iter)
    {
        // Pop branch with highest upper bound
        TBranch cur_b = q.top();
        q.pop();

        // Calculate lower bound
        auto [lb, x_opts] = calcLB(pert_rot_n_2D, p_3D, ids, cur_b, kernel_buffer);

        // Update best lower bound
        if (lb > best_lb + eps)
        {
            best_lb = lb;
            best_branches = {cur_b};
        }
        else if (std::abs(lb - best_lb) <= eps)
        {
            best_branches.emplace_back(cur_b);
        }

        // Prune branches with upper bound < best_lb
        BranchQueue new_q;
        while (!q.empty())
        {
            TBranch b = q.top();
            q.pop();
            if (b.upper_bound >= best_lb - eps)
            {
                new_q.push(b);
            }
            else
            {
                break; // No need to check further, queue is sorted
            }
        }
        q = std::move(new_q);

        // Check termination conditions
        double branch_size = std::max(cur_b.y_max - cur_b.y_min, cur_b.z_max - cur_b.z_min);
        if (branch_size < branch_resolution_)
            continue;

        if (cur_b.upper_bound < best_lb + eps && lb + eps > best_lb)
            continue;

        // Subdivide current branch
        std::vector<TBranch> sub_bs = cur_b.subDivide();
        for (auto &sb : sub_bs)
        {
            double ub = calcUB(pert_rot_n_2D, p_3D, ids, sb, kernel_buffer);
            if (ub >= best_lb - eps)
            {
                sb.upper_bound = ub;
                q.push(sb);
            }
        }
    }

    // Step 4: Generate final translation candidates
    std::vector<Eigen::Vector3d> t_candidates;
    for (const auto &best_branch : best_branches)
    {
        Eigen::Vector2d yz_center((best_branch.y_min + best_branch.y_max) / 2.0,
                                  (best_branch.z_min + best_branch.z_max) / 2.0);
        auto [score, x_opts] = calcLB(pert_rot_n_2D, p_3D, ids, best_branch, kernel_buffer);

        std::vector<double> clustered_x = SatIS::clusterStabber(x_opts, prox_threshold_);
        for (double x_opt : clustered_x)
        {
            t_candidates.emplace_back(x_opt, yz_center(0), yz_center(1));
        }
    }

    std::cout << "Translation estimation found " << t_candidates.size() << " candidates" << std::endl;
    return t_candidates;
}

// Complete translation estimation pipeline
Eigen::Vector3d TransFGO::solveWithPreprocessedData(const std::vector<Eigen::Vector3d> &pert_rot_n_2D,
                                                    const std::vector<Eigen::Vector3d> &endpoints_3D,
                                                    const std::vector<int> &ids,
                                                    const Eigen::Matrix3d &R_opt,
                                                    const Eigen::Matrix3d &intrinsic)
{
    // Step 1: Create kernel buffer once for both getCandidates and pruneTCandidates
    auto t_kernel_start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd kernel_buffer = createKernelBuffer(ids);
    auto t_kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kernel_duration = t_kernel_end - t_kernel_start;
    std::cout << "[BENCH] Translation kernel buffer creation: " << kernel_duration.count() << " ms" << std::endl;

    // Step 2: Get translation candidates
    std::vector<Eigen::Vector3d> t_candidates = getCandidates(pert_rot_n_2D, endpoints_3D, ids, kernel_buffer);

    if (t_candidates.empty())
    {
        std::cout << "No translation candidates found!" << std::endl;
        return Eigen::Vector3d::Zero();
    }

    // Step 3: Prune candidates and find best one
    auto [best_score, t_best_candidate] = pruneTCandidates(R_opt, intrinsic,
                                                           pert_rot_n_2D, endpoints_3D, ids,
                                                           t_candidates, kernel_buffer);

    // Step 4: Fine-tune translation by minimizing squared loss of inliers
    std::vector<Eigen::Vector3d> p_3D_for_tuning;
    for (size_t i = 0; i < endpoints_3D.size(); i += 2)
    {
        p_3D_for_tuning.emplace_back(endpoints_3D[i]);
    }

    std::vector<Eigen::Vector3d> t_candidate_vector = {t_best_candidate};
    Eigen::Vector3d t_fine_tuned = fineTuneTranslation(t_candidate_vector, pert_rot_n_2D, p_3D_for_tuning);

    std::cout << "Translation pipeline completed:" << std::endl;
    std::cout << "  Candidates: " << t_candidates.size() << std::endl;
    std::cout << "  Best score: " << best_score << std::endl;
    std::cout << "  Final translation: " << t_fine_tuned.transpose() << std::endl;

    return t_fine_tuned;
}

// Complete translation estimation pipeline with internal preprocessing
Eigen::Vector3d TransFGO::solve(const std::vector<int> &ids,
                                const Eigen::Matrix3d &R_opt,
                                const std::vector<Eigen::Vector3d> &v_3D,
                                const std::vector<Eigen::Vector3d> &n_2D,
                                const std::vector<Eigen::Vector3d> &endpoints_3D,
                                double epsilon_r,
                                const Eigen::Matrix3d &intrinsic)
{
    // Step 1: Preprocess rotation results to get inliers
    std::vector<Eigen::Vector3d> pert_rot_n_2D_inlier;
    std::vector<Eigen::Vector3d> endpoints_3D_inlier;
    std::vector<int> id_inliers_under_rot;

    preprocessRotation(ids, R_opt, v_3D, n_2D, endpoints_3D, epsilon_r,
                       pert_rot_n_2D_inlier, endpoints_3D_inlier, id_inliers_under_rot);

    if (pert_rot_n_2D_inlier.empty())
    {
        std::cout << "No inliers found for translation estimation!" << std::endl;
        return Eigen::Vector3d::Zero();
    }

    // Step 2: Use the preprocessed data to solve for translation
    return solveWithPreprocessedData(pert_rot_n_2D_inlier, endpoints_3D_inlier,
                                     id_inliers_under_rot, R_opt, intrinsic);
}

Eigen::MatrixXd TransFGO::createKernelBuffer(const std::vector<int> &ids)
{
    if (ids.empty())
    {
        return Eigen::MatrixXd::Zero(1, 1);
    }

    int max_id = *std::max_element(ids.begin(), ids.end());
    std::vector<int> match_count(max_id + 1, 0);

    // Count matches for each 2D line
    for (int id : ids)
    {
        match_count[id]++;
    }

    int max_matches = *std::max_element(match_count.begin(), match_count.end());
    Eigen::MatrixXd kernel_buffer = Eigen::MatrixXd::Zero(max_id + 1, max_matches);

    if (use_saturated_)
    {
        // Saturated consensus maximization
        double q = 0.9;
        double L_trans = (1.0 / epsilon_t_) * q / (1.0 - q);

        for (int i = 0; i <= max_id; i++)
        {
            if (match_count[i] == 0)
                continue;
            for (int j = 1; j <= match_count[i]; j++)
            {
                kernel_buffer(i, j - 1) = std::log(1.0 + L_trans * j / match_count[i]) -
                                          std::log(1.0 + L_trans * (j - 1) / match_count[i]);
            }
        }
    }
    else
    {
        // Classic consensus maximization
        kernel_buffer.setOnes();
    }

    return kernel_buffer;
}

double TransFGO::calcScore(const std::vector<int> &inlier_ids,
                           const Eigen::MatrixXd &kernel_buffer)
{
    double score = 0.0;
    std::vector<int> unique_ids = inlier_ids;
    std::sort(unique_ids.begin(), unique_ids.end());
    unique_ids.erase(std::unique(unique_ids.begin(), unique_ids.end()), unique_ids.end());

    for (int id : unique_ids)
    {
        int count = std::count(inlier_ids.begin(), inlier_ids.end(), id);
        for (int j = 0; j < count; j++)
        {
            if (id < kernel_buffer.rows() && j < kernel_buffer.cols())
            {
                score += kernel_buffer(id, j);
            }
        }
    }

    return score;
}

double TransFGO::calcUB(const std::vector<Eigen::Vector3d> &pert_rot_n,
                        const std::vector<Eigen::Vector3d> &p_3D,
                        const std::vector<int> &ids,
                        const TBranch &branch,
                        const Eigen::MatrixXd &kernel_buffer)
{
    // Constrain branch within scene limits
    double y_max = std::min(branch.y_max, space_size_(1));
    double z_max = std::min(branch.z_max, space_size_(2));

    if (y_max <= branch.y_min || z_max <= branch.z_min)
    {
        return -1.0;
    }

    // Define vertices of the YZ rectangle
    std::vector<Eigen::Vector2d> vertices = {
        {branch.y_min, branch.z_min}, {branch.y_min, z_max}, {y_max, branch.z_min}, {y_max, z_max}};

    // Prepare intervals for stabbing
    std::vector<double> intervals_upper;
    std::vector<int> ids_upper;

    for (size_t i = 0; i < pert_rot_n.size(); i++)
    {
        std::vector<double> interval = transUpperInterval(pert_rot_n[i], p_3D[i],
                                                          epsilon_t_, space_size_(0), vertices);
        if (!interval.empty())
        {
            intervals_upper.insert(intervals_upper.end(), interval.begin(), interval.end());
            for (size_t j = 0; j < interval.size() / 2; j++)
            {
                ids_upper.emplace_back(ids[i]);
            }
        }
    }

    if (ids_upper.empty())
    {
        return -1.0;
    }

    auto [upper_bound, stabbers] = SatIS::saturatedIntervalStabbing(intervals_upper, ids_upper, kernel_buffer, prox_threshold_);
    return upper_bound;
}

std::pair<double, std::vector<double>>
TransFGO::calcLB(const std::vector<Eigen::Vector3d> &pert_rot_n,
                 const std::vector<Eigen::Vector3d> &p_3D,
                 const std::vector<int> &ids,
                 const TBranch &branch,
                 const Eigen::MatrixXd &kernel_buffer)
{
    // Constrain branch within scene limits
    double y_max = std::min(branch.y_max, space_size_(1));
    double z_max = std::min(branch.z_max, space_size_(2));

    if (y_max <= branch.y_min || z_max <= branch.z_min)
    {
        return {-1.0, {}};
    }

    // Sample center point
    Eigen::Vector2d yz_sampled((branch.y_min + y_max) / 2.0, (branch.z_min + z_max) / 2.0);

    // Prepare intervals for stabbing
    std::vector<double> intervals_lower;
    std::vector<int> ids_lower;

    for (size_t i = 0; i < pert_rot_n.size(); i++)
    {
        std::vector<double> interval = transLowerInterval(pert_rot_n[i], p_3D[i],
                                                          epsilon_t_, yz_sampled, space_size_(0));
        if (!interval.empty())
        {
            intervals_lower.insert(intervals_lower.end(), interval.begin(), interval.end());
            for (size_t j = 0; j < interval.size() / 2; j++)
            {
                ids_lower.emplace_back(ids[i]);
            }
        }
    }

    if (ids_lower.empty())
    {
        return {-1.0, {}};
    }

    return SatIS::saturatedIntervalStabbing(intervals_lower, ids_lower, kernel_buffer, prox_threshold_);
}

std::vector<double> TransFGO::transUpperInterval(const Eigen::Vector3d &n_2D_rot,
                                                 const Eigen::Vector3d &p_3D,
                                                 double epsilon_t,
                                                 double x_limit,
                                                 const std::vector<Eigen::Vector2d> &vertices)
{
    // Regularize n_x >= 0
    Eigen::Vector3d n_normalized = n_2D_rot;
    double n_x = n_normalized(0);
    if (n_x < 0)
    {
        n_normalized = -n_normalized;
        n_x = -n_x;
    }

    // Solve linear programming by comparing values at all vertices
    Eigen::Vector2d n_yz = n_normalized.segment<2>(1);
    double max_v = -std::numeric_limits<double>::infinity();
    double min_v = std::numeric_limits<double>::infinity();

    for (const auto &vertex : vertices)
    {
        double value = -n_yz.dot(vertex);
        max_v = std::max(value, max_v);
        min_v = std::min(value, min_v);
    }

    double const_term = n_normalized.dot(p_3D);
    double const_max = const_term + max_v + epsilon_t;
    double const_min = const_term + min_v - epsilon_t;

    // n_x*x - const_max <= 0 & n_x*x - const_min >= 0
    std::vector<double> interval;
    if (n_x == 0)
    {
        if (const_max >= 0 && const_min <= 0)
        {
            interval = {0.0, x_limit};
        }
    }
    else
    {
        double u = std::min(const_max / n_x, x_limit);
        double l = std::max(const_min / n_x, 0.0);
        if (u >= 0 && l <= x_limit && l <= u)
        {
            interval = {l, u};
        }
    }

    return interval;
}

std::vector<double> TransFGO::transLowerInterval(const Eigen::Vector3d &n_2D_rot,
                                                 const Eigen::Vector3d &p_3D,
                                                 double epsilon_t,
                                                 const Eigen::Vector2d &yz_sampled,
                                                 double x_limit)
{
    // Regularize n_x >= 0
    Eigen::Vector3d n_normalized = n_2D_rot;
    double n_x = n_normalized(0);
    if (n_x < 0)
    {
        n_normalized = -n_normalized;
        n_x = -n_x;
    }

    // Calculate constant term with yz fixed at yz_sampled
    double const_term = n_normalized.dot(p_3D) - n_normalized.segment<2>(1).dot(yz_sampled);

    std::vector<double> interval;
    if (n_x == 0)
    {
        if (std::abs(const_term) <= epsilon_t)
        {
            interval = {0.0, x_limit};
        }
    }
    else
    {
        double lower = (const_term - epsilon_t) / n_x;
        double upper = (const_term + epsilon_t) / n_x;

        if (upper >= 0 && lower <= x_limit)
        {
            lower = std::max(0.0, lower);
            upper = std::min(x_limit, upper);
            if (lower <= upper)
            {
                interval = {lower, upper};
            }
        }
    }

    return interval;
}

bool TransFGO::checkLineRect(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2,
                             int width, int height)
{
    auto isIn = [width, height](const Eigen::Vector2d &p)
    {
        return p(0) >= 0 && p(0) <= width && p(1) >= 0 && p(1) <= height;
    };

    if (isIn(p1) || isIn(p2))
    {
        return true;
    }

    // Check intersection with rectangle edges
    std::vector<std::array<Eigen::Vector2d, 2>> edges = {
        {Eigen::Vector2d(0, 0), Eigen::Vector2d(0, height)},         // left
        {Eigen::Vector2d(width, 0), Eigen::Vector2d(width, height)}, // right
        {Eigen::Vector2d(0, 0), Eigen::Vector2d(width, 0)},          // bottom
        {Eigen::Vector2d(0, height), Eigen::Vector2d(width, height)} // top
    };

    auto ccw = [](const Eigen::Vector2d &a, const Eigen::Vector2d &b, const Eigen::Vector2d &c)
    {
        return (b(0) - a(0)) * (c(1) - a(1)) - (b(1) - a(1)) * (c(0) - a(0));
    };

    auto onSeg = [](const Eigen::Vector2d &a, const Eigen::Vector2d &b, const Eigen::Vector2d &c)
    {
        return std::min(a(0), b(0)) <= c(0) && c(0) <= std::max(a(0), b(0)) &&
               std::min(a(1), b(1)) <= c(1) && c(1) <= std::max(a(1), b(1));
    };

    for (const auto &edge : edges)
    {
        const Eigen::Vector2d &a = edge[0];
        const Eigen::Vector2d &b = edge[1];

        double d1 = ccw(p1, p2, a);
        double d2 = ccw(p1, p2, b);
        double d3 = ccw(a, b, p1);
        double d4 = ccw(a, b, p2);

        if (d1 * d2 < 0 && d3 * d4 < 0)
        {
            return true;
        }
        if ((d1 == 0 && onSeg(p1, p2, a)) || (d2 == 0 && onSeg(p1, p2, b)) ||
            (d3 == 0 && onSeg(a, b, p1)) || (d4 == 0 && onSeg(a, b, p2)))
        {
            return true;
        }
    }

    return false;
}

std::vector<int> TransFGO::pruneInliers(const Eigen::Matrix3d &R_opt,
                                        const Eigen::Matrix3d &intrinsic,
                                        const std::vector<int> &inliers,
                                        const std::vector<Eigen::Vector3d> &endpoints_3D,
                                        const Eigen::Vector3d &t_candidate)
{
    std::vector<int> real_inliers;

    for (int idx : inliers)
    {
        // Get 3D line endpoints
        Eigen::Vector3d endpoint1 = endpoints_3D[idx * 2];
        Eigen::Vector3d endpoint2 = endpoints_3D[idx * 2 + 1];

        // Transform to camera coordinates
        Eigen::Vector3d cam_point1 = R_opt.transpose() * (endpoint1 - t_candidate);
        Eigen::Vector3d cam_point2 = R_opt.transpose() * (endpoint2 - t_candidate);

        // Check if line is behind camera
        if (cam_point1(2) < 0 && cam_point2(2) < 0)
        {
            continue;
        }

        // Project to image coordinates
        Eigen::Vector3d pixel1_h = intrinsic * cam_point1;
        Eigen::Vector3d pixel2_h = intrinsic * cam_point2;

        if (pixel1_h(2) != 0 && pixel2_h(2) != 0)
        {
            Eigen::Vector2d pixel1 = pixel1_h.head<2>() / pixel1_h(2);
            Eigen::Vector2d pixel2 = pixel2_h.head<2>() / pixel2_h(2);

            // Check if line intersects with image rectangle (assuming 1920x1440)
            if (checkLineRect(pixel1, pixel2, 1920, 1440))
            {
                real_inliers.emplace_back(idx);
            }
        }
    }

    return real_inliers;
}

Eigen::Vector3d TransFGO::fineTuneTranslation(const std::vector<Eigen::Vector3d> &t_candidates,
                                              const std::vector<Eigen::Vector3d> &pert_rot_n_2D,
                                              const std::vector<Eigen::Vector3d> &p_3D)
{
    double best_residual_norm = std::numeric_limits<double>::infinity();
    Eigen::Vector3d t_best = Eigen::Vector3d::Zero();

    for (const auto &t_candidate : t_candidates)
    {
        // Find inliers under current candidate
        std::vector<int> inliers;
        for (size_t i = 0; i < pert_rot_n_2D.size(); i++)
        {
            double residual = pert_rot_n_2D[i].dot(p_3D[i] - t_candidate);
            if (std::abs(residual) < epsilon_t_)
            {
                inliers.emplace_back(i);
            }
        }

        if (inliers.size() < 3)
        {
            continue; // Need at least 3 points for reliable estimation
        }

        // Solve least squares problem
        Eigen::MatrixXd A(inliers.size(), 3);
        Eigen::VectorXd b(inliers.size());

        for (size_t j = 0; j < inliers.size(); j++)
        {
            int idx = inliers[j];
            A.row(j) = pert_rot_n_2D[idx].transpose();
            b(j) = pert_rot_n_2D[idx].dot(p_3D[idx]);
        }

        Eigen::Vector3d t_fine_tuned = (A.transpose() * A).ldlt().solve(A.transpose() * b);
        double residual_norm = (A * t_fine_tuned - b).norm();

        if (residual_norm < best_residual_norm)
        {
            best_residual_norm = residual_norm;
            t_best = t_fine_tuned;
        }
    }

    return t_best;
}

std::pair<double, Eigen::Vector3d> TransFGO::pruneTCandidates(const Eigen::Matrix3d &R_opt,
                                                              const Eigen::Matrix3d &intrinsic,
                                                              const std::vector<Eigen::Vector3d> &pert_rot_n_2D,
                                                              const std::vector<Eigen::Vector3d> &endpoints_3D,
                                                              const std::vector<int> &ids,
                                                              const std::vector<Eigen::Vector3d> &t_candidates,
                                                              const Eigen::MatrixXd &kernel_buffer)
{
    double best_score = -1.0;
    Eigen::Vector3d t_best = Eigen::Vector3d::Zero();

    // Extract points from endpoints (take every first endpoint)
    std::vector<Eigen::Vector3d> p_3D;
    for (size_t i = 0; i < endpoints_3D.size(); i += 2)
    {
        p_3D.emplace_back(endpoints_3D[i]);
    }

    for (const auto &t_test : t_candidates)
    {
        // Find geometric inliers
        std::vector<int> inliers;
        for (size_t i = 0; i < pert_rot_n_2D.size(); i++)
        {
            double residual = pert_rot_n_2D[i].dot(p_3D[i] - t_test);
            if (std::abs(residual) <= epsilon_t_)
            {
                inliers.emplace_back(i);
            }
        }

        // Filter out lines behind camera and outside image
        std::vector<int> real_inliers;
        for (int idx : inliers)
        {
            // Transform endpoints to camera coordinates
            Eigen::Vector3d endpoint1 = R_opt.transpose() * (endpoints_3D[idx * 2] - t_test);
            Eigen::Vector3d endpoint2 = R_opt.transpose() * (endpoints_3D[idx * 2 + 1] - t_test);

            // Check if line is behind camera
            if (endpoint1(2) < 0 && endpoint2(2) < 0)
            {
                continue;
            }

            // Project to image coordinates
            if (endpoint1(2) > 0 && endpoint2(2) > 0)
            {
                Eigen::Vector3d pixel1_h = intrinsic * endpoint1;
                Eigen::Vector3d pixel2_h = intrinsic * endpoint2;
                Eigen::Vector2d pixel1 = pixel1_h.head<2>() / pixel1_h(2);
                Eigen::Vector2d pixel2 = pixel2_h.head<2>() / pixel2_h(2);

                // Check if line intersects image bounds (assuming 1920x1440)
                bool intersects = checkLineRect(pixel1, pixel2, 1920, 1440);

                if (intersects)
                {
                    real_inliers.emplace_back(idx);
                }
            }
        }

        // Calculate score using internal kernel buffer
        std::vector<int> real_inlier_ids;
        for (int idx : real_inliers)
        {
            real_inlier_ids.emplace_back(ids[idx]);
        }

        double score = calcScore(real_inlier_ids, kernel_buffer);

        if (score > best_score)
        {
            best_score = score;
            t_best = t_test;
        }
    }

    return {best_score, t_best};
}

void TransFGO::preprocessRotation(const std::vector<int> &ids,
                                  const Eigen::Matrix3d &R_opt,
                                  const std::vector<Eigen::Vector3d> &v_3D,
                                  const std::vector<Eigen::Vector3d> &n_2D,
                                  const std::vector<Eigen::Vector3d> &endpoints_3D,
                                  double epsilon_r,
                                  std::vector<Eigen::Vector3d> &pert_rot_n_2D_inlier,
                                  std::vector<Eigen::Vector3d> &endpoints_3D_inlier,
                                  std::vector<int> &id_inliers_under_rot)
{
    // Clear output vectors
    id_inliers_under_rot.clear();
    pert_rot_n_2D_inlier.clear();
    endpoints_3D_inlier.clear();

    // Find inliers under rotation and count them for efficient memory allocation
    std::vector<int> inlier_indices;
    inlier_indices.reserve(v_3D.size()); // Reserve maximum possible space

    for (size_t i = 0; i < v_3D.size(); i++)
    {
        Eigen::Vector3d rotated_v = R_opt.transpose() * v_3D[i];
        double dot_product = rotated_v.dot(n_2D[i]);
        if (std::abs(dot_product) <= epsilon_r)
        {
            inlier_indices.emplace_back(static_cast<int>(i));
        }
    }

    // Reserve space for all output vectors based on inlier count
    const size_t num_inliers = inlier_indices.size();
    id_inliers_under_rot.reserve(num_inliers);
    pert_rot_n_2D_inlier.reserve(num_inliers);
    endpoints_3D_inlier.reserve(num_inliers * 2); // 2 endpoints per line

    // Temporary vectors for intermediate data with reserved space
    std::vector<Eigen::Vector3d> n_2D_inlier, v_3D_inlier;
    n_2D_inlier.reserve(num_inliers);
    v_3D_inlier.reserve(num_inliers);

    // Extract inlier data using emplace_back for efficiency
    for (int idx : inlier_indices)
    {
        id_inliers_under_rot.emplace_back(ids[idx]);
        n_2D_inlier.emplace_back(n_2D[idx]);
        v_3D_inlier.emplace_back(v_3D[idx]);

        // Add corresponding endpoints
        endpoints_3D_inlier.emplace_back(endpoints_3D[idx * 2]);
        endpoints_3D_inlier.emplace_back(endpoints_3D[idx * 2 + 1]);
    }

    // Perturb normal vectors to be orthogonal to direction vectors after rotation
    pert_rot_n_2D_inlier.resize(num_inliers);
    for (size_t i = 0; i < num_inliers; i++)
    {
        Eigen::Vector3d rotated_n = R_opt * n_2D_inlier[i];
        const Eigen::Vector3d &v = v_3D_inlier[i];

        // Project out the component along v to make n orthogonal to v
        Eigen::Vector3d n_pert = rotated_n - (rotated_n.dot(v)) * v;
        n_pert.normalize();
        pert_rot_n_2D_inlier[i] = std::move(n_pert);
    }

    std::cout << "Rotation preprocessing: " << num_inliers
              << " inliers from " << v_3D.size() << " associations" << std::endl;
}
