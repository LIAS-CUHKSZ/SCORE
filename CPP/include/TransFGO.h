/*
FGO-PnL Translation Solver 

Author:  Qingcheng Zeng <qzeng450@connect.hkust-gz.edu.cn>
         Haodong Jiang  <221049033@link.cuhk.edu.cn>
License: MIT
*/

#ifndef TRANSFGO_H
#define TRANSFGO_H

#include <Eigen/Dense>
#include <queue>
#include <vector>
#include <cmath>
#include "helper.h"

struct TBranch
{
    double y_min, z_min, y_max, z_max;
    double upper_bound, lower_bound;

    TBranch()
        : y_min(0), z_min(0), y_max(0), z_max(0), upper_bound(-1), lower_bound(-1) {}

    TBranch(double y_min_, double z_min_, double y_max_, double z_max_)
        : y_min(y_min_), z_min(z_min_), y_max(y_max_), z_max(z_max_),
          upper_bound(-1), lower_bound(-1) {}

    TBranch(double y_min_, double z_min_, double y_max_, double z_max_,
            double u_bound, double l_bound)
        : y_min(y_min_), z_min(z_min_), y_max(y_max_), z_max(z_max_),
          upper_bound(u_bound), lower_bound(l_bound) {}

    // Get the size of the branch (for tie-breaking)
    constexpr double size() const noexcept
    {
        return (y_max - y_min) * (z_max - z_min);
    }

    std::vector<TBranch> subDivide() const
    {
        // Compute midpoints for y and z
        double y_mid = 0.5 * (y_min + y_max);
        double z_mid = 0.5 * (z_min + z_max);
        return {
            {y_min, z_min, y_mid, z_mid}, // lower-left
            {y_mid, z_min, y_max, z_mid}, // lower-right
            {y_min, z_mid, y_mid, z_max}, // upper-left
            {y_mid, z_mid, y_max, z_max}  // upper-right
        };
    }

    // by upper_bound (descending), then by size (descending)
    struct Comparator
    {
        inline bool operator()(const TBranch &a, const TBranch &b) const noexcept
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

class TransFGO
{
public:
    using BranchQueue = std::priority_queue<TBranch, std::vector<TBranch>, TBranch::Comparator>;

    // configuration structure for TransFGO parameters
    struct TransFGOConfig
    {
        double branch_resolution;
        double epsilon_t;
        double prox_threshold;
        Eigen::Vector3d space_size;

        TransFGOConfig(double branch_res, double eps, double prox_thresh,
                       const Eigen::Vector3d &space_sz)
            : branch_resolution(branch_res), epsilon_t(eps), prox_threshold(prox_thresh),
              space_size(space_sz) {}
    };

    TransFGO(const TransFGOConfig &config)
        : branch_resolution_(config.branch_resolution), epsilon_t_(config.epsilon_t),
          prox_threshold_(config.prox_threshold), space_size_(config.space_size),
          pF_buffer(nullptr) {}

    // Legacy constructor for backward compatibility
    TransFGO(double branch_resolution, double epsilon_t, double prox_threshold,
             const Eigen::Vector3d &space_size)
        : branch_resolution_(branch_resolution), epsilon_t_(epsilon_t),
          prox_threshold_(prox_threshold), space_size_(space_size),
          pF_buffer(nullptr) {}

    ~TransFGO() {}

    // Complete translation estimation pipeline given an upstream rotation result
    // including pre-processing, branch-and-bound, and post-pruning
    std::pair<double, Eigen::Vector3d> solve(const std::vector<int> &ids,
                          const Eigen::Matrix3d &R_opt,
                          const std::vector<Eigen::Vector3d> &v_3D,
                          const std::vector<Eigen::Vector3d> &n_2D,
                          const std::vector<Eigen::Vector3d> &endpoints_3D,
                          double epsilon_r,
                          const Eigen::Matrix3d &intrinsic);

    // branch-and-bound
    std::vector<Eigen::Vector3d>
    getCandidates(const std::vector<Eigen::Vector3d> &pert_rot_n_2D,
                  const std::vector<Eigen::Vector3d> &p_3D,
                  const std::vector<int> &ids,
                  const Eigen::MatrixXd &kernel_buffer);
    // update best solution
    void updateBestSolution(const TBranch &branch,
                            const std::vector<double> &x_candidates,
                            double &best_lb,
                            std::vector<Eigen::Vector3d> &best_t_candidates);

    // prune branch queue
    void pruneBranchQueue(BranchQueue &bq, double best_lb);

    // Evaluate and prune translation candidates based on geometric constraints
    std::pair<double, std::vector<Eigen::Vector3d>> pruneTCandidates(const Eigen::Matrix3d &R_opt,
                                                        const Eigen::Matrix3d &intrinsic,
                                                        const std::vector<Eigen::Vector3d> &pert_rot_n_2D,
                                                        const std::vector<Eigen::Vector3d> &points_head_3D,
                                                        const std::vector<Eigen::Vector3d> &points_tail_3D,
                                                        const std::vector<int> &ids,
                                                        const std::vector<Eigen::Vector3d> &t_candidates,
                                                        const Eigen::MatrixXd &kernel_buffer);

    // Fine-tune translation candidates
    Eigen::Vector3d fineTuneTranslation(const std::vector<Eigen::Vector3d> &t_candidates,
                                        const std::vector<Eigen::Vector3d> &pert_rot_n_2D,
                                        const std::vector<Eigen::Vector3d> &p_3D);



    // Preprocess rotation results for translation estimation (optimized)
    void preprocessRotation(const std::vector<int> &ids,
                            const Eigen::Matrix3d &R_opt,
                            const std::vector<Eigen::Vector3d> &v_3D,
                            const std::vector<Eigen::Vector3d> &n_2D,
                            const std::vector<Eigen::Vector3d> &endpoints_3D,
                            double epsilon_r,
                            std::vector<Eigen::Vector3d> &pert_rot_n_2D_inlier,
                            std::vector<Eigen::Vector3d> &points_head_3D_inlier,
                            std::vector<Eigen::Vector3d> &points_tail_3D_inlier,
                            std::vector<int> &id_inliers_under_rot);

    // set saturation function buffer generator
    void setSatBufferFunc(pF_Buffer_Gen pF_buffer_gen)
    {
      pF_buffer = pF_buffer_gen;
    }

private:

    // Upper bound calculation
    double calcUB(const std::vector<Eigen::Vector3d> &pert_rot_n,
                  const std::vector<Eigen::Vector3d> &p_3D,
                  const std::vector<int> &ids,
                  const TBranch &branch,
                  const Eigen::MatrixXd &kernel_buffer);

    // Lower bound calculation
    std::pair<double, std::vector<double>>
    calcLB(const std::vector<Eigen::Vector3d> &pert_rot_n,
           const std::vector<Eigen::Vector3d> &p_3D,
           const std::vector<int> &ids,
           const TBranch &branch,
           const Eigen::MatrixXd &kernel_buffer);

    // Interval calculation for upper bounds
    std::vector<double> transUpperInterval(const Eigen::Vector3d &n_2D_rot,
                                           const Eigen::Vector3d &p_3D,
                                           double epsilon_t,
                                           double x_limit,
                                           const std::vector<Eigen::Vector2d> &vertices);

    // Interval calculation for lower bounds
    std::vector<double> transLowerInterval(const Eigen::Vector3d &n_2D_rot,
                                           const Eigen::Vector3d &p_3D,
                                           double epsilon_t,
                                           const Eigen::Vector2d &yz_sampled,
                                           double x_limit);

    // Check if line intersects with image rectangle
    bool checkLineRect(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2,
                       int width, int height);

    // Prune inliers based on camera constraints
    std::vector<int> pruneInliers(const Eigen::Matrix3d &R_opt,
                                  const Eigen::Matrix3d &intrinsic,
                                  const std::vector<int> &inliers,
                                  const std::vector<Eigen::Vector3d> &points_head_3D,
                                  const std::vector<Eigen::Vector3d> &points_tail_3D,
                                  const Eigen::Vector3d &t_candidate);

    // hyperparameters
    double branch_resolution_;
    double epsilon_t_;
    double prox_threshold_;
    Eigen::Vector3d space_size_;
    double eps = 1e-10;
    // function pointer to saturation function buffer generator
    pF_Buffer_Gen pF_buffer;
};

#endif // TRANSFGO_H
