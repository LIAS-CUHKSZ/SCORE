/*
Saturated Interval Stabbing algorithm

Author:  Qingcheng Zeng <qzeng450@connect.hkust-gz.edu.cn>
         Haodong Jiang  <221049033@link.cuhk.edu.cn>
License: MIT
*/

#ifndef SATIS_H
#define SATIS_H

#include <vector>
#include <algorithm>
#include <Eigen/Dense>

namespace SatIS
{

    /**
     * @brief Saturated Interval Stabbing algorithm
     *
     * Finds the stabbing point(s) that maximize the score according to the kernel buffer.
     * This implements the saturated consensus maximization approach for interval stabbing.
     *
     * @param intervals Vector of interval endpoints (size = 2*L, where L is number of intervals)
     *                  Format: [start1, end1, start2, end2, ...]
     * @param ids Vector of interval IDs (size = L), indicating which 2D line each interval belongs to
     * @param kernel_buffer Scoring matrix where kernel_buffer(id, count-1) gives the weight
     *                      for the count-th match of line id
     * @param prox_threshold Proximity threshold for sampling stabbers within intervals
     * @return Pair of (best_score, stabber_candidates)
     */
    inline std::pair<double, std::vector<double>> saturatedIntervalStabbing(
        const std::vector<double> &intervals,
        const std::vector<int> &ids,
        const Eigen::MatrixXd &kernel_buffer,
        double prox_threshold)
    {
        size_t L = ids.size();
        std::vector<std::pair<double, std::pair<int, int>>>
            events; // (value, (mask, id_index))

        for (size_t i = 0; i < L; i++)
        {
            events.emplace_back(intervals[2 * i], std::make_pair(0, i));     // interval start
            events.emplace_back(intervals[2 * i + 1], std::make_pair(1, i)); // interval end
        }

        // Sort events by value
        std::sort(events.begin(), events.end());

        int max_id = *std::max_element(ids.begin(), ids.end());
        std::vector<int> count_buffer(max_id + 1, 0);

        double score = 0.0;
        double best_score = 0.0;
        std::vector<double> stabbers;

        for (size_t i = 0; i < events.size() - 1; i++)
        {
            auto [value, event_data] = events[i];
            auto [mask, id_index] = event_data;

            if (mask == 0)
            {
                // Entering an interval
                int id = ids[id_index];
                count_buffer[id]++;
                if (id < kernel_buffer.rows() && count_buffer[id] <= kernel_buffer.cols())
                {
                    score += kernel_buffer(id, count_buffer[id] - 1);
                }
            }
            else
            {
                // Exiting an interval
                int id = ids[id_index];
                if (id < kernel_buffer.rows() && count_buffer[id] <= kernel_buffer.cols())
                {
                    score -= kernel_buffer(id, count_buffer[id] - 1);
                }
                count_buffer[id]--;
            }

            // Update best stabbers
            if (score >= best_score)
            {
                double start = events[i].first;
                double end = events[i + 1].first;

                if (score > best_score)
                {
                    stabbers.clear();
                    best_score = score;
                }

                // Sample stabbers in the interval
                for (double s = start; s <= end; s += prox_threshold)
                {
                    stabbers.push_back(s);
                }
                if (stabbers.empty() || stabbers.back() != end)
                {
                    stabbers.push_back(end);
                }
            }
        }

        return {best_score, stabbers};
    }

    /**
     * @brief Cluster similar stabbers using proximity threshold
     *
     * Groups nearby stabbers into clusters and returns the median of each cluster.
     * This reduces the number of candidate solutions while maintaining coverage.
     *
     * @param stabbers Vector of stabber positions (will be sorted internally)
     * @param prox_threshold Maximum distance between stabbers in the same cluster
     * @return Vector of cluster medians
     */
    inline std::vector<double> clusterStabber(const std::vector<double> &stabbers, double prox_threshold)
    {
        if (stabbers.empty())
        {
            return {};
        }

        if (stabbers.size() == 1)
        {
            return {stabbers[0]};
        }

        std::vector<double> sorted_stabbers = stabbers;
        std::sort(sorted_stabbers.begin(), sorted_stabbers.end());

        std::vector<double> clustered;
        std::vector<double> current_cluster;
        current_cluster.push_back(sorted_stabbers[0]);

        for (size_t i = 1; i < sorted_stabbers.size(); i++)
        {
            if (sorted_stabbers[i] - current_cluster[0] <= prox_threshold)
            {
                current_cluster.push_back(sorted_stabbers[i]);
            }
            else
            {
                // Finish current cluster - compute median
                size_t mid_index = (current_cluster.size() - 1) / 2;
                if (current_cluster.size() % 2 == 0)
                {
                    mid_index = current_cluster.size() / 2 - 1;
                }
                clustered.push_back(current_cluster[mid_index]);

                // Start new cluster
                current_cluster.clear();
                current_cluster.push_back(sorted_stabbers[i]);
            }
        }

        // Finish last cluster
        if (!current_cluster.empty())
        {
            size_t mid_index = (current_cluster.size() - 1) / 2;
            if (current_cluster.size() % 2 == 0)
            {
                mid_index = current_cluster.size() / 2 - 1;
            }
            clustered.push_back(current_cluster[mid_index]);
        }

        return clustered;
    }

} // namespace SatIS

#endif // SATIS_H
