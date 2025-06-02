"""
Line Extractor Part 2

This script merges redundant 3D lines based on parallel and proximity conditions.
Parameters can be tuned in helper.py.

Output:
- RGB images annotated with extracted lines and their semantic labels
- Mesh file (.ply) with all merged 3D lines
- One 3D line Mesh file (.ply) for each semantic label
- A numpy file containing all extracted 2D lines and regressed 3D lines

Author: Haodong JIANG <221049033@link.cuhk.edu.cn>
Version: 1.0
License: MIT
"""
import os
import numpy as np
import open3d as o3d
import argparse
from joblib import Parallel, delayed
from scipy import stats
from tqdm import tqdm
import helper

class LineStates:
    """Holds all input data."""
    def __init__(self, scene_id):
        self.scene_id = scene_id
        self.root_dir = "/data1/home/lucky/IROS25/"
        self.scene_data_path = os.path.join(
            self.root_dir, f"SCORE/line_map_extractor/out/{scene_id}/{scene_id}_results_raw.npy"
        )
        self.line_data_folder = os.path.join(
            self.root_dir, f"SCORE/line_map_extractor/out/{scene_id}/"
        )
        self.line_mesh_folder = os.path.join(
            self.root_dir, f"SCORE/line_map_extractor/out/{scene_id}/line_mesh_merged/"
        )
        self.ensure_dirs()
        self.load_scene_data()
        # Merged data containers
        self.merged_semantic_id_3D = []
        self.merged_scene_line_3D_end_points = []
        self.scene_projection_error_r = {}
        self.scene_projection_error_t = {}
        self.scene_line_2D_match_idx_updated = {}

    def ensure_dirs(self):
        for out_path in [self.line_data_folder, self.line_mesh_folder]:
            if not os.path.exists(out_path):
                os.makedirs(out_path)

    def load_scene_data(self):
        scene_data = np.load(self.scene_data_path, allow_pickle=True).item()
        self.scene_pose = scene_data["scene_pose"]
        self.scene_intrinsic = scene_data["scene_intrinsic"]
        self.id_label_dict = scene_data["id_label_dict"]
        self.scene_line_2D_end_points = scene_data["scene_line_2D_end_points"]
        self.scene_line_2D_semantic_ids = scene_data["scene_line_2D_semantic_ids"]
        self.scene_line_2D_params = scene_data["scene_line_2D_params"]
        self.scene_line_2D_match_idx = scene_data["scene_line_2D_match_idx"]
        self.scene_line_3D_end_points = scene_data["scene_line_3D_end_points"]
        self.scene_line_3D_image_source = scene_data["scene_line_3D_image_source"]
        self.scene_line_3D_semantic_ids = scene_data["scene_line_3D_semantic_ids"]

def construct_graph(state):
    """
    Constructs a graph based on the 3D lines.
    Each 3D line is a vertex; edges are defined by parallel and proximity conditions.
    """
    nnode = len(state.scene_line_3D_semantic_ids)
    pi_list = np.array([(state.scene_line_3D_end_points[i][0] + state.scene_line_3D_end_points[i][1]).reshape(1, 3) / 2 for i in range(nnode)])
    p_diff_list = np.array([(state.scene_line_3D_end_points[i][1] - state.scene_line_3D_end_points[i][0]).reshape(1, 3) for i in range(nnode)])
    vi_list = np.array([p_diff_list[i] / np.linalg.norm(p_diff_list[i]) for i in range(nnode)])
    project_null_list = np.eye(3) - np.einsum('ijk,ijl->ikl', vi_list, vi_list)
    scene_line_3D_image_source = state.scene_line_3D_image_source
    print("Constructing the consistency graph")

    def find_neighbors(i):
        edges_i, edges_j = [], []
        if i % 1000 == 0:
            print("Finding neighbors in progress:", i / nnode * 100, "%")
        cur_image_indices = [scene_line_3D_image_source[i]]
        for j in range(i + 1, nnode):
            if scene_line_3D_image_source[j] not in cur_image_indices:
                if abs(np.dot(vi_list[i], vi_list[j].T)) >= helper.params_3D["parrallel_thresh_3D"]:
                    if np.linalg.norm(np.dot(project_null_list[i], (pi_list[i] - pi_list[j]).T)) <= helper.params_3D["overlap_thresh_3D"]:
                        edges_i.append(i)
                        edges_j.append(j)
                        cur_image_indices.append(scene_line_3D_image_source[j])
        return edges_i, edges_j

    results = Parallel(n_jobs=helper.params_3D["thread_number"])(delayed(find_neighbors)(i) for i in range(nnode))
    edges_i, edges_j = [], []
    for edges_i_, edges_j_ in results:
        edges_i.extend(edges_i_)
        edges_j.extend(edges_j_)
    np.save(
        os.path.join(state.line_data_folder, state.scene_id + "_edges.npy"),
        {"edges_i": edges_i, "edges_j": edges_j}
    )

def merge_lines(state):
    """
    Merges the 3D lines based on the constructed graph.
    Iteratively finds the vertex with largest degree and merges all its neighbors.
    """
    nnode = len(state.scene_line_3D_semantic_ids)
    edge_data = np.load(os.path.join(state.line_data_folder, state.scene_id + "_edges.npy"), allow_pickle=True).item()
    edges_i = np.array(edge_data["edges_i"])
    edges_j = np.array(edge_data["edges_j"])
    print("# 3D lines before merging", nnode)
    mapping = list(range(nnode))
    edges_i_ = np.concatenate((edges_i, np.arange(nnode)))
    edges_j_ = np.concatenate((edges_j, np.arange(nnode)))
    vertex_concat = np.concatenate((edges_i_, edges_j_))

    # Step 0: Remove suspicious lines observed by too few images
    unique_elements, counts = np.unique(vertex_concat, return_counts=True)
    vertex_deleted = unique_elements[counts < helper.params_3D["degree_threshold"] + 2]
    for ver in vertex_deleted:
        mapping[ver] = np.nan
    index_deleted = []
    for i in range(len(edges_i_)):
        if edges_i_[i] in vertex_deleted or edges_j_[i] in vertex_deleted:
            index_deleted.append(i)
    edges_i_ = np.delete(edges_i_, index_deleted)
    edges_j_ = np.delete(edges_j_, index_deleted)

    # Step 1: Iteratively merge
    countt = 0
    while len(edges_i_) > 0:
        vertex_concat = np.concatenate((edges_i_, edges_j_))
        mode_result = stats.mode(vertex_concat)
        most_frequent_index = mode_result.mode
        index_1 = np.where(edges_i_ == most_frequent_index)
        index_2 = np.where(edges_j_ == most_frequent_index)
        neighbors = np.unique(np.concatenate((edges_j_[index_1], edges_i_[index_2])))
        # Remove neighbor nodes and edges
        for neighbor in neighbors:
            index_1 = np.where(edges_i_ == neighbor)
            index_2 = np.where(edges_j_ == neighbor)
            index_delete_neighbor = np.unique(np.concatenate((index_1[0], index_2[0])))
            edges_i_ = np.delete(edges_i_, index_delete_neighbor)
            edges_j_ = np.delete(edges_j_, index_delete_neighbor)
        # Update endpoints
        end_points = state.scene_line_3D_end_points[most_frequent_index]
        v = end_points[1] - end_points[0]
        v = v / np.linalg.norm(v)
        sig_dim = np.argmax(np.abs(v))
        for neighbor in neighbors:
            end_points_temp = state.scene_line_3D_end_points[neighbor]
            if end_points_temp[0][sig_dim] < end_points[0][sig_dim]:
                end_points[0] = end_points[0] + (end_points_temp[0][sig_dim] - end_points[0][sig_dim]) * (v / v[sig_dim])
            if end_points_temp[1][sig_dim] > end_points[1][sig_dim]:
                end_points[1] = end_points[1] + (end_points_temp[1][sig_dim] - end_points[1][sig_dim]) * (v / v[sig_dim])
        # For each unique semantic label, create a 3D line in the map
        cluster_semantic_ids = []
        for neighbor in neighbors:
            cluster_semantic_ids = np.append(cluster_semantic_ids, state.scene_line_3D_semantic_ids[neighbor])
        unique_cluster_semantic_ids = np.unique(cluster_semantic_ids)
        unique_cluster_semantic_ids = unique_cluster_semantic_ids[unique_cluster_semantic_ids != 0]
        for label in unique_cluster_semantic_ids:
            state.merged_semantic_id_3D.append(label)
            state.merged_scene_line_3D_end_points.append(end_points)
            for neighbor in neighbors:
                if label == state.scene_line_3D_semantic_ids[neighbor]:
                    mapping[neighbor] = len(state.merged_semantic_id_3D) - 1
        # Debug: output the 3D line with more than 3 semantic labels
        if len(unique_cluster_semantic_ids) > 3:
            point_diff = end_points[1] - end_points[0]
            point_sets = [end_points[0] + point_diff * sample / 299 for sample in range(300)]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_sets)
            o3d.io.write_point_cloud(
                os.path.join(state.line_mesh_folder, f"multiple_semantic_{countt}.ply"), pcd
            )
            countt += 1
            for k in range(len(unique_cluster_semantic_ids)):
                print(f"{state.id_label_dict[unique_cluster_semantic_ids[k]]},", end="")
    print("# 3D lines after merging:", len(state.merged_scene_line_3D_end_points))
    return mapping

def update_err(state, mapping):
    """
    Updates the projection error after merging the 3D lines.
    Computes the projection error for each 2D line based on the merged 3D lines.
    """
    print("Updating projection error after merging")
    for basename in state.scene_line_2D_match_idx.keys():
        projection_error_r = []
        projection_error_t = []
        intrinsic = state.scene_intrinsic[basename]
        pose_matrix = np.array(state.scene_pose[basename])
        line_2D_match_idx = np.array(state.scene_line_2D_match_idx[basename])
        line_2D_match_idx_updated = line_2D_match_idx.copy()
        for j in range(len(line_2D_match_idx)):
            if np.isnan(line_2D_match_idx[j]) or np.isnan(mapping[line_2D_match_idx[j]]):
                line_2D_match_idx_updated[j] = -1
                projection_error_r.append(np.nan)
                projection_error_t.append(np.nan)
            else:
                mapping_idx = mapping[line_2D_match_idx[j]]
                line_2D_match_idx_updated[j] = mapping_idx
                n_j = state.scene_line_2D_params[basename][j].reshape(1, 3)
                end_points_3D = state.merged_scene_line_3D_end_points[mapping_idx]
                v = end_points_3D[1] - end_points_3D[0]
                v = v / np.linalg.norm(v)
                error_rot, error_trans = helper.calculate_error(
                    n_j, v, intrinsic, pose_matrix, end_points_3D[0], end_points_3D[1]
                )
                projection_error_r.append(np.abs(error_rot))
                projection_error_t.append(np.abs(error_trans))
                if np.abs(error_trans) > 0.1:
                    print(f"Warning: projection error too large for {basename} at index {j}, error_t={error_trans}")
        state.scene_line_2D_match_idx_updated[basename] = line_2D_match_idx_updated
        state.scene_projection_error_r[basename] = projection_error_r
        state.scene_projection_error_t[basename] = projection_error_t

def save_merged_line(state, sample_num):
    """
    Saves the merged 3D lines and their semantic labels.
    Also saves the 3D line mesh for visualization.
    """
    point_sets = []
    for i in range(len(state.merged_semantic_id_3D)):
        end_points = state.merged_scene_line_3D_end_points[i]
        point_diff = end_points[1] - end_points[0]
        for sample in range(sample_num):
            point_sets.append(end_points[0] + point_diff * sample / (sample_num - 1))
    point_sets = np.vstack(point_sets)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_sets)
    o3d.io.write_point_cloud(
        os.path.join(state.line_mesh_folder, state.scene_id + f"_merged_3D_line_mesh.ply"), pcd
    )
    # Save the 3D line mesh for each semantic label
    semantic_ids_all = np.unique(state.merged_semantic_id_3D)
    for i, semantic_id in enumerate(semantic_ids_all):
        if int(semantic_id) == 0:
            continue
        index = np.where(state.merged_semantic_id_3D == semantic_id)
        print("semantic label:" + f"{state.id_label_dict[int(semantic_id)]}" + " number of lines:", len(index[0]))
        point_sets = []
        for j in range(len(index[0])):
            end_points = state.merged_scene_line_3D_end_points[i]
            point_diff = end_points[1] - end_points[0]
            for sample in range(sample_num):
                point_sets.append(end_points[0] + point_diff * sample / (sample_num - 1))
        if point_sets:
            point_sets = np.vstack(point_sets)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_sets)
            o3d.io.write_point_cloud(
                os.path.join(state.line_mesh_folder, f"{state.id_label_dict[int(semantic_id)]}.ply"), pcd
            )

def save_results(state):
    """
    Saves all merged and processed data to a numpy file.
    """
    np.save(
        os.path.join(state.line_data_folder, state.scene_id + "_results_merged.npy"),
        {
            "scene_pose": state.scene_pose,
            "scene_intrinsic": state.scene_intrinsic,
            "id_label_dict": state.id_label_dict,
            "scene_line_2D_semantic_ids": state.scene_line_2D_semantic_ids,
            "scene_line_2D_params": state.scene_line_2D_params,
            "scene_line_2D_end_points": state.scene_line_2D_end_points,
            "scene_line_2D_match_idx_updated": state.scene_line_2D_match_idx_updated,
            "scene_projection_error_r": state.scene_projection_error_r,
            "scene_projection_error_t": state.scene_projection_error_t,
            "merged_scene_line_3D_semantic_ids": state.merged_semantic_id_3D,
            "merged_scene_line_3D_end_points": state.merged_scene_line_3D_end_points,
            "params_3D": helper.params_3D,
        }
    )

def run(scene_id, reuse_graph_flag):
    state = LineStates(scene_id)
    if reuse_graph_flag:
        print("Use previously constructed graph")
    else:
        construct_graph(state)
    mapping = merge_lines(state)
    update_err(state, mapping)
    save_results(state)
    sample_num = 300
    save_merged_line(state, sample_num)
    print("line extraction pt2 finished")

if __name__ == "__main__":
    scene_list = ["69e5939669", "689fec23D7", "c173f62b15", "55b2bf8036"]
    scene_id = scene_list[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse', '-r', default='n', choices=['y', 'n'], help='use constructed graph, y or n')
    args = parser.parse_args()
    reuse_graph_flag = args.reuse == "y"
    run(scene_id, reuse_graph_flag)
    