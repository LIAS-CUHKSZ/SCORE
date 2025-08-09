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
 
License: MIT
"""
import os
import numpy as np
import open3d as o3d
import argparse
import struct
from joblib import Parallel, delayed
from scipy import stats
from tqdm import tqdm
from util import helper
from scipy.spatial.transform import Rotation

class LineStates:
    """Holds all input data."""
    def __init__(self, scene_id):
        self.scene_id = scene_id
        self.root_dir = "/data1/home/lucky/IROS25/"
        self.scene_data_path = os.path.join(
            self.root_dir, f"SCORE/line_map_extractor/out/{scene_id}/map/{scene_id}_results_raw.npy"
        )
        self.line_data_folder = os.path.join(
            self.root_dir, f"SCORE/line_map_extractor/out/{scene_id}/map/"
        )
        self.line_mesh_folder = os.path.join(
            self.root_dir, f"SCORE/line_map_extractor/out/{scene_id}/map/line_mesh/"
        )
        self.sfm_data_folder = os.path.join(
            self.root_dir, f"SCORE/line_map_extractor/out/{scene_id}/map/sfm/"
        )
        self.ensure_dirs()
        self.load_scene_data()
        # Merged data containers
        self.merged_semantic_ids_3D = []
        self.merged_end_points_3D = []
        self.scene_proj_error_r = {}
        self.scene_proj_error_t = {}
        self.scene_line_2D_match_idx = {}

    def ensure_dirs(self):
        for out_path in [self.line_data_folder, self.line_mesh_folder, self.sfm_data_folder]:
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
        self.scene_line_2D_match_idx_raw = scene_data["scene_line_2D_match_idx_raw"]
        self.scene_line_3D_end_points = scene_data["scene_line_3D_end_points"]
        self.scene_line_3D_image_source = scene_data["scene_line_3D_image_source"]
        self.scene_line_3D_semantic_ids = scene_data["scene_line_3D_semantic_ids"]

def construct_graph(state):
    """
    Constructs a graph based on the 3D lines.
    Each 3D line is a vertex; edges are defined by parallel and proximity conditions.
    """
    nnode = len(state.scene_line_3D_semantic_ids)
    p_head_list = np.array([state.scene_line_3D_end_points[i][0] for i in range(nnode)])
    p_tail_list = np.array([state.scene_line_3D_end_points[i][1] for i in range(nnode)])
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
                if abs(np.dot(vi_list[i], vi_list[j].T)) >= helper.params_merge_prune["parrallel_thresh_3D"]:
                    distance_set = np.zeros(4)
                    distance_set[0] = np.linalg.norm(np.dot(project_null_list[j], (p_head_list[i] - p_head_list[j]).T))
                    distance_set[1] = np.linalg.norm(np.dot(project_null_list[j], (p_tail_list[i] - p_tail_list[j]).T))
                    distance_set[2] = np.linalg.norm(np.dot(project_null_list[i], (p_head_list[i] - p_head_list[j]).T))
                    distance_set[3] = np.linalg.norm(np.dot(project_null_list[i], (p_tail_list[i] - p_tail_list[j]).T))
                    if np.max(distance_set) <= helper.params_merge_prune["overlap_thresh_3D"]:
                        edges_i.append(i)
                        edges_j.append(j)
                        cur_image_indices.append(scene_line_3D_image_source[j])
        return edges_i, edges_j

    results = Parallel(n_jobs=helper.thread_number)(delayed(find_neighbors)(i) for i in range(nnode))
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
    vertex_deleted = unique_elements[counts < helper.params_merge_prune["degree_threshold"] + 2]
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
            state.merged_semantic_ids_3D.append(label)
            state.merged_end_points_3D.append(end_points)
            for neighbor in neighbors:
                if label == state.scene_line_3D_semantic_ids[neighbor]:
                    mapping[neighbor] = len(state.merged_semantic_ids_3D) - 1
        # # Debug: output the 3D line with more than 3 semantic labels
        # if len(unique_cluster_semantic_ids) > 3:
        #     point_diff = end_points[1] - end_points[0]
        #     point_sets = [end_points[0] + point_diff * sample / 299 for sample in range(300)]
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(point_sets)
        #     o3d.io.write_point_cloud(
        #         os.path.join(state.line_mesh_folder, f"multiple_semantic_{countt}.ply"), pcd
        #     )
        #     countt += 1
        #     for k in range(len(unique_cluster_semantic_ids)):
        #         print(f"{state.id_label_dict[unique_cluster_semantic_ids[k]]},", end="")
    print("# 3D lines after merging:", len(state.merged_end_points_3D))
    return mapping

def update_err(state, mapping):
    """
    Updates the projection error after merging the 3D lines.
    Computes the projection error for each 2D line based on the merged 3D lines.
    """
    print("Updating projection error after merging")
    for basename in state.scene_line_2D_match_idx_raw.keys():
        proj_error_r = []
        proj_error_t = []
        intrinsic = state.scene_intrinsic[basename]
        pose_matrix = np.array(state.scene_pose[basename])
        line_2D_match_idx_raw = np.array(state.scene_line_2D_match_idx_raw[basename])
        line_2D_end_points = np.array(state.scene_line_2D_end_points[basename])
        line_2D_semantic_ids = np.array(state.scene_line_2D_semantic_ids[basename])
        line_2D_params = np.array(state.scene_line_2D_params[basename])
        line_2D_match_idx = line_2D_match_idx_raw.copy().astype(np.double)
        for j in range(len(line_2D_match_idx_raw)):
            if np.isnan(line_2D_match_idx_raw[j]) or np.isnan(mapping[line_2D_match_idx_raw[j]]):
                pixel_a = line_2D_end_points[j][0]  
                pixel_b = line_2D_end_points[j][1]
                semantic_id = line_2D_semantic_ids[j]
                v_2D = pixel_b - pixel_a
                v_2D = v_2D / np.linalg.norm(v_2D)
                n_j = line_2D_params[j].reshape(1, 3)
                line_2D_match_idx[j] = helper.find_closest_3D_line(v_2D, n_j, semantic_id, intrinsic, pose_matrix, state.merged_end_points_3D, state.merged_semantic_ids_3D) 
            else:
                mapping_idx = mapping[line_2D_match_idx_raw[j]]
                line_2D_match_idx[j] = mapping_idx
            #
            if np.isnan(line_2D_match_idx[j]):
                proj_error_r.append(np.nan)
                proj_error_t.append(np.nan)
            else:
                n_j = state.scene_line_2D_params[basename][j].reshape(1, 3)
                end_points_3D = state.merged_end_points_3D[line_2D_match_idx[j].astype(np.int32)]
                v = end_points_3D[1] - end_points_3D[0]
                v = v / np.linalg.norm(v)
                error_rot, error_trans = helper.calculate_error(
                    n_j, v, intrinsic, pose_matrix, end_points_3D[0], end_points_3D[1]
                )
                if np.abs(error_trans) > 0.2:
                   error_rot = np.nan
                   error_trans = np.nan
                   line_2D_match_idx[j] = np.nan
                proj_error_r.append(np.abs(error_rot))
                proj_error_t.append(np.abs(error_trans))
        # store updated match_idx and projection error in the state           
        state.scene_line_2D_match_idx[basename] = line_2D_match_idx
        state.scene_proj_error_r[basename] = proj_error_r
        state.scene_proj_error_t[basename] = proj_error_t

def save_merged_line(state, sample_num):
    """
    Saves the merged 3D lines and their semantic labels.
    Also saves the 3D line mesh for visualization.
    """
    point_sets = []
    for i in range(len(state.merged_semantic_ids_3D)):
        end_points = state.merged_end_points_3D[i]
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
    semantic_ids_all = np.unique(state.merged_semantic_ids_3D)
    for i, semantic_id in enumerate(semantic_ids_all):
        if int(semantic_id) == 0:
            continue
        index = np.where(state.merged_semantic_ids_3D == semantic_id)
        print("semantic label:" + f"{state.id_label_dict[int(semantic_id)]}" + " number of lines:", len(index[0]))
        point_sets = []
        for j in range(len(index[0])):
            end_points = state.merged_end_points_3D[i]
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
            "scene_line_2D_match_idx": state.scene_line_2D_match_idx,
            "scene_proj_error_r": state.scene_proj_error_r,
            "scene_proj_error_t": state.scene_proj_error_t,
            "merged_semantic_ids_3D": state.merged_semantic_ids_3D,
            "merged_end_points_3D": state.merged_end_points_3D,
            "params_merge_prune": helper.params_merge_prune,
        }
    )

def output_colmap_format(state):
    """
    Outputs the line map in COLMAP format.
    ""camera.bin""  directly copied from the original COLMAP output
    ""lines3D.bin"" line ID, two endpoints, semantic label
    ""images.bin""  Image name, qw, qx, qy, qz, tx, ty, tz, associated 3D line IDs
    """
    # 0. output camera.bin
    # directly copied from the original COLMAP output, omitted here
    # 1. output lines3D.bin
    lines3D_bin_path = os.path.join(state.sfm_data_folder, "lines3D.bin")
    with open(lines3D_bin_path, "wb") as f:
        # total number of lines
        f.write(struct.pack("I", len(state.merged_semantic_ids_3D)))
        for i in range(len(state.merged_semantic_ids_3D)):
            end_points = state.merged_end_points_3D[i]
            line_id = i
            semantic_label = state.merged_semantic_ids_3D[i]
            semantic_label = int(semantic_label)
            f.write(struct.pack("I", line_id))
            f.write(struct.pack("3f", end_points[0][0], end_points[0][1], end_points[0][2]))
            f.write(struct.pack("3f", end_points[1][0], end_points[1][1], end_points[1][2]))
            f.write(struct.pack("I", semantic_label))

    # 2. output images.bin  
    images_bin_path = os.path.join(state.sfm_data_folder, "images.bin")
    with open(images_bin_path, "wb") as f:
        # total number of images
        f.write(struct.pack("I", len(state.scene_line_2D_match_idx.keys())))
        for key in state.scene_line_2D_match_idx.keys():
            # image name
            image_name = key.split(".")[0]
            f.write(struct.pack("s", image_name.encode()))
            # poses
            pose_matrix = np.array(state.scene_pose[key])
            tx, ty, tz = pose_matrix[:3, 3]
            qx, qy, qz, qw = Rotation.from_matrix(pose_matrix[:3, :3]).as_quat()
            f.write(struct.pack("4f", qx, qy, qz, qw))
            f.write(struct.pack("3f", tx, ty, tz))
            # number of associated 3D line IDs
            line_2D_match_idx = np.array(state.scene_line_2D_match_idx[key])
            line_2D_match_idx = line_2D_match_idx[~np.isnan(line_2D_match_idx)]
            num_matched_line = len(line_2D_match_idx)
            f.write(struct.pack("I", num_matched_line))
            # associated 3D line IDs
            for i in range(num_matched_line):
                line_id = line_2D_match_idx[i]
                line_id = int(line_id)
                f.write(struct.pack("I", line_id))

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
    print("3D line clustering and projection error update finished")
    output_colmap_format(state)

if __name__ == "__main__":
    scene_list = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"]
    scene_id = scene_list[0]
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse', '-r', default='n', choices=['y', 'n'], help='use constructed graph, y or n')
    args = parser.parse_args()
    reuse_graph_flag = args.reuse == "y"
    # reuse_graph_flag = True
    run(scene_id, reuse_graph_flag)
    