"""
line_extractor_pt1.py 

This script extracts 2D lines from images and regresses 3D lines from the 2D lines based on pose and depth.
We assign each pair of 2D and 3D line with the same semantic label. 
You can tune the paramters defined in helper.py, and edit the id_remapping.txt file under /dictionary.

Output:
- Rgb images annotated with extracted lines thier semantic labels. 
- Mesh file(.ply) with all regressed 3D lines for visualization.
- A numpy file containing all the extracted 2D lines and regressed 3D lines.

Author: Haodong JIANG <221049033@link.cuhk.edu.cn>
 
License: MIT
"""

from util import helper
import cv2
import numpy as np
import glob
import os
import json
import open3d as o3d
from tqdm import tqdm
from skimage.measure import LineModelND, ransac
from joblib import Parallel, delayed
from scipy import stats
from scipy.spatial.transform import Rotation

def load_config(data_root_dir,output_root_dir,scene_id,scene_name): 
    """Load configuration and paths""" 
    # Setup input directories
    config = {
        'data_root_dir': data_root_dir,
        'output_root_dir': output_root_dir,
        'scene_id': scene_id,
        'ref_list': os.path.join(output_root_dir, f"line_map_extractor/out/{scene_id}/ref.txt"),
        'rgb_folder': os.path.join(data_root_dir, f"data/{scene_id}/iphone/rgb/"),
        'depth_image_folder': os.path.join(data_root_dir, f"data/{scene_id}/iphone/render_depth/"),
        'pose_file': os.path.join(data_root_dir, f"data/{scene_id}/iphone/colmap/images.txt"),
        'intrinsic_file': os.path.join(data_root_dir, f"data/{scene_id}/iphone/colmap/cameras.txt"),
        'anno_file': os.path.join(data_root_dir, f"data/{scene_id}/scans/segments_anno.json"),
        'segments_file': os.path.join(data_root_dir, f"data/{scene_id}/scans/segments.json"),
        'instance_path': os.path.join(data_root_dir, f"semantic_2D_iphone/obj_ids/{scene_id}/"),
        if scene_id == 
        'dictionary_folder': os.path.join(output_root_dir, f"dictionary/{scene_name}/")
    }
    # Setup output directories
    config.update({
        'line_image_folder': os.path.join(output_root_dir, f"line_map_extractor/out/{scene_id}/map/rgb_line_image/"),
        'line_mesh_folder': os.path.join(output_root_dir, f"line_map_extractor/out/{scene_id}/map/line_mesh/"),
        'line_data_folder': os.path.join(output_root_dir, f"line_map_extractor/out/{scene_id}/map/")
    })
    # Create output directories
    for out_path in [config['line_image_folder'], config['line_mesh_folder'], config['dictionary_folder']]:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    # load obj_labels
    with open(config['anno_file'], "r") as f:
        anno = json.load(f)
    objId_label_dict = {obj['id']: obj['label'] for obj in anno['segGroups']} 

    # Assign numeric labels to semantics
    label_id_dict = {}
    label_id = 1
    for obj, label in objId_label_dict.items():
        if label in label_id_dict:
            continue
        else:
            label_id_dict[label] = label_id
            label_id += 1
    # Map object IDs to label IDs
    objId_id_dict = {obj: label_id_dict[label] for obj, label in objId_label_dict.items()}
    objId_id_dict[0] = 0

    id_label_dict = {v:k for k,v in label_id_dict.items()}
    # update config
    config.update({
        'objId_id_dict': objId_id_dict,
        'id_label_dict': id_label_dict
    })

    return config

def process_id_remapping(config):
    """remap semantic labels to delete unwanted labels and merge similar labels """
    # the id_remapping_file is a txt file with two columns, the first column is the original semantic id, and the second column is the new id
    # if the second column is zero, the corresponding semantic is deleted
    # if the second column is not zero, the original semantic is remapped to the new semantic, e.g. door frame -> door 
    id_remapping_file = os.path.join(config['dictionary_folder'], "id_remapping.txt")
    dictionary_file = os.path.join(config['dictionary_folder'], "dictionary.txt")
    
    # Read remapping
    with open(id_remapping_file, "r") as f:
        id_remapping = {}
        for line in f:
            id_1, id_2 = map(int, line.strip().split(","))
            id_remapping[id_1] = id_2

    # Output remapping results
    with open(dictionary_file, "w") as dict_file:
        dict_file.write("##########adopted semantic labels##########\n")
        for v, k in config['id_label_dict'].items():
            if v not in id_remapping:
                dict_file.write(f"{v},{k}\n")
                
        dict_file.write("\n##########deleted semantic labels##########\n")
        for v, k in config['id_label_dict'].items():
            if v in id_remapping:
                dict_file.write(f"{v},{k}\n")

    return id_remapping

def load_ref_image_lists(config):
    """Load images with both rgb and depth data"""
    depth_img_list = sorted(glob.glob(config['depth_image_folder'] + "*.png"))
    rgb_img_list = sorted(glob.glob(config['rgb_folder'] + "*.jpg"))
    with open(config['ref_list'], "r") as f:
        ref_list = f.readlines()
    ref_list = [line.strip() for line in ref_list]
    rgb_img_list = [img for img in rgb_img_list if os.path.basename(img) in ref_list]
    # Remove depth images without corresponding RGB images
    k = 0
    while k < len(depth_img_list):
        depth_img_name = depth_img_list[k]
        basename = os.path.basename(depth_img_name).split(".")[0]
        rgb_file = config['rgb_folder']+basename+".jpg"
        if rgb_file not in rgb_img_list:
            depth_img_list.remove(depth_img_name)
        else:
            k += 1
    return depth_img_list

def process_file(depth_img_name, config, pose_data, id_remapping):
    """Process a single image file to extract 2D and 3D lines"""
    basename = os.path.basename(depth_img_name).split(".")[0]
    intrinsic = pose_data[basename]["intrinsic"]
    pose_matrix = np.array(pose_data[basename]["aligned_pose"])

    print(f"Processing {basename}")
    line_2D_end_points = []
    line_2D_points = []
    line_2D_params = []
    line_2D_semantic_id = []
    line_2D_match_idx = []
    line_3D_end_points = []
    line_3D_semantic_id = []
    proj_error_r = []
    proj_error_t = []

    # Load images
    render_depth = cv2.imread(depth_img_name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000  # millimeter to meter
    rgb_file = config['rgb_folder']+basename+".jpg"
    objId_file = os.path.join(config['instance_path'], str(os.path.basename(depth_img_name)).replace(".png", ".jpg.npy"))
    obj_ids = np.load(objId_file)
    gray_img = cv2.imread(rgb_file, cv2.IMREAD_GRAYSCALE)
    rgb_img = cv2.imread(rgb_file)

    # Extract and Prune 2D Line segments
    segments = helper.extract_and_prune_2Dlines(gray_img, helper.params_2D_map)
    line_2D_count = 0  # valid 2D line id in the cur image

    for j, segment in enumerate(segments):
        x1, y1, x2, y2 = segment.astype(np.int32)
        if x2 >= render_depth.shape[1] or y1 >= render_depth.shape[0] or y2 >= render_depth.shape[0]:
            continue

        # Get all pixels on the line
        if x1 == x2:  # special case for a vertical line
            y = np.arange(min(y1, y2), max(y1, y2))
            x = np.ones_like(y) * x1
        else:
            m, c = helper.get_line_eq(x1, y1, x2, y2)  # y = mx + c
            if abs(m) > 1:  # sample points on the longer axis
                y = np.arange(min(y1, y2), max(y1, y2))
                x = (y - c) / m
            else:
                x = np.arange(min(x1, x2), max(x1, x2))
                y = m * x + c

        x = x.astype(np.int32)
        y = y.astype(np.int32)
        v = np.array([x2-x1, y2-y1])
        if np.linalg.norm(v) == 0:
            continue
        v = v/np.linalg.norm(v)

        # Get the foreground points by multi-hypothesis perturbation
        depth_mean, xyz_list, foreground_idices, background_flag = helper.perturb_and_extract(
            x, y, render_depth, v, helper.params_2D_map["num_hypo"]*2+1
        )
        if np.min(depth_mean) == 255:  # no valid points
            continue

        # Process best hypothesis
        best_one = np.argmin(depth_mean)
        foreground_x = xyz_list[best_one][foreground_idices[best_one],0].astype(np.int32)
        foreground_y = xyz_list[best_one][foreground_idices[best_one],1].astype(np.int32)
        semantic_id = helper.extract_dominant_id(foreground_y, foreground_x, obj_ids, config['objId_id_dict'])

        # Check adjacent hypotheses
        while (best_one < helper.params_2D_map["num_hypo"]):
            if depth_mean[best_one+1]-depth_mean[best_one] > helper.params_2D_map["background_depth_diff_thresh"]:
                break
            foreground_x = xyz_list[best_one+1][foreground_idices[best_one+1],0].astype(np.int32)
            foreground_y = xyz_list[best_one+1][foreground_idices[best_one+1],1].astype(np.int32)
            cur_semantic_id = helper.extract_dominant_id(foreground_y, foreground_x, obj_ids, config['objId_id_dict'])
            if cur_semantic_id == semantic_id:
                best_one = best_one+1
            else:
                break

        while (best_one > helper.params_2D_map["num_hypo"]):
            if depth_mean[best_one-1]-depth_mean[best_one] > helper.params_2D_map["background_depth_diff_thresh"]:
                break
            foreground_x = xyz_list[best_one-1][foreground_idices[best_one-1],0].astype(np.int32)
            foreground_y = xyz_list[best_one-1][foreground_idices[best_one-1],1].astype(np.int32)
            cur_semantic_id = helper.extract_dominant_id(foreground_y, foreground_x, obj_ids, config['objId_id_dict'])
            if cur_semantic_id == semantic_id:
                best_one = best_one-1
            else:
                break

        # Get final foreground points
        foreground_x = xyz_list[best_one][foreground_idices[best_one],0].astype(np.int32)
        foreground_y = xyz_list[best_one][foreground_idices[best_one],1].astype(np.int32)
        foreground_z = xyz_list[best_one][foreground_idices[best_one],2]
        semantic_id = helper.extract_dominant_id(foreground_y, foreground_x, obj_ids, config['objId_id_dict'])
        if semantic_id in id_remapping:
            semantic_id = id_remapping[semantic_id]
    
        if semantic_id == 0: 
            continue
        # Regress the 3D line with found foreground points
        points_2D = np.concatenate([foreground_x[:, None], foreground_y[:, None], np.ones_like(foreground_x)[:, None]], axis=1)
        points_camera_3D = (np.linalg.inv(intrinsic) @ (points_2D * foreground_z[:, None]).T).T

        # Transform to world frame
        points_world_3D = (pose_matrix @ np.concatenate([points_camera_3D, np.ones((points_camera_3D.shape[0], 1))], axis=1).T)
        points_world_3D = points_world_3D[:3, :].T

        # RANSAC line fitting
        try:
            model_robust, inliers = ransac(
                points_world_3D, LineModelND, min_samples=3, residual_threshold=0.02, max_trials=3000
            )
        except:
            continue

        # Check inlier count
        inlier_points = points_world_3D[np.where(inliers == True)]
        if len(inlier_points) < helper.params_2D_map["line_points_num_thresh"]:
            continue

        # Get line parameters
        p, v = model_robust.params[0:2]
        sig_dim = np.argmax(abs(v))
        min_val = np.min(inlier_points[:, sig_dim])
        max_val = np.max(inlier_points[:, sig_dim])

        # Regulate vector v if close to principle axis
        if np.abs(np.dot(v, np.array([1, 0, 0]))) > helper.params_merge_prune["parrallel_thresh_3D"]:
            v = np.array([1, 0, 0])
        elif np.abs(np.dot(v, np.array([0, 1, 0]))) > helper.params_merge_prune["parrallel_thresh_3D"]:
            v = np.array([0, 1, 0])
        elif np.abs(np.dot(v, np.array([0, 0, 1]))) > helper.params_merge_prune["parrallel_thresh_3D"]:
            v = np.array([0, 0, 1])
        # find two endpoints along direction vector v and pass point p
        point_min = p + v/v[sig_dim]*(min_val-p[sig_dim])
        point_max = p + v/v[sig_dim]*(max_val-p[sig_dim])

        # calculate 2D line parameters
        x1, y1 = xyz_list[best_one][0,0:2]
        x2, y2 = xyz_list[best_one][-1,0:2]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x1 == x2:
            A, B, C = 1, 0, -x1
        else:
            m, c = helper.get_line_eq(x1, y1, x2, y2)
            A, B, C = m, -1, c
        line_2D_param_pixel = np.array([A, B, C])
        
        # Calculate projection error
        error_rot, error_trans = helper.calculate_error(
            line_2D_param_pixel.reshape(1,3), v, intrinsic, pose_matrix, point_min, point_max
        )
        if error_rot > 0.1 or error_trans > 0.1: # discard lines with large projection error
            print(f"Warning: large projection error for {basename} at index {j}, error_rot ={error_rot}, error_t={error_trans}")
            continue

        ### Store data
        # 3D line 
        line_3D_semantic_id.append(semantic_id)
        line_3D_end_points.append([point_min, point_max])
        # 2D kube
        line_2D_params.append(line_2D_param_pixel)
        line_2D_points.append([xyz_list[best_one][:,0], xyz_list[best_one][:,1]])
        line_2D_end_points.append([[x1,y1], [x2,y2]])
        line_2D_semantic_id.append(semantic_id)
        line_2D_match_idx.append(len(line_3D_semantic_id) - 1)
        proj_error_r.append(np.abs(error_rot))
        proj_error_t.append(np.abs(error_trans))

        # Draw 2D lines with semantic labels
        line_2D_count += 1
        cv2.line(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rgb_img, str(line_2D_count), (int((x1 + x2) / 2), int((y1 + y2) / 2)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.putText(rgb_img, config['id_label_dict'][semantic_id], (x1, y1), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        if background_flag[best_one]:
            cv2.line(rgb_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if len(line_3D_semantic_id) < 5:
        return (basename, [], [], [], [], [], [], [], [], [])

    # Save annotated image
    cv2.imwrite(os.path.join(config['line_image_folder'], f"{basename}.jpg"), rgb_img)

    return (basename, line_2D_points, line_2D_end_points, line_2D_params, line_2D_semantic_id, 
            line_2D_match_idx, line_3D_semantic_id, line_3D_end_points, proj_error_r, proj_error_t)

def aggregate_results(results, pose_data):
    """Aggregate processed results from all images"""
    scene_data = {
        'scene_pose': {},
        'scene_intrinsic': {},
        'scene_line_2D_points': {},
        'scene_line_2D_end_points': {},
        'scene_line_2D_semantic_ids': {},
        'scene_line_2D_params': {},
        'scene_line_2D_match_idx_raw': {},
        'scene_proj_error_r_raw': {},
        'scene_proj_error_t_raw': {},
        'scene_line_3D_semantic_ids': [],
        'scene_line_3D_end_points': [],
        'scene_line_3D_image_source': []
    }
    
    for result in results:            
        # Unpack results
        (basename, line_2D_points, line_2D_end_points, line_2D_params, 
         line_2D_semantic_id, line_2D_match_idx, line_3D_semantic_id, 
         line_3D_end_points, proj_error_r, proj_error_t) = result
        if line_2D_points == []:  # Skip empty results
            continue

        # Store 2D data
        scene_data['scene_line_2D_points'][basename] = line_2D_points
        scene_data['scene_line_2D_end_points'][basename] = line_2D_end_points
        scene_data['scene_line_2D_params'][basename] = line_2D_params
        scene_data['scene_line_2D_semantic_ids'][basename] = line_2D_semantic_id
        
        # Update match indices
        for i in range(len(line_2D_match_idx)):
            line_2D_match_idx[i] += len(scene_data['scene_line_3D_semantic_ids'])
        scene_data['scene_line_2D_match_idx_raw'][basename] = np.array(line_2D_match_idx)
        
        # Store 3D data
        scene_data['scene_line_3D_semantic_ids'].extend(line_3D_semantic_id)
        scene_data['scene_line_3D_end_points'].extend(line_3D_end_points)
        image_index = np.int32(basename[-6:])
        scene_data['scene_line_3D_image_source'].extend([image_index] * len(line_3D_semantic_id))
        
        # Store error data
        scene_data['scene_proj_error_r_raw'][basename] = proj_error_r
        scene_data['scene_proj_error_t_raw'][basename] = proj_error_t
        
        # Store pose and intrinsic
        scene_data['scene_pose'][basename] = pose_data[basename]["aligned_pose"]
        scene_data['scene_intrinsic'][basename] = pose_data[basename]["intrinsic"]
    
    # Add metadata
    scene_data.update({
        'params_2D_map': helper.params_2D_map
    })
    
    return scene_data

def save_results_and_mesh(config, results_dict):
    """Save processed results to files"""
    # Save numpy results
    np.save(os.path.join(config['line_data_folder'], f"{config['scene_id']}_results_raw.npy"), results_dict)
    print("Save results in numpy file successfully.")
    
    # Save visualization mesh
    point_sets = []
    for i in range(len(results_dict['scene_line_3D_semantic_ids'])):
        point_a, point_b = results_dict['scene_line_3D_end_points'][i][0:2]
        point_diff = point_b - point_a
        for sample in range(300):
            point_sets.append(point_a + point_diff*sample/299)
            
    point_sets = np.vstack(point_sets)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_sets)
    o3d.io.write_point_cloud(
        os.path.join(config['line_mesh_folder'], f"{config['scene_id']}_raw_3D_line_mesh.ply"), 
        pcd
    )
    print("Save raw 3D line mesh successfully.")

def main(data_root_dir,output_root_dir,scene_id,scene_name):
    # Load configuration
    config = load_config(data_root_dir,output_root_dir,scene_id,scene_name)
    
    # Process label remapping
    id_remapping = process_id_remapping(config)
    
    # Load image lists
    depth_img_list = load_ref_image_lists(config)
    
    # Load pose data
    pose_data = {}
    with open(config['intrinsic_file'], "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            _, _, width, height, fx, fy, cx, cy, k1, k2, p1, p2 = line.strip().split()
            fx, fy, cx, cy, k1, k2, p1, p2 = map(float, [fx, fy, cx, cy, k1, k2, p1, p2])
            intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            distortion_params = np.array([k1, k2, p1, p2])

    with open(config['pose_file'], "r") as f:
        for line in f:
            if line.startswith("#") or line.startswith("\n"):
                continue
            _, qw, qx, qy, qz, tx, ty, tz, _, basename = line.strip().split()
            basename = basename.split(".")[0]
            rot_matrix = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = rot_matrix.T
            tx, ty, tz = map(float, [tx, ty, tz])
            pose_matrix[:3, 3] = -rot_matrix.T @ np.array([tx, ty, tz])
            pose_data[basename] = {
                "aligned_pose": pose_matrix,
                "intrinsic": intrinsic_matrix,
                "distortion_params": distortion_params
            }

    # # Process images one by one (for debugging)
    # results=[]
    # for depth_img_name in tqdm(depth_img_list):
    #     # num = int(depth_img_name[-8:-4])
    #     # if num < 5000:
    #     #     continue
    #     result = process_file(
    #         depth_img_name,
    #         config,
    #         pose_data,
    #         id_remapping
    #     )    
    #     results.append(result)

    # Process images in parallel
    results = Parallel(n_jobs=helper.thread_number)(
        delayed(process_file)(
            depth_img_name,
            config,
            pose_data,
            id_remapping
        ) for depth_img_name in depth_img_list
    )

    # Aggregate results
    results_dict = aggregate_results(results, pose_data)
    results_dict['id_label_dict'] = config['id_label_dict']
    
    # Save results as a numpy file and output line mesh as ply files
    save_results_and_mesh(config, results_dict)
    print("Process completed.")


if __name__ == "__main__":
    data_root_dir = "/data2/scannetppv2"
    output_root_dir = "/data1/home/lucky/IROS25/SCORE/"
    scene_list = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"]
    scene_names = ["S1","S2","S3","S4"]
    scene_id = scene_list[0]
    scene_name = scene_names[0]
    main(data_root_dir,output_root_dir,scene_id,scene_name)


