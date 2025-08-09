# load the query images
# extract 2D lines and semantics
# extract corresponding 3D lines to find the cloest match stored in the map
# save the results
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
from scipy.spatial.transform import Rotation

def load_config(data_root_dir,output_root_dir,scene_id,scene_name): 
    """Load configuration and paths""" 
    # Setup input directories
    config = {
        'data_root_dir': data_root_dir,
        'output_root_dir': output_root_dir,
        'scene_id': scene_id,
        'query_list': os.path.join(output_root_dir, f"line_map_extractor/out/{scene_id}/query.txt"),
        'rgb_folder': os.path.join(data_root_dir, f"data/{scene_id}/iphone/rgb/"),
        'depth_image_folder': os.path.join(data_root_dir, f"data/{scene_id}/iphone/render_depth/"),
        'pose_file': os.path.join(data_root_dir, f"data/{scene_id}/iphone/colmap/images.txt"),
        'intrinsic_file': os.path.join(data_root_dir, f"data/{scene_id}/iphone/colmap/cameras.txt"),
        # 'retriveal_file': os.path.join(data_root_dir, f"best_view_cache_iphone/iphone/{scene_id}_similar.json"),
        'retriveal_file': os.path.join(output_root_dir, f"line_map_extractor/NetVLAD20/{scene_id}/pairs-loc.txt"),
        'anno_file': os.path.join(data_root_dir, f"data/{scene_id}/scans/segments_anno.json"),
        'segments_file': os.path.join(data_root_dir, f"data/{scene_id}/scans/segments.json"),
        'instance_path': os.path.join(data_root_dir, f"semantic_2D_iphone/obj_ids/{scene_id}/"),
        'dictionary_folder': os.path.join(output_root_dir, f"dictionary/{scene_name}/"),
        'line_map_file': os.path.join(output_root_dir, f"line_map_extractor/out/{scene_id}/map/{scene_id}_results_merged.npy")
    }
    # Setup output directories
    config.update({
        'line_image_folder': os.path.join(output_root_dir, f"line_map_extractor/out/{scene_id}/query/rgb_line_image/"),
        'line_data_folder': os.path.join(output_root_dir, f"line_map_extractor/out/{scene_id}/query/")
    })
    # Create output directories
    for out_path in [config['line_image_folder'], config['dictionary_folder']]:
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

def load_query_image_lists(config):
    """Load images with both rgb and depth data"""
    depth_img_list = sorted(glob.glob(config['depth_image_folder'] + "*.png"))
    rgb_img_list = sorted(glob.glob(config['rgb_folder'] + "*.jpg"))
    with open(config['query_list'], "r") as f:
        query_list = f.readlines()
    query_list = [line.strip() for line in query_list]
    rgb_img_list = [img for img in rgb_img_list if os.path.basename(img) in query_list]
    # depth_img_list = depth_img_list[::2]  # downsample
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

def process_file(depth_img_name, config, pose_data, id_remapping, line_3D_end_points, line_3D_semantic_ids):
    """Process a single image file to extract 2D and 3D lines"""
    basename = os.path.basename(depth_img_name).split(".")[0]
    intrinsic = pose_data[basename]["intrinsic"]
    pose_matrix = np.array(pose_data[basename]["aligned_pose"])

    # print(f"Processing {basename}")
    line_2D_end_points = []
    line_2D_points = []
    line_2D_params = []
    line_2D_semantic_id = []
    line_2D_match_idx = []
    proj_error_r = []
    proj_error_t = []

    # Load images
    render_depth = cv2.imread(depth_img_name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000  # millimeter to meter
    rgb_file = config['rgb_folder'] + basename + ".jpg"
    objId_file = os.path.join(config['instance_path'], str(os.path.basename(depth_img_name)).replace(".png", ".jpg.npy"))
    obj_ids = np.load(objId_file)
    gray_img = cv2.imread(rgb_file, cv2.IMREAD_GRAYSCALE)
    rgb_img = cv2.imread(rgb_file)

    # Extract and Prune 2D Line segments
    segments = helper.extract_and_prune_2Dlines(gray_img, helper.params_query)
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
            x, y, render_depth, v, helper.params_query["num_hypo"]*2+1
        )
        if np.min(depth_mean) == 255:  # no valid points
            continue

        # Process best hypothesis
        best_one = np.argmin(depth_mean)
        foreground_x = xyz_list[best_one][foreground_idices[best_one],0].astype(np.int32)
        foreground_y = xyz_list[best_one][foreground_idices[best_one],1].astype(np.int32)
        semantic_id = helper.extract_dominant_id(foreground_y, foreground_x, obj_ids, config['objId_id_dict'])

        # Check adjacent hypotheses
        while (best_one < helper.params_query["num_hypo"]):
            if depth_mean[best_one+1]-depth_mean[best_one] > helper.params_query["background_depth_diff_thresh"]:
                break
            foreground_x = xyz_list[best_one+1][foreground_idices[best_one+1],0].astype(np.int32)
            foreground_y = xyz_list[best_one+1][foreground_idices[best_one+1],1].astype(np.int32)
            cur_semantic_id = helper.extract_dominant_id(foreground_y, foreground_x, obj_ids, config['objId_id_dict'])
            if cur_semantic_id == semantic_id:
                best_one = best_one+1
            else:
                break

        while (best_one > helper.params_query["num_hypo"]):
            if depth_mean[best_one-1]-depth_mean[best_one] > helper.params_query["background_depth_diff_thresh"]:
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
        # Find the closest 3D line
        pixel_a = np.array([x1, y1])
        pixel_b = np.array([x2, y2])
        v_2D = pixel_b - pixel_a
        v_2D = v_2D/np.linalg.norm(v_2D)
        n_2D = line_2D_param_pixel
        matched_3D_lined_idx = helper.find_closest_3D_line(v_2D, n_2D, semantic_id, intrinsic, pose_matrix, line_3D_end_points, line_3D_semantic_ids) 
        if np.isnan(matched_3D_lined_idx):
            error_rot = np.nan
            error_trans = np.nan
            print(f"No 3D line found for {basename} at line NO.{line_2D_count+1}")
        else:
            point_matched_3D = line_3D_end_points[matched_3D_lined_idx]
            point_matched_3D_a, point_matched_3D_b = point_matched_3D[0], point_matched_3D[1]
            v_matched_3D = point_matched_3D_b - point_matched_3D_a
            v_matched_3D = v_matched_3D / np.linalg.norm(v_matched_3D)
            error_rot, error_trans = helper.calculate_error(
                line_2D_param_pixel.reshape(1,3), v_matched_3D, intrinsic, pose_matrix, point_matched_3D_a, point_matched_3D_b
            )
            # print(f"Error_rot: {error_rot}, Error_trans: {error_trans} for {basename} at line NO.{line_2D_count+1}")
        ### Store data
        # 2D kube
        line_2D_params.append(line_2D_param_pixel)
        line_2D_points.append([xyz_list[best_one][:,0], xyz_list[best_one][:,1]])
        line_2D_end_points.append([[x1,y1], [x2,y2]])
        line_2D_semantic_id.append(semantic_id)
        line_2D_match_idx.append(matched_3D_lined_idx)
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

    # Save annotated image
    cv2.imwrite(os.path.join(config['line_image_folder'], f"{basename}.jpg"), rgb_img)

    return (basename, line_2D_points, line_2D_end_points, line_2D_params, line_2D_semantic_id, 
            line_2D_match_idx, proj_error_r, proj_error_t)

def aggregate_results(results, pose_data, config):
    """Aggregate processed results from all images"""
    scene_data = {
        'scene_pose': {},
        'scene_intrinsic': {},
        'scene_line_2D_points': {},
        'scene_line_2D_end_points': {},
        'scene_line_2D_semantic_ids': {},
        'scene_line_2D_params': {},
        'scene_line_2D_match_idx': {},
        'scene_proj_error_r': {},
        'scene_proj_error_t': {},
        'scene_retrived_3D_line_idx': {},
        'scene_retrived_poses': {}
    }
    #
    line_map_file = np.load(config['line_map_file'], allow_pickle=True).item()
    scene_line_2D_match_idx = line_map_file['scene_line_2D_match_idx']
    scene_pose = line_map_file['scene_pose']
    
    # read the image retrieval file
    pairs = {}
    retrived_poses = {}
    with open(config['retriveal_file'], "r") as f:
        lines = f.readlines()
        for line in lines:
            query_name = line.strip().split(" ")[0]
            paired_name = line.strip().split(" ")[1].split(".")[0]
            if query_name not in pairs:
                count = 0
                pairs[query_name] = []
                retrived_poses[query_name] = {}
            if paired_name in scene_pose:
                if count < 12:
                    pairs[query_name].append(paired_name)
                    retrived_poses[query_name][count] = scene_pose[paired_name]
                    count += 1

    for result in results:            
        # Unpack results
        (basename, line_2D_points, line_2D_end_points, line_2D_params, 
         line_2D_semantic_id, line_2D_match_idx, proj_error_r, proj_error_t) = result
        if line_2D_points == []:  # Skip empty results
            continue
        # Store 2D data
        scene_data['scene_line_2D_points'][basename] = line_2D_points
        scene_data['scene_line_2D_end_points'][basename] = line_2D_end_points
        scene_data['scene_line_2D_params'][basename] = line_2D_params
        scene_data['scene_line_2D_semantic_ids'][basename] = line_2D_semantic_id
        scene_data['scene_line_2D_match_idx'][basename] = np.array(line_2D_match_idx)
        # Store error data
        scene_data['scene_proj_error_r'][basename] = proj_error_r
        scene_data['scene_proj_error_t'][basename] = proj_error_t
        
        # Store pose and intrinsic
        scene_data['scene_pose'][basename] = pose_data[basename]["aligned_pose"]
        scene_data['scene_intrinsic'][basename] = pose_data[basename]["intrinsic"]

        # Store retrived 3D line idx
        key = basename+".jpg"
        retrived_img_lists = pairs[key]
        retrived_3D_line_idx = line_2D_match_idx
        for img in retrived_img_lists:
            img_basename = img.split(".")[0]
            if img_basename in scene_line_2D_match_idx:
                appended_idx = scene_line_2D_match_idx[img_basename]
                retrived_3D_line_idx = np.concatenate([retrived_3D_line_idx, appended_idx])
        retrived_3D_line_idx = np.unique(retrived_3D_line_idx)
        # delete the nan values
        retrived_3D_line_idx = retrived_3D_line_idx[~np.isnan(retrived_3D_line_idx)]
        scene_data['scene_retrived_3D_line_idx'][basename] = retrived_3D_line_idx.astype(np.int32)
        scene_data['scene_retrived_poses'][basename] = retrived_poses[key]
    # Add metadata
    scene_data.update({
        'params_query': helper.params_query,
    })
    
    return scene_data
def main(data_root_dir,output_root_dir,scene_id,scene_name):
    # Load configuration
    config = load_config(data_root_dir,output_root_dir,scene_id,scene_name)
    # Load id remapping
    with open(os.path.join(config['dictionary_folder'], "id_remapping.txt"), "r") as f:
        id_remapping = {}
        for line in f:
            id_1, id_2 = map(int, line.strip().split(","))
            id_remapping[id_1] = id_2
    # Load image lists
    depth_img_list = load_query_image_lists(config)
    
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
            # colmap: world to camera --> camera to world
            pose_matrix[:3, :3] = rot_matrix.T
            tx, ty, tz = map(float, [tx, ty, tz])
            pose_matrix[:3, 3] = -rot_matrix.T @ np.array([tx, ty, tz])
            pose_data[basename] = {
                "aligned_pose": pose_matrix,
                "intrinsic": intrinsic_matrix,
                "distortion_params": distortion_params
            }

    # Load 3D lines
    line_map = np.load(config['line_map_file'], allow_pickle=True).item()
    line_3D_end_points = line_map['merged_end_points_3D']
    line_3D_semantic_ids = line_map['merged_semantic_ids_3D']
    scene_line_2D_match_idx = line_map['scene_line_2D_match_idx']

    # Process images one by one (for debugging)
    # results=[]
    # for depth_img_name in tqdm(depth_img_list):
    #     # num = int(depth_img_name[-8:-4])
    #     # if num < 9500:
    #     #     continue
    #     result = process_file(
    #     depth_img_name, config, pose_data, id_remapping, line_3D_end_points, line_3D_semantic_ids
    #     )    
    #     results.append(result)

    # Process images in parallel
    results = Parallel(n_jobs=helper.thread_number)(
        delayed(process_file)(
            depth_img_name, config, pose_data, id_remapping, line_3D_end_points, line_3D_semantic_ids
        ) for depth_img_name in depth_img_list
    )

    # Retrive sub 3D line map and Aggregate results
    query_data_dict = aggregate_results(results, pose_data,config)
    query_data_dict['id_label_dict'] = config['id_label_dict']

    # Save results as a numpy file
    np.save(os.path.join(config['line_data_folder'], f"{config['scene_id']}_query_data.npy"), query_data_dict)
    print("Save data successfully.")
    print("Process completed.")
    
if __name__ == "__main__":
    data_root_dir = "/data2/scannetppv2"
    output_root_dir = "/data1/home/lucky/IROS25/SCORE/"
    scene_list = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"]
    scene_names = ["S1","S2","S3","S4"]
    scene_id = scene_list[0]
    scene_name = scene_names[0]
    main(data_root_dir,output_root_dir,scene_id,scene_name)