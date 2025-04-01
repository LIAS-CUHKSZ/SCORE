"""
line_extractor_pt1.py 

This script extracts 2D lines from images and regresses 3D lines from the 2D lines based on pose and depth.
We assign each pair of 2D and 3D line with the same semantic label. 
You can tune the paramters defined in helper.py, and edit the label_remapping.txt file.

Output:
- Rgb images annotated with extracted lines thier semantic labels. 
- Mesh file(.ply) with all regressed 3D lines.
- A numpy file containing all the extracted 2D lines and regressed 3D lines.

Author: Haodong JIANG <221049033@link.cuhk.edu.cn>
Version: 1.0
License: MIT
"""

import helper
import cv2
import numpy as np
import glob
import os
import json
import open3d as o3d
from skimage.measure import LineModelND, ransac
# from collections import Counter
from joblib import Parallel, delayed
from scipy import stats

################################### Loading and Configuring ###################################
home = os.path.expanduser("~")
scene_list = ["69e5939669","689fec23d7","c173f62b15","55b2bf8036"]
scene_id = scene_list[2]
rgb_folder = home+f"/SCORE/dataset/{scene_id}/rgb/"
depth_image_folder = home+f"/SCORE/dataset/{scene_id}/render_depth/"
depth_img_list = sorted(glob.glob(depth_image_folder + "*.png"))
rgb_img_list = sorted(glob.glob(rgb_folder + "*.jpg"))
# remove the depth images that do not have corresponding rgb images
k = 0
while(k<len(depth_img_list)):
    depth_img_name = depth_img_list[k]
    rgb_file = depth_img_name.replace("render_depth", "rgb").replace("png", "jpg")
    if rgb_file not in rgb_img_list:
        depth_img_list.remove(depth_img_name)
    else:
        k=k+1
pose_file = home+f"/SCORE/dataset/{scene_id}/pose_intrinsic_imu.json" # camera pose to world
anno_file = home+f"/SCORE/dataset/{scene_id}/scans/segments_anno.json"
segments_file = home+f"/SCORE/dataset/{scene_id}/scans/segments.json"
instance_path = home+f"/SCORE/dataset/{scene_id}/obj_ids/"
### result saving path
dictionary_folder = home+f"/SCORE/dictionary/{scene_id}/"
line_image_folder = home+f"/SCORE/line_map_extractor/out/{scene_id}/rgb_line_image/"   # rgb images with extracted 2d lines and labels
line_mesh_raw_folder = home+f"/SCORE/line_map_extractor/out/{scene_id}/line_mesh_raw/" # regressed 3d lines
line_data_folder = home+f"/SCORE/line_map_extractor/out/{scene_id}/" # numpy file with all the extracted 2d lines and regressed 3d lines
for out_path in [line_image_folder, line_mesh_raw_folder,dictionary_folder]:
    if not os.path.exists(out_path):
        os.makedirs(out_path)
label_remapping_file = dictionary_folder+"label_remapping.txt"
dictionary_file = dictionary_folder+"dictionary.txt"
### load data
with open(pose_file, "r") as f:  # load pose
    pose_data = json.load(f)
with open(anno_file, "r") as f:  # load semantics for each object
    anno = json.load(f)
obj_label = {}
for obj in anno['segGroups']:
    obj_label[obj['id']] = obj['label']
# give number label for each semantic
labels = {}
label_id = 1
for obj, label in obj_label.items():
    if label in labels:
        continue
    else:
        labels[label] = label_id
        label_id += 1
# estabish mapping between object id and semantic label
obj_id_label_id = {}
for obj, label in obj_label.items():
    obj_id_label_id[obj] = labels[label]
obj_id_label_id[0] = 0
label_2_semantic_dict = {v:k for k,v in labels.items()}

#### remap semantic labels to delete unwanted labels and merge similar labels 
# the label_remapping_file is a txt file with two columns, the first column is the original label, and the second column is the new label
# if the second column is zero, the label is deleted
# if the second column is not zero, the original label is remapped to the new label 
label_remapping_file = open(label_remapping_file, "r")
label_remapped = np.array(range(len(label_2_semantic_dict)))+1
for line in label_remapping_file:
    line = line.strip().split(",")
    label_1 = int(line[0])
    label_2 = int(line[1])
    label_remapped[label_1-1] = label_2
new_label_2_semantic_dict = {}
# output the adopted and deleted labels to dictionary.txt
with open(dictionary_file, "w") as dict_file:
    dict_file.write("##########adopted semantic labels##########\n")
    print("adopted semantic labels")
    for v, k in label_2_semantic_dict.items():
        if label_remapped[int(v) - 1] == int(v):
            new_label_2_semantic_dict[v] = k
            dict_file.write(f"{v},{k}\n")
            print(v, k)
    
    dict_file.write("\n##########deleted semantic labels##########\n")
    print("deleted semantic labels")
    for v, k in label_2_semantic_dict.items():
        if label_remapped[int(v) - 1] != int(v):
            dict_file.write(f"{v},{k}\n")
            print(v, k)

################################### Processing and Saving ###################################
### Initialize data sturcture to be stored
scene_pose = {}
scene_intrinsic = {}
# Store 2d lines from different images separately  
scene_line_2d_points = {}
scene_line_2d_end_points = {}
scene_line_2d_semantic_labels = {}
scene_line_2d_params = {}
scene_line_2d_match_idx = {}
scene_proj_error_r_raw = {}
scene_proj_error_t_raw = {}
# Store all 3d lines together 
scene_line_3d_semantic_labels = []
scene_line_3d_params = []
scene_line_3d_end_points = []
scene_line_3d_image_source = []
### function for processing a single image     
def process_file(i, depth_img_name):
    basename = os.path.basename(depth_img_name).split(".")[0]
    intrinsic = pose_data[basename]["intrinsic"]
    pose_matrix = np.array(pose_data[basename]["aligned_pose"])
    #
    print(f"Processing {basename}")
    line_2d_end_points = []
    line_2d_points=[]
    line_2d_params = []
    line_2d_semantic_label = []
    line_2d_match_idx = []
    #
    line_3d_params = []
    line_3d_end_points = []
    line_3d_semantic_label = []
    #
    proj_error_r = []
    proj_error_t = []
    # 
    render_depth = cv2.imread(depth_img_name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000 # millimeter to meter
    rgb_file = depth_img_name.replace("render_depth", "rgb").replace("png", "jpg") # read rgb
    obj_id_file = os.path.join(instance_path, str(os.path.basename(depth_img_name)).replace(".png", ".jpg.npy"))
    obj_ids = np.load(obj_id_file)
    rgb = cv2.imread(rgb_file, cv2.IMREAD_GRAYSCALE)
    rgb_color = cv2.imread(rgb_file)
    ### Extract and Prune 2d Line segments
    segments = helper.extract_and_prune_2dlines(rgb) 
    ### Regress 3d Lines
    line_2d_count = 0
    for j, segment in enumerate(segments):
        x1, y1, x2, y2 = segment.astype(np.int32)
        if x2 >= render_depth.shape[1] or y1 >= render_depth.shape[0] or y2 >= render_depth.shape[0]:
            continue
        # get all pixels on the line
        if x1 == x2:
            y = np.arange(min(y1, y2), max(y1, y2))
            x = np.ones_like(y) * x1
            A,B,C = 1,0,-x1
        else:
            m, c = helper.get_line_eq(x1, y1, x2, y2)  # y = mx + c
            A,B,C = m,-1,c
            if abs(m) > 1:  # sample points on the longer axis
                y = np.arange(min(y1, y2), max(y1, y2))
                x = (y - c) / m
            else:
                x = np.arange(min(x1, x2), max(x1, x2))
                y = m * x + c
        x = x.astype(np.int32) # since we only have depth on integer pixels
        y = y.astype(np.int32) # since we only have depth on integer pixels
        line_2d_param_pixel = np.array([A, B, C])
        v = np.array([x2-x1,y2-y1])
        if np.linalg.norm(v)==0:
            continue
        v = v/np.linalg.norm(v)
        ### Get the foregound points by multi-hypothesis pertubation
        num_hypo=helper.params_2d["num_hypo"]
        depth_mean,xyz_list,foreground_idices,background_flag = helper.perturb_and_extract(x,y,render_depth,v,num_hypo*2+1)
        best_one = np.argmin(depth_mean)
        while (best_one < num_hypo):
            if depth_mean[best_one+1]-depth_mean[best_one] < helper.params_2d["background_depth_diff_thresh"]:
                best_one = best_one+1
            else:
                break
        while (best_one>num_hypo):
            if depth_mean[best_one-1]-depth_mean[best_one] < helper.params_2d["background_depth_diff_thresh"]:
                best_one = best_one-1
            else:
                break
        if depth_mean[best_one]==255:
            print("extrct 3d points fail")
            error_rot   = -1
            error_trans = -1
            continue
        foreground_x = xyz_list[best_one][foreground_idices[best_one],0].astype(np.int32)
        foreground_y = xyz_list[best_one][foreground_idices[best_one],1].astype(np.int32)
        foreground_z = xyz_list[best_one][foreground_idices[best_one],2]
        ### get the (dominant) semantic label for this 2d line
        all_points_obj_ids = obj_ids[foreground_y, foreground_x]
        all_points_semantic_label = []
        for point_obj_id in all_points_obj_ids:
            all_points_semantic_label.append(obj_id_label_id[point_obj_id])
        all_points_semantic_label = np.array(all_points_semantic_label)
        unique_labels, counts = np.unique(all_points_semantic_label, return_counts=True)
        frequent_labels = unique_labels[counts > helper.params_2d["line_points_num_thresh"]]
        semantic_label = 0
        for label in frequent_labels:
            if label == 0:
                continue
            if label_remapped[label-1] != 0:
                semantic_label = label_remapped[label-1]
                break
        if semantic_label == 0: # no valid label for this line
            continue  
        else: 
            line_2d_points.append([x,y])
            # draw the 2D lines along with their semantic labels
            cv2.line(rgb_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rgb_color, str(line_2d_count), (int((x1 + x2) / 2), int((y1 + y2) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.putText(rgb_color, label_2_semantic_dict[semantic_label], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
            if background_flag[best_one]:
                cv2.line(rgb_color, (x1, y1), (x2, y2), (0, 0, 255), 2)
            line_2d_params.append(line_2d_param_pixel)
            line_2d_end_points.append([[x1,y1],[x2,y2]])
            line_2d_semantic_label.append(semantic_label)
            line_2d_count+=1
        ### regress the 3D line with found foreground points
        points_2d = np.concatenate([foreground_x[:, None], foreground_y[:, None], np.ones_like(foreground_x)[:, None]], axis=1)
        points_camera_3d = (np.linalg.inv(intrinsic) @ (points_2d * foreground_z[:, None]).T).T # get 3D points in the camera frame
        points_world_3d = (
            pose_matrix @ np.concatenate([points_camera_3d, np.ones((points_camera_3d.shape[0], 1))], axis=1).T
        ) # transform to the world frame
        points_world_3d = points_world_3d[:3, :].T
        try:
            model_robust, inliers = ransac(
                points_world_3d, LineModelND, min_samples=3, residual_threshold=0.02, max_trials=3000
            )
        except:
            print("extrct 3d line fails")
            cv2.line(rgb_color, (x1, y1), (int((x1+x2)/2), int((y1+y2)/2)), (255, 0, 0), 2) # mark the failed lines with red color
            error_rot   = -1
            error_trans = -1
            continue
        inlier_index = inliers == True
        # if the number of inliers is less than a threshold, we discard this 3d line
        if len(inlier_index) < helper.params_2d["line_points_num_thresh"]:
            print("extrct 3d line fails")
            cv2.line(rgb_color, (x1, y1), (int((x1+x2)/2), int((y1+y2)/2)), (255, 0, 0), 2) # mark the failed lines with red color
            error_rot   = -1
            error_trans = -1
            continue
        # obtain and store 3D line paramaters
        v = model_robust.params[1]
        sig_dim = np.argmax(abs(v))
        inlier_points = points_world_3d[inlier_index]
        min_index = np.argmin(inlier_points[:, sig_dim])
        max_index = np.argmax(inlier_points[:, sig_dim])
        point_min = inlier_points[min_index]
        point_max = inlier_points[max_index]
        p = (point_min+point_max)/2       
        line_3d_semantic_label.append(semantic_label)
        line_3d_end_points.append([point_min,point_max]) 
        line_3d_params.append([p,v])
        line_2d_match_idx.append(len(line_3d_params) - 1)
        # calculate projection error based on the pose
        error_rot,error_trans = helper.calculate_error(line_2d_param_pixel.reshape(1,3),v,intrinsic,pose_matrix,p)
        if error_rot > 0.1 or error_trans>0.1:
            print(basename,j,error_rot,error_trans)
        proj_error_r.append(np.abs(error_rot))
        proj_error_t.append(np.abs(error_trans))
    # save rgb images with 2D line and semantic annotation
    cv2.imwrite(os.path.join(line_image_folder, f"{basename}.jpg"), rgb_color)    
    return (basename, line_2d_points,line_2d_end_points, line_2d_params,line_2d_semantic_label, line_2d_match_idx, line_3d_semantic_label, line_3d_params, line_3d_end_points,proj_error_r,proj_error_t)
results=[]
## unparalled version for debugging:
# for i, depth_img_name in enumerate(depth_img_list):
#     if i > -1:
#         result = process_file(i,depth_img_name)
#         results.append(result)

### Use parallel processing to process files
results = Parallel(n_jobs=helper.params_2d["thread_number"])(delayed(process_file)(i, depth_img_name) for i, depth_img_name in enumerate(depth_img_list))
for result in results:
    basename, line_2d_points,line_2d_end_points, line_2d_params,line_2d_semantic_label, line_2d_match_idx, line_3d_semantic_label, line_3d_params, line_3d_end_points,proj_error_r,proj_error_t = result
    #
    scene_line_2d_points[basename] = line_2d_points
    scene_line_2d_end_points[basename] = line_2d_end_points
    scene_line_2d_params[basename] = line_2d_params
    scene_line_2d_semantic_labels[basename] = line_2d_semantic_label
    for i in range(len(line_2d_match_idx)):
        if line_2d_match_idx[i] != None:
            line_2d_match_idx[i] = line_2d_match_idx[i]+len(scene_line_3d_params)
    scene_line_2d_match_idx[basename] = np.array(line_2d_match_idx)
    #
    scene_line_3d_semantic_labels.extend(line_3d_semantic_label)
    scene_line_3d_params.extend(line_3d_params)
    scene_line_3d_end_points.extend(line_3d_end_points)
    image_index = np.int32(basename[-6:])
    scene_line_3d_image_source.extend(np.ones(len(line_3d_params))*image_index)
    #
    scene_proj_error_r_raw[basename]=proj_error_r
    scene_proj_error_t_raw[basename]=proj_error_t

for i in range(0,len(depth_img_list)):
    basename = os.path.basename(depth_img_list[i]).split(".")[0]
    intrinsic = pose_data[basename]["intrinsic"]
    scene_pose[basename] = pose_data[basename]["aligned_pose"]
    scene_intrinsic[basename] = intrinsic
### save the results
np.save(line_data_folder+f"results_raw.npy", {
    "scene_pose": scene_pose,
    "scene_intrinsic": scene_intrinsic,
    "obj_id_label_id": obj_id_label_id,
    "label_2_semantic_dict": new_label_2_semantic_dict,
    #
    "scene_line_2d_points": scene_line_2d_points,
    "scene_line_2d_end_points": scene_line_2d_end_points,
    "scene_line_2d_semantic_labels": scene_line_2d_semantic_labels,
    "scene_line_2d_params": scene_line_2d_params,
    "scene_line_2d_match_idx": scene_line_2d_match_idx,
    #
    "scene_line_3d_params": scene_line_3d_params,
    "scene_line_3d_end_points": scene_line_3d_end_points,
    "scene_line_3d_image_source": scene_line_3d_image_source, # image id
    "scene_line_3d_semantic_labels": scene_line_3d_semantic_labels,
    #
    "scene_proj_error_r_raw": scene_proj_error_r_raw,
    "scene_proj_error_t_raw": scene_proj_error_t_raw,
    #
    "params_2d": helper.params_2d
})
print("Save data successfully.")

### save all regressed 3d lines for visualization
point_sets=[]
for i in range(len(scene_line_3d_params)):
    end_points = scene_line_3d_end_points[i]
    p, v = scene_line_3d_params[i]
    sig_dim = np.argmax(abs(v))
    min_d = end_points[0][sig_dim]
    max_d = end_points[1][sig_dim]
    # get 300 points between min_d and max_d with equal interval
    sample_d = np.linspace(min_d, max_d, 300)
    for d in sample_d:
        point_sets.append(p + (d - p[sig_dim]) / v[sig_dim] * v)
point_sets = np.vstack(point_sets)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_sets)
o3d.io.write_point_cloud(line_mesh_raw_folder + f"raw_3d_line_mesh.ply", pcd)
print("Save raw 3D line mesh successfully.")
print("Process completed.")



