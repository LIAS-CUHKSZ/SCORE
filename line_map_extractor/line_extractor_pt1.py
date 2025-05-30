"""
line_extractor_pt1.py 

This script extracts 2D lines from images and regresses 3D lines from the 2D lines based on pose and depth.
We assign each pair of 2D and 3D line with the same semantic label. 
You can tune the paramters defined in helper.py, and edit the label_remapping.txt file under /dictionary.

Output:
- Rgb images annotated with extracted lines thier semantic labels. 
- Mesh file(.ply) with all regressed 3D lines for visualization.
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
from tqdm import tqdm
from skimage.measure import LineModelND, ransac
# from collections import Counter
from joblib import Parallel, delayed
from scipy import stats


################################### Loading and Configuring ###################################
data_root_dir = "/data2/scannetppv2"
output_root_dir = "/data1/home/lucky/IROS25/"
scene_list = ["69e5939669","689fec23D7","c173f62b15","55b2bf8036"]
scene_id = scene_list[2]
rgb_folder = data_root_dir+f"/data/{scene_id}/iphone/rgb/"
depth_image_folder = data_root_dir+f"/data/{scene_id}/iphone/render_depth/"
depth_img_list = sorted(glob.glob(depth_image_folder + "*.png"))
rgb_img_list = sorted(glob.glob(rgb_folder + "*.jpg"))
depth_img_list= depth_img_list[::2] #downsample
# remove the depth images that do not have corresponding rgb images
k = 0
while(k<len(depth_img_list)):
    depth_img_name = depth_img_list[k]
    rgb_file = depth_img_name.replace("render_depth", "rgb").replace("png", "jpg")
    if rgb_file not in rgb_img_list:
        depth_img_list.remove(depth_img_name)
    else:
        k=k+1
pose_file = data_root_dir+f"/data/{scene_id}/iphone/pose_intrinsic_imu.json" # camera pose to world
anno_file = data_root_dir+f"/data/{scene_id}/scans/segments_anno.json"
segments_file = data_root_dir+f"/data/{scene_id}/scans/segments.json"
instance_path = data_root_dir+f"/semantic_2D_iphone/obj_ids/{scene_id}/"
### result saving path
dictionary_folder = output_root_dir+f"SCORE/dictionary/{scene_id}/"
line_image_folder = output_root_dir+f"SCORE/line_map_extractor/out/{scene_id}/rgb_line_image/"   # rgb images with extracted 2D lines and labels
line_mesh_raw_folder = output_root_dir+f"SCORE/line_map_extractor/out/{scene_id}/line_mesh_raw/" # regressed 3D lines
line_data_folder = output_root_dir+f"SCORE/line_map_extractor/out/{scene_id}/" # numpy file with all the extracted 2D lines and regressed 3D lines
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

### function for processing a single image     
def process_file(depth_img_name):
    basename = os.path.basename(depth_img_name).split(".")[0]
    intrinsic = pose_data[basename]["intrinsic"]
    pose_matrix = np.array(pose_data[basename]["aligned_pose"])
    #
    print(f"Processing {basename}")
    line_2D_end_points = []
    line_2D_points=[]
    line_2D_params = []
    line_2D_semantic_label = []
    line_2D_match_idx = []
    #
    line_3D_end_points = []
    line_3D_semantic_label = []
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
    ### Extract and Prune 2D Line segments
    segments = helper.extract_and_prune_2Dlines(rgb) 
    ### Regress 3D Lines
    line_2D_count = 0 # valid 2D line id in the cur image
    for j, segment in enumerate(segments):
        x1, y1, x2, y2 = segment.astype(np.int32)
        if x2 >= render_depth.shape[1] or y1 >= render_depth.shape[0] or y2 >= render_depth.shape[0]:
            continue
        # get all pixels on the line
        if x1 == x2: # special case for a vertical line
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
        x = x.astype(np.int32) # since scannet++ only provides depth on integer pixels
        y = y.astype(np.int32) # since scannet++ only provides depth on integer pixels
        v = np.array([x2-x1,y2-y1])
        if np.linalg.norm(v)==0:
            continue
        v = v/np.linalg.norm(v)
        ### Get the foregound points by multi-hypothesis pertubation
        depth_mean,xyz_list,foreground_idices,background_flag = helper.perturb_and_extract(x,y,render_depth,v,helper.params_2D["num_hypo"]*2+1)
        if np.min(depth_mean) == 255: # no valid points
            continue
        best_one = np.argmin(depth_mean)
        foreground_x = xyz_list[best_one][foreground_idices[best_one],0].astype(np.int32)
        foreground_y = xyz_list[best_one][foreground_idices[best_one],1].astype(np.int32)
        semantic_label = helper.extract_dominant_label(foreground_y,foreground_x,obj_ids,obj_id_label_id)
        while (best_one < helper.params_2D["num_hypo"]):
            if depth_mean[best_one+1]-depth_mean[best_one] > helper.params_2D["background_depth_diff_thresh"]:
                break
            foreground_x = xyz_list[best_one+1][foreground_idices[best_one+1],0].astype(np.int32)
            foreground_y = xyz_list[best_one+1][foreground_idices[best_one+1],1].astype(np.int32)
            cur_semantic_label = helper.extract_dominant_label(foreground_y,foreground_x,obj_ids,obj_id_label_id)
            if cur_semantic_label == semantic_label: # no valid label for this line
                best_one = best_one+1
            else:
                break
        while (best_one>helper.params_2D["num_hypo"]):
            if depth_mean[best_one-1]-depth_mean[best_one] > helper.params_2D["background_depth_diff_thresh"]:
                break
            foreground_x = xyz_list[best_one-1][foreground_idices[best_one-1],0].astype(np.int32)
            foreground_y = xyz_list[best_one-1][foreground_idices[best_one-1],1].astype(np.int32)
            cur_semantic_label = helper.extract_dominant_label(foreground_y,foreground_x,obj_ids,obj_id_label_id)
            if cur_semantic_label == semantic_label: # no valid label for this line
                best_one = best_one-1
            else:
                break
 
        foreground_x = xyz_list[best_one][foreground_idices[best_one],0].astype(np.int32)
        foreground_y = xyz_list[best_one][foreground_idices[best_one],1].astype(np.int32)
        foreground_z = xyz_list[best_one][foreground_idices[best_one],2]
        semantic_label = helper.extract_dominant_label(foreground_y,foreground_x,obj_ids,obj_id_label_id)
        if semantic_label == 0 or label_remapped[semantic_label-1]==0: # no valid label for this line
            continue
        semantic_label = label_remapped[semantic_label-1] 
        ### regress the 3D line with found foreground points
        points_2D = np.concatenate([foreground_x[:, None], foreground_y[:, None], np.ones_like(foreground_x)[:, None]], axis=1)
        points_camera_3D = (np.linalg.inv(intrinsic) @ (points_2D * foreground_z[:, None]).T).T # get 3D points in the camera frame
        # transform to the world frame
        points_world_3D = (
            pose_matrix @ np.concatenate([points_camera_3D, np.ones((points_camera_3D.shape[0], 1))], axis=1).T) 
        points_world_3D = points_world_3D[:3, :].T
        try:
            model_robust, inliers = ransac(
                points_world_3D, LineModelND, min_samples=3, residual_threshold=0.02, max_trials=3000
            )
        except: # if the line regression fails, we skip this line
            continue
        # if the number of inliers is less than a threshold, we discard this 3D line
        inlier_points = points_world_3D[np.where(inliers == True)]
        if len(inlier_points) < helper.params_2D["line_points_num_thresh"]:
            continue
        p,v = model_robust.params[0:2]
        sig_dim = np.argmax(abs(v))
        min_val = np.min(inlier_points[:, sig_dim])
        max_val = np.max(inlier_points[:, sig_dim])
        # regulate the vector v if it is close to principle axis
        if np.abs(np.dot(v, np.array([1, 0, 0]))) > helper.params_3D["parrallel_thresh_3D"]:
            v = np.array([1, 0, 0])
        elif np.abs(np.dot(v, np.array([0, 1, 0]))) > helper.params_3D["parrallel_thresh_3D"]:
            v = np.array([0, 1, 0])
        elif np.abs(np.dot(v, np.array([0, 0, 1]))) > helper.params_3D["parrallel_thresh_3D"]:
            v = np.array([0, 0, 1])
        point_min = p + v/v[sig_dim]*(min_val-p[sig_dim]) 
        point_max = p + v/v[sig_dim]*(max_val-p[sig_dim])
        line_3D_semantic_label.append(semantic_label)
        line_3D_end_points.append([point_min,point_max])
        ###    
        x1,y1 = xyz_list[best_one][0,0:2]
        x2,y2 = xyz_list[best_one][-1,0:2]
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        if x1 == x2:
            A,B,C = 1,0,-x1
        else:
            m, c = helper.get_line_eq(x1, y1, x2, y2)  # y = mx + c
            A,B,C = m,-1,c
        line_2D_param_pixel = np.array([A, B, C])
        line_2D_params.append(line_2D_param_pixel)
        line_2D_points.append([xyz_list[best_one][:,0],xyz_list[best_one][:,1]])
        line_2D_end_points.append([[x1,y1],[x2,y2]])
        line_2D_semantic_label.append(semantic_label)
        line_2D_match_idx.append(len(line_3D_semantic_label) - 1)
        # calculate projection error based on the pose
        error_rot,error_trans = helper.calculate_error(line_2D_param_pixel.reshape(1,3),v,intrinsic,pose_matrix,point_min,point_max)
        if error_rot > 0.1 or error_trans>0.1:
            print(basename,j,error_rot,error_trans)
        proj_error_r.append(np.abs(error_rot))
        proj_error_t.append(np.abs(error_trans))
        # draw the 2D lines along with their semantic labels
        line_2D_count+=1
        cv2.line(rgb_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rgb_color, str(line_2D_count), (int((x1 + x2) / 2), int((y1 + y2) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.putText(rgb_color, label_2_semantic_dict[semantic_label], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        if background_flag[best_one]: # red color for highlighting that there are background points when extracting this line.
            cv2.line(rgb_color, (x1, y1), (x2, y2), (0, 0, 255), 2)
    ###
    if len(line_3D_semantic_label) < 5:
        return (basename, [],[],[],[],[],[],[],[],[])
    # save rgb images with 2D line and semantic annotation
    cv2.imwrite(os.path.join(line_image_folder, f"{basename}.jpg"), rgb_color)    
    return (basename, line_2D_points,line_2D_end_points, line_2D_params,line_2D_semantic_label, line_2D_match_idx, line_3D_semantic_label , line_3D_end_points,proj_error_r,proj_error_t)

################################### Processing and Saving ###################################
### Initialize data sturcture to be stored
scene_pose = {}
scene_intrinsic = {}
# Store 2D lines from different images separately  
scene_line_2D_points = {}
scene_line_2D_end_points = {}
scene_line_2D_semantic_labels = {}
scene_line_2D_params = {}
scene_line_2D_match_idx = {}
scene_proj_error_r_raw = {}
scene_proj_error_t_raw = {}
# Store all 3D lines together 
scene_line_3D_semantic_labels = []
scene_line_3D_end_points = []
scene_line_3D_image_source = []
print(helper.params_2D["background_depth_diff_thresh"])

# unparalled code to process files
results=[]
for depth_img_name  in tqdm(depth_img_list):
        result = process_file(depth_img_name)
        results.append(result)

# parallel code to process files
# results = Parallel(n_jobs=helper.params_2D["thread_number"])(delayed(process_file)(depth_img_name) for depth_img_name in depth_img_list)

# 


for result in results:
    basename, line_2D_points,line_2D_end_points, line_2D_params,line_2D_semantic_label, line_2D_match_idx, line_3D_semantic_label, line_3D_end_points,proj_error_r,proj_error_t = result
    #
    if line_2D_points == []:
        continue
    scene_line_2D_points[basename] = line_2D_points
    scene_line_2D_end_points[basename] = line_2D_end_points
    scene_line_2D_params[basename] = line_2D_params # n_c=[A B C]
    scene_line_2D_semantic_labels[basename] = line_2D_semantic_label
    for i in range(len(line_2D_match_idx)): ## idx of the matched 3D line 
            line_2D_match_idx[i] = line_2D_match_idx[i]+len(scene_line_3D_semantic_labels)
    scene_line_2D_match_idx[basename] = np.array(line_2D_match_idx)
    #
    scene_line_3D_semantic_labels.extend(line_3D_semantic_label)
    scene_line_3D_end_points.extend(line_3D_end_points)
    image_index = np.int32(basename[-6:])
    scene_line_3D_image_source.extend(np.ones(len(line_3D_semantic_label))*image_index)
    #
    scene_proj_error_r_raw[basename]=proj_error_r
    scene_proj_error_t_raw[basename]=proj_error_t

for i in range(0,len(depth_img_list)):
    basename = os.path.basename(depth_img_list[i]).split(".")[0]
    intrinsic = pose_data[basename]["intrinsic"]
    scene_pose[basename] = pose_data[basename]["aligned_pose"]
    scene_intrinsic[basename] = intrinsic
### save the results
np.save(line_data_folder+scene_id+f"_results_raw.npy", {
    "scene_pose": scene_pose,
    "scene_intrinsic": scene_intrinsic,
    "label_2_semantic_dict": new_label_2_semantic_dict,
    #
    "scene_line_2D_points": scene_line_2D_points,
    "scene_line_2D_end_points": scene_line_2D_end_points,
    "scene_line_2D_semantic_labels": scene_line_2D_semantic_labels,
    "scene_line_2D_params": scene_line_2D_params,
    "scene_line_2D_match_idx": scene_line_2D_match_idx,
    #
    "scene_line_3D_end_points": scene_line_3D_end_points,
    "scene_line_3D_image_source": scene_line_3D_image_source, # image id
    "scene_line_3D_semantic_labels": scene_line_3D_semantic_labels,
    #
    "scene_proj_error_r_raw": scene_proj_error_r_raw,
    "scene_proj_error_t_raw": scene_proj_error_t_raw,
    #
    "params_2D": helper.params_2D
})
print("Save data successfully.")

#################################################
### save all regressed 3D lines for visualization
point_sets=[]
for i in range(len(scene_line_3D_semantic_labels)):
    point_a,point_b = scene_line_3D_end_points[i][0:2]
    point_diff = point_b - point_a
    for sample in range(300):
        point_sets.append(point_a + point_diff*sample/299)
point_sets = np.vstack(point_sets)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_sets)
o3d.io.write_point_cloud(line_mesh_raw_folder + scene_id+f"_raw_3D_line_mesh.ply", pcd)
print("Save raw 3D line mesh successfully.")
print("Process completed.")