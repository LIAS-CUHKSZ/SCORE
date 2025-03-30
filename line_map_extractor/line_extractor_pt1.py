import pyelsed
import cv2
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R
import os
import matplotlib.pyplot as plt
import json
import open3d as o3d
from skimage.measure import LineModelND, ransac
from sklearn.cluster import KMeans
# from collections import Counter
from joblib import Parallel, delayed
from scipy import stats
# parameters

params_2d = {
    # for parallel computing
    "thread_number": 16,                                 
    # for 2d line extractor
    "sigma": 1,
    "gradientThreshold": 20,
    "minLineLen": 150,                       # tune this            
    "lineFitErrThreshold": 0.2,
    "pxToSegmentDistTh": 1.5,
    "validationTh": 0.15,
    "validate": True,
    "treatJunctions": True,
    # for 2d line merging
    "pix_dis_thresh": 10,                    # tune this         
    "parallel_thres_2d": np.cos(3*np.pi/180), 
    # for 3d line regression
    "background_depth_diff_thresh": 0.3,     # tune this
    "line_points_num_thresh": 30,          # tune this 
    "perturb_length": 16,                    # tune this
     "num_hypo":8,                          # tune this
}
# scene_id = "55b2bf8036"
# scene_id = "689fec23d7"
scene_id = "69e5939669"
# scene_id = "c173f62b15"
init_data_folder = f"/data1/home/lucky/ELSED/dataset/data_selected/{scene_id}/iphone/rgb_used_clear/"
mesh_file = (
    f"/data1/home/lucky/ELSED/dataset/data_selected/{scene_id}/scans/mesh_aligned_0.05.ply"
)
render_image = f"/data1/home/lucky/ELSED/dataset/data_selected/{scene_id}/iphone/render_depth"
render_depth_list = sorted(glob.glob(render_image + "/*.png"))
# downsample the list to half the size
# render_depth_list = render_depth_list[::2]
origin_img_list = sorted(glob.glob(init_data_folder + "*.jpg"))
k = 0
while(k<len(render_depth_list)):
    render_depth_file = render_depth_list[k]
    rgb_file = render_depth_file.replace("render_depth", "rgb_used_clear").replace("png", "jpg")
    if rgb_file not in origin_img_list:
        render_depth_list.remove(render_depth_file)
    else:
        k=k+1
# load pose, camera to world
pose_file = (
    f"/data1/home/lucky/ELSED/dataset/data_selected/{scene_id}/iphone/pose_intrinsic_imu.json"
)
anno_file = f"/data1/home/lucky/ELSED/dataset/data_selected/{scene_id}/scans/segments_anno.json"
segments_file = f"/data1/home/lucky/ELSED/dataset/data_selected/{scene_id}/scans/segments.json"
instance_path = f"/data1/home/lucky/ELSED/dataset/selected_semantic_2d_iphone/obj_ids/{scene_id}/"
label_filter_file = f"/data1/home/lucky/ELSED/dataset/data_selected/{scene_id}/label_remapping.txt"
### result saving path
line_image_folder = (
    f"/data1/home/lucky/ELSED/dataset/data_jhd/{scene_id}/iphone/rgb_line_image/"
)
if not os.path.exists(line_image_folder):
    os.makedirs(line_image_folder)
# unmerged 3d lines
line_mesh_raw_folder = (
    f"/data1/home/lucky/ELSED/dataset/data_jhd/{scene_id}/iphone/line_mesh_raw/"
)
if not os.path.exists(line_mesh_raw_folder):
    os.makedirs(line_mesh_raw_folder)
with open(pose_file, "r") as f:
    pose_data = json.load(f)
# load semantics for each object
with open(anno_file, "r") as f:
    anno = json.load(f)
# load mesh
mesh = o3d.io.read_triangle_mesh(mesh_file)
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
# for v,k in label_2_semantic_dict.items():
        # print(v,k)
# filter the unwanted labels
label_filter_file = open(label_filter_file, "r")
label_filtering = np.array(range(len(label_2_semantic_dict)))+1
for line in label_filter_file:
    line = line.strip()
    # get the two number seperated by comma:
    line = line.split(",")
    label_1 = int(line[0])
    label_2 = int(line[1])
    label_filtering[label_1-1] = label_2
new_label_2_semantic_dict = {}
print("adopted semantic labels")
for v,k in label_2_semantic_dict.items():
    if label_filtering[int(v)-1] == int(v):
        new_label_2_semantic_dict[v] = k
        print(v,k)
print("deleted semantic labels")
for v,k in label_2_semantic_dict.items():
    if label_filtering[int(v)-1] != int(v):
        print(v,k)

#### extract 2d lines and regress 3d lines
# Initialize variables
scene_pose = {}
scene_intrinsic = {}
### We store 2d lines from different images separately  
scene_line_2d_points = {}
scene_line_2d_end_points = {}
scene_line_2d_semantic_labels = {}
scene_line_2d_params = {}
scene_line_2d_match_idx = {}
scene_proj_error_r_raw = {}
scene_proj_error_t_raw = {}
### We gather all 3d lines 
scene_line_3d_semantic_labels = []
scene_line_3d_params = []
scene_line_3d_end_points = []
scene_line_3d_image_source = []
def get_line_eq(x0, y0, x1, y1):
    m = (y1 - y0) / (x1 - x0)
    c = y0 - m * x0
    return m, c

def extract_and_prune_2dlines(rgb):
    # get 2d lines
    segments, scores = pyelsed.detect(
        rgb,
        sigma=params_2d["sigma"],
        gradientThreshold=params_2d["gradientThreshold"],
        minLineLen=params_2d["minLineLen"],
        lineFitErrThreshold=params_2d["lineFitErrThreshold"],
        pxToSegmentDistTh=params_2d["pxToSegmentDistTh"],
        validationTh=params_2d["validationTh"],
        validate=params_2d["validate"],
        treatJunctions=params_2d["treatJunctions"],
    )
    # regulate data format
    for j, segment in enumerate(segments):
        x1, y1, x2, y2 = segment
        if x2 < x1:
            segments[j] = x2, y2, x1, y1
    # prune 2d lines
    sorted_index = np.argsort(scores)[::-1]
    segments = segments[sorted_index]
    scores = scores[sorted_index]
    j = 0
    while j < len(segments):
        x1, y1, x2, y2 = segments[j]
        vj = np.array([x2 - x1, y2 - y1]).reshape(1,2)
        vj = vj / np.linalg.norm(vj)
        project_matrix = np.eye(2) - vj.T @ vj
        k = j + 1
        while k < len(segments):
            x3, y3, x4, y4 = segments[k]
            vk = np.array([x4 - x3, y4 - y3]).reshape(1,2)
            vk = vk / np.linalg.norm(vk)
            if abs(vj @ vk.T) > params_2d["parallel_thres_2d"]:
                pixel_diff = np.array([np.array([x1 - x3, y1 - y3]), np.array([x1 - x4, y1 - y4]), np.array([x2 - x3, y2 - y3]), np.array([x2 - x4, y2 - y4])])
                project_diff = project_matrix @ pixel_diff.T
                project_dis = np.linalg.norm(project_diff, axis=0)
                proximate_line_flag = min(project_dis) < params_2d["pix_dis_thresh"]
                sig_dim = np.argmax(abs(vj))
                if sig_dim == 0:
                    proximate_line_flag = proximate_line_flag and not (x3 > x2 or x4 < x1)
                else:
                    proximate_line_flag = proximate_line_flag and not (min(y3, y4) > max(y1, y2) or max(y3, y4) < min(y1, y2))
                if proximate_line_flag:  # proximate parallel lines
                    segments = np.delete(segments, k, 0)
                    scores = np.delete(scores, k, 0)
                else:  # disconnected segments on a same line
                    k += 1
            else:
                k += 1
        j += 1 
        # draw 2d lines and output images
    return segments
    
def get_foreground_points(valid_z):
    # Perform KMeans clustering for depth to distinguish fore- and back-ground
    kmeans = KMeans(n_clusters=2, random_state=0).fit(valid_z.reshape(-1, 1))
    depth_cluster_0 = valid_z[np.where(kmeans.labels_ == 0)]
    depth_cluster_1 = valid_z[np.where(kmeans.labels_ == 1)]
    foreground_idx = range(0, len(valid_z))
    centers = kmeans.cluster_centers_
    depth_mean = np.mean(valid_z) 
    background_flag=False
    if len(np.unique(kmeans.labels_))==1: # there is only one depth cluster
        return foreground_idx,depth_mean,background_flag   
    if centers[0]>centers[1] and min(depth_cluster_0)-max(depth_cluster_1) > params_2d["background_depth_diff_thresh"]:
        depth_mean=centers[1][0]
        foreground_idx = np.where(kmeans.labels_ == 1)[0]
        background_flag=True
    if  centers[1]>centers[0] and min(depth_cluster_1)-max(depth_cluster_0) > params_2d["background_depth_diff_thresh"]:
        depth_mean=centers[0][0]
        foreground_idx = np.where(kmeans.labels_ == 0)[0]
        background_flag=True
    return foreground_idx,depth_mean,background_flag

def perturb_and_extract(x,y,render_depth,v,num_hypo):
    move_length_list = np.linspace(-1,1,num_hypo)*params_2d["perturb_length"]
    background_flag = []
    depth_mean = []
    xyz_list = []
    for i in range(num_hypo):
        background_flag.append(False)
        depth_mean.append(255)
    foreground_idices = []
    for k in range(len(move_length_list)):
        mov_length = move_length_list[k]
        ### perturbed lines
        x_perturbed = x + int(mov_length*v[1]) 
        y_perturbed = y + int(-mov_length*v[0])
        ### shrink the perturbed line a little bit
        shrinked_points = int(len(x_perturbed)*0.05)
        x_perturbed = x_perturbed[shrinked_points:-shrinked_points]
        y_perturbed = y_perturbed[shrinked_points:-shrinked_points]
        ###
        valid_idx = np.logical_and(x_perturbed<render_depth.shape[1],y_perturbed<render_depth.shape[0])
        valid_idx_ = np.logical_and(valid_idx,x_perturbed>=0,y_perturbed>=0)
        x_perturbed = x_perturbed[valid_idx_]
        y_perturbed = y_perturbed[valid_idx_]    
        ###    
        z_perturbed = render_depth[y_perturbed, x_perturbed]
        valid_z = z_perturbed[np.where(z_perturbed != 0)]
        valid_x = x_perturbed[np.where(z_perturbed != 0)]
        valid_y = y_perturbed[np.where(z_perturbed != 0)]
        xyz_list.append(np.concatenate([valid_x[:, None], valid_y[:, None], valid_z[:, None]], axis=1))
        foreground_idx = list(range(len(valid_z)))
        if len(valid_z)>params_2d["line_points_num_thresh"]:
            foreground_idx,depth_mean[k],background_flag[k] = get_foreground_points(valid_z)
        foreground_idices.append(foreground_idx)
    return depth_mean,xyz_list,foreground_idices,background_flag

def calculate_error(n_j_pixel,v,intrinsic_matrix,pose_matrix,p):
    n_j_camera = n_j_pixel @ intrinsic_matrix
    n_j_camera = n_j_camera / np.linalg.norm(n_j_camera)
    n_j_camera = n_j_camera.reshape(3,1)
    error_rot = (pose_matrix[:3, :3] @ n_j_camera).T @ v
    error_trans = (p.T - np.array(pose_matrix)[:3, 3]) @ (pose_matrix[:3, :3] @ n_j_camera)
    return error_rot, error_trans

def process_file(i, render_depth_file):
    basename = os.path.basename(render_depth_file).split(".")[0]
    intrinsic = pose_data[basename]["intrinsic"]
    pose_matrix = np.array(pose_data[basename]["aligned_pose"])
    ###
    print(f"Processing {basename}")
    line_2d_end_points = []
    line_2d_points=[]
    line_2d_params = []
    line_2d_semantic_label = []
    line_2d_match_idx = []
    ###
    line_3d_params = []
    line_3d_end_points = []
    line_3d_semantic_label = []
    ###
    proj_error_r = []
    proj_error_t = []
    # millimeter to meter
    render_depth = (
        cv2.imread(render_depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
    )
    # read rgb
    rgb_file = render_depth_file.replace("render_depth", "rgb").replace("png", "jpg")
    obj_id_file = os.path.join(instance_path, str(os.path.basename(render_depth_file)).replace(".png", ".jpg.npy"))
    obj_ids = np.load(obj_id_file)
    rgb = cv2.imread(rgb_file, cv2.IMREAD_GRAYSCALE)
    rgb_color = cv2.imread(rgb_file)
    # Extract and Prune 2d Lines
    segments = extract_and_prune_2dlines(rgb)
    ######################## Regress 3d Lines #######################
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
            m, c = get_line_eq(x1, y1, x2, y2)  # y = mx + c
            A,B,C = m,-1,c
            if abs(m) > 1:  # sample points on the longer axis
                y = np.arange(min(y1, y2), max(y1, y2))
                x = (y - c) / m
            else:
                x = np.arange(min(x1, x2), max(x1, x2))
                y = m * x + c
        x = x.astype(np.int32) # since we only have depth on integer pixels
        y = y.astype(np.int32)
        line_2d_param_pixel = np.array([A, B, C])
        ### 
        v = np.array([x2-x1,y2-y1])
        if np.linalg.norm(v)==0:
            cv2.line(rgb_color, (x1, y1), (x2, y2), (0, 0, 0), 2)
            print(v)   
            continue
        v = v/np.linalg.norm(v)
        num_hypo=params_2d["num_hypo"]
        depth_mean,xyz_list,foreground_idices,background_flag = perturb_and_extract(x,y,render_depth,v,num_hypo*2+1)
        best_one = np.argmin(depth_mean)
        while (best_one < num_hypo):
            if depth_mean[best_one+1]-depth_mean[best_one] < params_2d["background_depth_diff_thresh"]:
                best_one = best_one+1
            else:
                break
        while (best_one>num_hypo):
            if depth_mean[best_one-1]-depth_mean[best_one] < params_2d["background_depth_diff_thresh"]:
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
        # get the (dominant) semantic id for this 2d line
        all_points_obj_ids = obj_ids[foreground_y, foreground_x]
        all_points_semantic_label = []
        for point_obj_id in all_points_obj_ids:
            all_points_semantic_label.append(obj_id_label_id[point_obj_id])
        all_points_semantic_label = np.array(all_points_semantic_label)
        unique_labels, counts = np.unique(all_points_semantic_label, return_counts=True)
        frequent_labels = unique_labels[counts > params_2d["line_points_num_thresh"]]
        semantic_label = 0
        for label in frequent_labels:
            if label == 0:
                continue
            if label_filtering[label-1] != 0:
                semantic_label = label_filtering[label-1]
                break
        if semantic_label == 0:
            continue  
        else:
            line_2d_points.append([x,y])
            cv2.line(rgb_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rgb_color, str(line_2d_count), (int((x1 + x2) / 2), int((y1 + y2) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.putText(rgb_color, label_2_semantic_dict[semantic_label], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
            if background_flag[best_one]:
                cv2.line(rgb_color, (x1, y1), (x2, y2), (0, 0, 255), 2)
            line_2d_params.append(line_2d_param_pixel)
            line_2d_end_points.append([[x1,y1],[x2,y2]])
            line_2d_semantic_label.append(semantic_label)
            line_2d_count+=1
        #######################################################
        points_2d = np.concatenate([foreground_x[:, None], foreground_y[:, None], np.ones_like(foreground_x)[:, None]], axis=1)
        points_camera_3d = (np.linalg.inv(intrinsic) @ (points_2d * foreground_z[:, None]).T).T
        points_world_3d = (
            pose_matrix @ np.concatenate([points_camera_3d, np.ones((points_camera_3d.shape[0], 1))], axis=1).T
        )
        points_world_3d = points_world_3d[:3, :].T
        try:
            model_robust, inliers = ransac(
                points_world_3d, LineModelND, min_samples=3, residual_threshold=0.02, max_trials=3000
            )
        except:
            print("extrct 3d line fails")
            cv2.line(rgb_color, (x1, y1), (int((x1+x2)/2), int((y1+y2)/2)), (255, 0, 0), 2)
            error_rot   = -1
            error_trans = -1
            continue
        inlier_index = inliers == True
        ### if the number of inliers is less than a threshold, we discard this 3d line
        if len(inlier_index) < params_2d["line_points_num_thresh"]:
            print("extrct 3d line fails")
            cv2.line(rgb_color, (x1, y1), (int((x1+x2)/2), int((y1+y2)/2)), (255, 0, 0), 2)
            error_rot   = -1
            error_trans = -1
            continue
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
        ### projection error
        error_rot,error_trans = calculate_error(line_2d_param_pixel.reshape(1,3),v,intrinsic,pose_matrix,p)
        if error_rot > 0.1 or error_trans>0.1:
            print(basename,j,error_rot,error_trans)
        proj_error_r.append(np.abs(error_rot))
        proj_error_t.append(np.abs(error_trans))
    # draw 2d lines and output images
    cv2.imwrite(os.path.join(line_image_folder, f"{basename}.jpg"), rgb_color)    
    return (basename, line_2d_points,line_2d_end_points, line_2d_params,line_2d_semantic_label, line_2d_match_idx, line_3d_semantic_label, line_3d_params, line_3d_end_points,proj_error_r,proj_error_t)
results=[]
# for i, render_depth_file in enumerate(render_depth_list):
#     if i > -1:
#         result = process_file(i,render_depth_file)
#         results.append(result)
# Use parallel processing to process files
results = Parallel(n_jobs=params_2d["thread_number"])(delayed(process_file)(i, render_depth_file) for i, render_depth_file in enumerate(render_depth_list))
for result in results:
    basename, line_2d_points,line_2d_end_points, line_2d_params,line_2d_semantic_label, line_2d_match_idx, line_3d_semantic_label, line_3d_params, line_3d_end_points,proj_error_r,proj_error_t = result
    ###
    scene_line_2d_points[basename] = line_2d_points
    scene_line_2d_end_points[basename] = line_2d_end_points
    scene_line_2d_params[basename] = line_2d_params
    scene_line_2d_semantic_labels[basename] = line_2d_semantic_label
    for i in range(len(line_2d_match_idx)):
        if line_2d_match_idx[i] != None:
            line_2d_match_idx[i] = line_2d_match_idx[i]+len(scene_line_3d_params)
    scene_line_2d_match_idx[basename] = np.array(line_2d_match_idx)
    ###
    scene_line_3d_semantic_labels.extend(line_3d_semantic_label)
    scene_line_3d_params.extend(line_3d_params)
    scene_line_3d_end_points.extend(line_3d_end_points)
    image_index = np.int32(basename[-6:])
    scene_line_3d_image_source.extend(np.ones(len(line_3d_params))*image_index)
    ###
    scene_proj_error_r_raw[basename]=proj_error_r
    scene_proj_error_t_raw[basename]=proj_error_t

for i in range(0,len(render_depth_list)):
    basename = os.path.basename(render_depth_list[i]).split(".")[0]
    intrinsic = pose_data[basename]["intrinsic"]
    scene_pose[basename] = pose_data[basename]["aligned_pose"]
    scene_intrinsic[basename] = intrinsic

np.save(f"/data1/home/lucky/ELSED/code_iphone/result/{scene_id}_results_raw.npy", {
    "scene_pose": scene_pose,
    "scene_intrinsic": scene_intrinsic,
    "obj_id_label_id": obj_id_label_id,
    "label_2_semantic_dict": new_label_2_semantic_dict,
    ###
    "scene_line_2d_points": scene_line_2d_points,
    "scene_line_2d_end_points": scene_line_2d_end_points,
    "scene_line_2d_semantic_labels": scene_line_2d_semantic_labels,
    "scene_line_2d_params": scene_line_2d_params,
    "scene_line_2d_match_idx": scene_line_2d_match_idx,
    ###
    "scene_line_3d_params": scene_line_3d_params,
    "scene_line_3d_end_points": scene_line_3d_end_points,
    "scene_line_3d_image_source": scene_line_3d_image_source, # image id
    "scene_line_3d_semantic_labels": scene_line_3d_semantic_labels,
    ###
    "scene_proj_error_r_raw": scene_proj_error_r_raw,
    "scene_proj_error_t_raw": scene_proj_error_t_raw,
    ###
    "params_2d": params_2d
})

### save 3d lines before merging
point_sets=[]
for i in range(len(scene_line_3d_params)):
    end_points = scene_line_3d_end_points[i]
    p, v = scene_line_3d_params[i]
    sig_dim = np.argmax(abs(v))
    min_d = end_points[0][sig_dim]
    max_d = end_points[1][sig_dim]
    # get 100 points between min_d and max_d with equal interval
    sample_d = np.linspace(min_d, max_d, 300)
    for d in sample_d:
        point_sets.append(p + (d - p[sig_dim]) / v[sig_dim] * v)
point_sets = np.vstack(point_sets)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_sets)
o3d.io.write_point_cloud(line_mesh_raw_folder + f"raw_3d_line_mesh.ply", pcd)
print("write mesh successfully")


