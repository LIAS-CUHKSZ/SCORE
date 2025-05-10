"""
helper.py 

This module provides functions and parameters for line_extractor_pt1 and pt2. 

Author: Haodong JIANG <221049033@link.cuhk.edu.cn>
Version: 1.0
License: MIT
"""
import pyelsed
import numpy as np
from sklearn.cluster import KMeans
# parameters for line_extractor_pt1
params_2d = {
    # for parallel computing
    "thread_number": 32,                                 
    # for 2d line extractor
    "sigma": 1,
    "gradientThreshold": 45,                 # tune this to filter weak 2D lines
    "minLineLen": 150,                       # tune this to filer short 2D lines            
    "lineFitErrThreshold": 0.2,
    "pxToSegmentDistTh": 1.5,
    "validationTh": 0.15,
    "validate": True,
    "treatJunctions": True,
    # for 2d line merging
    "pix_dis_thresh": 20,                    # tune this to filter nearby 2D lines
    "parallel_thres_2d": np.cos(3*np.pi/180), 
    # for 3d line regression
    "background_depth_diff_thresh": 0.3,     # tune this to threshold the depth leap between fore- and back-ground points 
    "line_points_num_thresh": 30,            # tune this to threshold the minimal number required to regress a reliable 3D line
    "perturb_length": 16,                    # tune this to adjust the perturbation 
     "num_hypo":8                            # tune this to adjust the hypothesis number
}

# parameters for line_extractor_pt2
params_3d = {
    # for parallel computing
    "thread_number": 32,                                 
    # for 3d line merging
    "parrallel_thresh_3d":np.cos(1.5*np.pi/180), # tune this
    "overlap_thresh_3d": 0.015,                  # tune this
    # for 3d line pruning
    "degree_threshold": 0,                       # tune this
}

def get_line_eq(x0, y0, x1, y1):
    # derive 2d line paramaters from two endpoints
    m = (y1 - y0) / (x1 - x0)
    c = y0 - m * x0
    return m, c

def extract_and_prune_2dlines(rgb):
    # extract 2d lines using ELSED
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
    # prune redundant 2d lines
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
                # uncomment this if you want to keep disconnected segments on a same line
                # if sig_dim == 0:
                #     proximate_line_flag = proximate_line_flag and not (x3 > x2 or x4 < x1)
                # else:
                #     proximate_line_flag = proximate_line_flag and not (min(y3, y4) > max(y1, y2) or max(y3, y4) < min(y1, y2))
                if proximate_line_flag:  # proximate parallel lines
                    segments = np.delete(segments, k, 0)
                    scores = np.delete(scores, k, 0)
                else:  
                    k += 1
            else:
                k += 1
        j += 1 
        # draw 2d lines and output images
    return segments

def get_foreground_points(valid_z):
    # Perform KMeans clustering for depth to distinguish fore- and back-ground
    kmeans = KMeans(n_clusters=2, random_state=0).fit(valid_z.reshape(-1, 1))
    actual_clusters = len(np.unique(kmeans.labels_))
    # special case
    if actual_clusters == 1:
        # there is only one depth cluster 
        return list(range(0, len(valid_z))), np.mean(valid_z), False
    # 
    depth_cluster_0 = valid_z[np.where(kmeans.labels_ == 0)]
    depth_cluster_1 = valid_z[np.where(kmeans.labels_ == 1)]
    foreground_idx = range(0, len(valid_z))
    centers = kmeans.cluster_centers_
    depth_mean = np.mean(valid_z) 
    background_flag=False
    if len(np.unique(kmeans.labels_))==1:
        # there is only one depth cluster 
        return foreground_idx,depth_mean,background_flag   
    if centers[0]>centers[1] and min(depth_cluster_0)-max(depth_cluster_1) > params_2d["background_depth_diff_thresh"]:
        # there is a depth leap between two clusters
        depth_mean=centers[1][0]
        foreground_idx = np.where(kmeans.labels_ == 1)[0]
        background_flag=True
    if  centers[1]>centers[0] and min(depth_cluster_1)-max(depth_cluster_0) > params_2d["background_depth_diff_thresh"]:
        # there is a depth leap between two clusters
        depth_mean=centers[0][0]
        foreground_idx = np.where(kmeans.labels_ == 0)[0]
        background_flag=True
    return foreground_idx,depth_mean,background_flag

def perturb_and_extract(x,y,render_depth,v,num_hypo):
    # perturb the 2D line and extract different 3D points
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
    # calculate the geometric error using pose data
    n_j_camera = n_j_pixel @ intrinsic_matrix
    n_j_camera = n_j_camera / np.linalg.norm(n_j_camera)
    n_j_camera = n_j_camera.reshape(3,1)
    error_rot = (pose_matrix[:3, :3] @ n_j_camera).T @ v
    error_trans = (p.T - np.array(pose_matrix)[:3, 3]) @ (pose_matrix[:3, :3] @ n_j_camera)
    return error_rot, error_trans