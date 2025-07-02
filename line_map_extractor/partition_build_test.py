import os
import glob

if __name__ == "__main__":
    data_root_dir = "/data2/scannetppv2/"
    output_root_dir = "/data1/home/lucky/IROS25/SCORE/"
    scene_list = ["69e5939669","689fec23D7","c173f62b15","55b2bf8036"]
    scene_id = scene_list[0]
    rgb_dir = os.path.join(data_root_dir, f"data/{scene_id}/iphone/rgb/kept_images/")
    output_dir = os.path.join(output_root_dir, f"line_map_extractor/out/{scene_id}/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    rgb_img_list = sorted(glob.glob(rgb_dir + "*.jpg"))
    count = 0
    test_file = os.path.join(output_dir, "test.txt")
    train_file = os.path.join(output_dir, "train.txt")
    # clear the text files
    with open(test_file, "w") as f:
        pass
    with open(train_file, "w") as f:
        pass
    for rgb_img_name in rgb_img_list:
        count += 1
        base_name = os.path.basename(rgb_img_name)
        # for every eight images, put the name of first seven in a text file, and the last one in another text file
        if count % 8 == 0:   
           with open(test_file, "a") as f: 
                f.write(base_name + "\n")
        else:
            with open(train_file, "a") as f: 
                f.write(base_name + "\n")