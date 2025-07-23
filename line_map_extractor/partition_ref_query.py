import os
import glob

if __name__ == "__main__":
    data_root_dir = "/data2/scannetppv2/"
    output_root_dir = "/data1/home/lucky/IROS25/SCORE/"
    scene_list = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"]
    for scene_id in scene_list:
        colmap_image_file = os.path.join(data_root_dir, f"data/{scene_id}/iphone/colmap/images.txt")
        colmap_image_list = []
        with open(colmap_image_file, 'r') as f:
            # Keep header lines
            for i in range(4):
                f.readline()
            # Process image entries (2 lines per image)
            while True:
                line1 = f.readline()
                line2 = f.readline() 
                if not line1 or not line2:
                    break
                image_name = line1.strip().split()[-1]
                colmap_image_list.append(image_name)
        
        output_dir = os.path.join(output_root_dir, f"line_map_extractor/out/{scene_id}/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        count = 0
        query_file = os.path.join(output_dir, "query.txt")
        ref_file = os.path.join(output_dir, "ref.txt")
        # clear the text files
        with open(query_file, "w") as f:
            pass
        with open(ref_file, "w") as f:
            pass
        for colmap_image_name in colmap_image_list:
            count += 1
            base_name = os.path.basename(colmap_image_name)
            # for every eight images, put the name of first seven in a text file, and the last one in another text file
            if count % 8 == 0:   
                with open(query_file, "a") as f: 
                        f.write(base_name + "\n")
            else:
                with open(ref_file, "a") as f: 
                    f.write(base_name + "\n")