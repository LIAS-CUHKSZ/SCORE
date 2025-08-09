  ### 0. Download
  - You can download scanNet++ dataset from the official website: [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/). 
  - Note that the scannet++ data is very large, make sure you have enough space.
  
  ### 1. Use our revised python code to render pose, depth, and semantic labels
-   You can create a new environment to handle data related to scannet++.
    ```bash
    conda create -n scannet python=3.10
    conda activate scannet
    cd scannetpp
    pip install -r requirements.txt
    ```

-   In our experiments, we use the following command to render the data (only iphone data is used in our experiments):
    ```bash
    # we have combined all parameters in one config file `merged_config.yaml`
    # please replace the `scene_ids`,`filter_scenes`, and input/output folders in the yaml file with your own path
    cd SCORE/scannetpp

    # extract depth for iphone images
    python -m common.render merged_config.yaml

    # extract iphone data
    # this code will extract the rgb, depth and mask data from the original scannet++ dataset
    python -m iphone.prepare_iphone_data merged_config.yaml

    # rasterize the mesh onto iPhone images and save the 2D-3D mappings (pixel-to-face) to file. 
    # this will use GPU to rasterize the mesh onto the images, if your GPU memory is not enough, you can set `batch_size` to a smaller number
    python -m semantic.prep.rasterize

    # get the object id on the 2D image
    # the obj id of each image will be saved tin `$save_dir_root$/$save_dir$/obj_ids/$scene_id$`, you can use numpy to read it
    # in extract 3d semantic line map code, we will convert object ID to label ID.
    python -m semantic.prep.semantics_2d