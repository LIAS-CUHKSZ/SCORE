# SCORE:Saturated Consensus Relocalization in Semantic Line Maps
This is the github repository for our IROS2025 paper(submitted).
## 1. Matlab Implementation for FGO-PnL solver and Examples

## 2. Cpp parallel Implementation for FGO-PnL solver and Examples
Provide the implementation of algorithm in CPP. We will add parallel computing soon.
You can use the following commands to compile and run.
```shell
cd ./cpp
mkdir build
cd build
cmake ..
./test
```
## 3. Go through the relocalization pipeline on ScanNet++ Dataset
### Download ScanNet++ Dataset
You can download scanNet++ dataset from the official website: [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/). Please note that the scannet++ data is very large, so make sure you have enough space to download it.
### Use our revised python code to render pose, depth, and semantic labels
You can create a new environment to handle data related to scannet++.
```bash
conda create -n scannet python=3.10
conda activate scannet
cd scannetpp
pip install -r requirements.txt
```

In our experiments, we use the following command to render the data (only iphone data is used in our experiments):
```bash
# extract iphone data
# please replace the `data_root` and `scene_ids` in `iphone/configs/prepare_iphone_data.yml` with your own path
# this code will extract the rgb, depth and mask data from the original scannet++ dataset
python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml

# rasterize the mesh onto iPhone images and save the 2D-3D mappings (pixel-to-face) to file. 
# please replace the `data_root`, `scene_list`, `filter_scenes` and `rasterout_dir` in `semantic/configs/rasterize.yaml` with your own path
# this will use GPU to rasterize the mesh onto the images, if your GPU memory is not enough, you can set `batch_size` to a smaller number
python -m semantic.prep.rasterize

# get the object id on the 2D image
# please replace the `data_root`, `scene_list`, `filter_scenes`, `rasterout_dir`, `visiblity_cache_dir`, `save_dir_root` and `save_dir` in `semantic/configs/semantic_2d.yaml` with your own path
# the obj id of each image will be saved tin `$save_dir_root$/$save_dir$/obj_ids/$scene_id$`, you can use numpy to read it
# in extract 3d semantic line map code, we will convert object ID to label ID.
python -m semantic.prep.semantics_2d
```
NOTE: extract semantic labels will take a long time.

### Extract 3D Semantic Line Maps
### Semantic Segmentation Pipeline combing RAM++ and Grounded-SAM 

## 4. Techinical Report for Interval Analysis
Uploaded on Arxiv:
