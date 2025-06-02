The code is still under construction, the dataset and code will be released in early June, stay tuned！
# SCORE:Saturated Consensus Relocalization in Semantic Line Maps
This is the github repository for our IROS2025 paper. 
The paper is under review, we provide the arxiv version here:
https://arxiv.org/pdf/2503.03254

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
```
NOTE: extract semantic labels will take a long time.

### Extract 3D Semantic Line Maps
We provide the code `line_map_extractor` to extract the 3D semantic line maps from the ScanNet++ dataset. 
As detailed in our paper, we extract 3D lines based on a posed image sequence with depth and semnatic masks. 
The procedure goes: 
1.  Extract 2D lines in the images, and assign a semantic label for each line according to the semantic mask.
2.  Regress the corresponding 3D line based on pose and depth, assign the 3D line with the same label as the 2D line.
3.  After processing all images, cluster the regressed 3D lines to reduce redundancy and gather multiple labels for a single 3D line.   

The readers will find the code comments in consistency with the description in our paper.
According to ScanNet++ term of use, we can not share the original data.
In order to facilicate a fast replication of our work, we share 
1. the adopted dictionary for each scene
2. the output 3d line mesh after running line_extractor_pt1.py
3. the output numpy file after running line_extractor_pt1.py

You can tune the dictionary by editing the output numpy file, and also use the numpy file to run line_extractor_pt2.py.
#### Install the dependencies
We use elsed as 2D line extractor, and use joblib to parallelize the extration proccess.
```bash
pip install -r line_map_extractor/requirements.txt
```
#### Procedures 1&2 by running
``` bash
# Remeber to revise variables data_root_dir and output_root_dir in the code
python line_map_extractor/line_extractor_pt1.py
```

#### Procedure 3 by running
``` bash
# Construct the consistency graph 
python line_map_extractor/line_extractor_pt2.py
# Use previously contructed consistency graph
python line_map_extractor/line_extractor_pt2.py -r y
```
### 
### Semantic Segmentation Pipeline combing RAM++ and Grounded-SAM
We provide the code `semantic_pipeline/` to get the semantic segmentation results using the RAM++ and Grounded-SAM.

#### Install the dependencies
```bash
conda create -n ram-sam python=3.10

# install pytorch and torchvision
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# install sam2
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .

# install ground-dino
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .

# install ram++
git clone https://github.com/xinyu1205/recognize-anything.git
cd recognize-anything
pip install -e .
```

#### Download the pretrained model
```bash
mkdir pretrain_models
wget https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

#### Generate the Tag Description for RAM++
```bash

# please set the azure openai config by environment variables
export OPENAI_API_KEY=YOUR_AZURE_OPENAI_KEY
export OPENAI_API_BASE=YOUR_AZURE_OPENAI_BASE
export OPENAI_DEPLOYMENT_NAME=YOUR_AZURE_OPENAI_DEPLOYMENT_NAME
export OPENAI_API_VERSION=YOUR_AZURE_OPENAI_API_VERSION

# you can reference `data/689fec23d7_dictionary.txt` for text file format
# generate the tag description for RAM++, this will output the tag description for each object
python generate_tag_des.py --text_file xxxx.txt
```

#### Generate the Semantic BBOX
```bash
# des file is the tag description file generated by the previous step
# your also need to prepare the txt file for the object id and object name mapping with the format `object_id object_name`, and its name should be `xxx_index.txt`
# please reference `data/689fec23d7_index.txt` for the format
python ram_ground_sam_openset.py --des_file xxx.json --image-dir xxx --output-dir xxx
```
## Citation
If you find our work helpful, please cite:
```
@article{jiang2025score,
  title={SCORE: Saturated Consensus Relocalization in Semantic Line Maps},
  author={Jiang, Haodong and Zheng, Xiang and Zhang, Yanglin and Zeng, Qingcheng and Li, Yiqian and Hong, Ziyang and Wu, Junfeng},
  journal={arXiv preprint arXiv:2503.03254},
  year={2025}
}
```