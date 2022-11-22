import shutil
import sys
import os
from pathlib import Path
import json
import re

movienet_path = Path('/home/jianghui/dataset/MovieNet/')
movienet_subset_path = Path('/home/jianghui/dataset/MovieNet-Subset/')


file_list=[]
with open('./img_scene_movie', 'r') as f_list:
    for tt_id in f_list:
        # mkdir
        # tt_file_path = Path(movienet_subset_path, 'all', tt_id.replace("\n",""))
        # Path.mkdir(tt_file_path, exist_ok=True)
        file_list.append(tt_id.replace("\n", ""))

# copy img
# movienet_img_path = Path(movienet_path, "movie1K.keyframes.240p.v1/240P/")
# movienet_subset_img_path = Path(movienet_subset_path, "all/img/")

# for tt_id in file_list:
#     movienet_img_dir = Path(movienet_img_path, tt_id)
#     movienet_subset_img_dir = Path(movienet_subset_img_path, tt_id)
#     shutil.copytree(movienet_img_dir, movienet_subset_img_dir)


# copy json
movienet_json_path = Path(movienet_path, "/home/jianghui/dataset/MovieNet/annotation.v1/annotation")
movienet_subset_json_path = Path(movienet_subset_path, "all/scene/")

for tt_id in file_list:#["tt0047396"]:#

    # read src json file
    src_file_path = Path(movienet_json_path, tt_id + '.json')
    with open(src_file_path, "r") as f_json:
        src_json = json.load(f_json)
    
    des_json = []
    
    # generate scene segmentation
    for i in src_json["scene"]:
        scene_id = int(re.search(r'(?<=_)\d+', i["id"]).group(0))
        des_json.append({"scene_id":scene_id, "shot_start_id":i["shot"][0], "shot_end_id":i["shot"][1]})

    #save dest json file
    des_file_path = Path(movienet_subset_json_path, tt_id + '.json')
    with open(des_file_path, "w") as f_json:
        json.dump(des_json, f_json, indent=2)

