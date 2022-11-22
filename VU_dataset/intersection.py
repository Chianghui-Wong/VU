import os
import sys
from pathlib import Path

scene_list=[]
with open('./scene_annotated_movie', 'r') as f_list:
    for tt_id in f_list:
        scene_list.append(tt_id.replace("\n", ""))

img_list=[]
with open('/home/jianghui/dataset/MovieNet/list.v1.txt', 'r') as f_list:
    for tt_id in f_list:
        img_list.append(tt_id.replace("\n", ""))

img_scene_list = [i for i in scene_list if i in img_list]


with open('./img_scene_movie','a') as f:
    for tt_id in img_scene_list:
        f.write(tt_id + '\n')