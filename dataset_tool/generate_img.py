import shutil
import sys
import os
from pathlib import Path
import json
import re

movienet_path = Path('/home/jianghui/dataset/MovieNet/')
movienet_subset_path = Path('/home/jianghui/dataset/MovieNet-Subset/')

# read file list
file_list=[]
with open('./img_scene_movie', 'r') as f_list:
    for tt_id in f_list:
        file_list.append(tt_id.replace("\n", ""))

movienet_img_path = Path(movienet_path, "movie1K.keyframes.240p.v1/240P/")
movienet_subset_img_path = Path(movienet_subset_path, "all/img/")

prog_shot = re.compile('(?<=shot_)\d+(?=_img_)')
prog_img = re.compile('(?<=img_)\d+(?=.jpg)')

for tt_id in file_list:
    movienet_img_dir = Path(movienet_img_path, tt_id)
    movienet_subset_img_dir = Path(movienet_subset_img_path, tt_id)
    Path.mkdir(movienet_subset_img_dir, exist_ok=True)
    for movienet_img_dir_file in movienet_img_dir.iterdir():
        movienet_img_dir_file_name = movienet_img_dir_file.name
        shot_id_match = prog.search(movienet_img_dir_file_name)
        if shot_id_match != None:
            shot_id = int(shot_id_match.group(0))
            movienet_subset_img_file = Path(movienet_subset_img_dir, f'{shot_id}.jpg')
            print(f'copying {movienet_subset_img_file} ...')
            shutil.copyfile(movienet_img_dir_file, movienet_subset_img_file)
            
            


    
    
    # Path.mkdir(movienet_subset_img_dir, exist_ok=True)
    # shutil.copytree(movienet_img_dir, movienet_subset_img_dir)


