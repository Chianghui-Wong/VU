import json
from pathlib import Path
import os
import sys

# get file path
dis_dir_path = Path('/home/jianghui/dataset/MovieNet/annotation.v1/annotation')
# dis_file_name = 'tt0038650.json'
# dis_file_path = Path(dis_dir_path, dis_file_name)

# read the json file
f_save = open("/home/jianghui/VideoUnderStanding_all/dataset_tool/scene_annotated_movie", "a")

for dis_file_path in dis_dir_path.iterdir():
    with open(dis_file_path, "r") as f_json:
        json_dict = json.load(f_json)
    if ("scene" in json_dict) and (json_dict["scene"] != None) and (json_dict["story"] != None):
        f_save.write(dis_file_path.stem + '\n')

f_save.close()


# process json file