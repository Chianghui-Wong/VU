import shutil
import sys
import os
from pathlib import Path
import json
import pysubs2
from pysubs2 import SSAFile, SSAEvent, make_time
import re

# common path
movienet_path = Path('/home/jianghui/dataset/MovieNet/')
movienet_subset_path = Path('/home/jianghui/dataset/MovieNet-Subset/')

# read tt_id
tt_list=[]
with open('./img_scene_movie', 'r') as f_list:
    for tt_id in f_list:
        tt_list.append(tt_id.replace("\n", ""))

# read info file (with fps)
info_path = Path(movienet_path, "movie1K.video_info.v1.json")
with open(info_path, 'r') as f_info:
    info_json = json.load(f_info)

# main
for tt_id in tt_list:#["tt0047396"]:#
    print(f'processing {tt_id} ...')

    # read shot list
    # [[start_frame, end_frame] ... ]
    shot_path = Path(movienet_path, 'movie1K.shot_detection.v1/shot', tt_id + ".txt")
    shot_list = []
    with open(shot_path, "r") as f_shot:
        for line in f_shot:
            line_list = line.split()
            start_frame = int(line_list[0])
            end_frame = int(line_list[1])
            shot_list.append([start_frame, end_frame])
    
    # read fps
    fps = info_json[tt_id]["fps"]

    # read srt list
    # [[start_frame, end_frame, subtitle] ... ]
    srt_path = Path(movienet_path, "subtitle1K.v1/subtitle", tt_id + '.srt')
    srt_list = []
    try:
        srt_file = SSAFile.load(srt_path, encoding='utf-8')
    except:
        try:
            srt_file = SSAFile.load(srt_path, encoding='ISO-8859-1')
        except:
            srt_file = SSAFile.load(srt_path, encoding='utf-16')

    for srt in srt_file:
        start_frame = pysubs2.time.ms_to_frames(srt.start, fps)
        end_frame = pysubs2.time.ms_to_frames(srt.end, fps)
        srt_element = [start_frame, end_frame, srt.plaintext.replace("\n"," ")]
        srt_list.append(srt_element)    

    # match shot with subtitle
    shot_list_format = []
    srt_index = 0
    for shot_index, shot in enumerate(shot_list):
        tmp_sentences = []
        tmp_sentences_id = []
        for srt in srt_list[srt_index:]:
            if (srt[0] >= shot[0]) and (srt[0] <= shot[1]) and (srt[1] <= shot[1]):
                tmp_sentences.append(srt[2])
                tmp_sentences_id.append(srt_index)
                srt_index += 1
            
            if (srt[0] >= shot[0]) and (srt[0] <= shot[1]) and (srt[1] > shot[1]):
                tmp_sentences.append(srt[2])
                tmp_sentences_id.append(srt_index)

            if (srt[0] <= shot[0]) and (srt[1] >= shot[0]) and (srt[1] <= shot[1]):
                tmp_sentences.append(srt[2])
                tmp_sentences_id.append(srt_index)
                srt_index += 1
            
            if (srt[0] <= shot[0]) and (srt[1] >= shot[0]) and (srt[1] > shot[1]):
                tmp_sentences.append(srt[2])
                tmp_sentences_id.append(srt_index)
        
        shot_list_format_element = {
            "shot_id":shot_index, 
            "scene_id":None,
            "subtitle_start_id":None if len(tmp_sentences) == 0 else tmp_sentences_id[0], 
            "subtitle_end_id":None if len(tmp_sentences) == 0 else tmp_sentences_id[-1], 
            "subtitle":tmp_sentences
            }
        shot_list_format.append(shot_list_format_element)

    # read scene     
    scene_path = Path(movienet_subset_path, "all/scene/", tt_id + '.json')
    with open(scene_path, 'r') as f_scene:
        scene_json = json.load(f_scene)
    for scene_json_element in scene_json:
        for shot_id in range(scene_json_element['shot_start_id'], scene_json_element['shot_end_id']+1):
            shot_list_format[shot_id]["scene_id"] = scene_json_element['scene_id']


    # save
    des_file_path = Path(movienet_subset_path, "all/shot/", tt_id + '.json')
    with open(des_file_path, "w") as f_json:
        json.dump(shot_list_format, f_json, indent=2)    

