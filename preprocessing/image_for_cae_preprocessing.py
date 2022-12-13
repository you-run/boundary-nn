import os

import glob
import skvideo
import skvideo.io
from tqdm import tqdm
from PIL import Image

VIDEO_PATH = '../data/MemSeg_clips'
SAVE_PATH = '../data/video_frame'

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, 'HB'), exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, 'NB'), exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, 'SB'), exist_ok=True)

file_path_list = glob.glob(VIDEO_PATH + "/**/*.mp4", recursive=True)
for n, file_path in enumerate(file_path_list):
    # file_path: ../data/MemSeg_clips/HB/HB_1.mp4
    boundary_type, video_name_with_ext = file_path.split('/')[-2:] # HB, HB_1.mp4 
    video_name = video_name_with_ext.split('.')[0] # HB_1
    print(f"[{n + 1}/{len(file_path_list)}] Now Processing {video_name} ...")

    frame_save_dir_path = os.path.join(SAVE_PATH, boundary_type, video_name)
    os.makedirs(frame_save_dir_path, exist_ok=True)
    
    video_data = skvideo.io.vread(file_path)[20:-20, ...]
    for i, frame in enumerate(video_data):
        frame_name = f"{video_name}_{i}.png" # HB_1_i.png
        im = Image.fromarray(frame)
        im.save(os.path.join(frame_save_dir_path, frame_name))
