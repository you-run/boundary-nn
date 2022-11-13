import os

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

with open('./video_names.txt','r') as f:
    v_names = f.read().splitlines()

    for data in tqdm(v_names): # data: HB_1.mp4
        data_name = data.split(".")[0] # data_name: HB_1
        data_type = data[0:2] # data_type: HB
        data_path = os.path.join(VIDEO_PATH, data_type, data) # MemSeg_clips/HB/HB_1.mp4
        save_dir_path = os.path.join(SAVE_PATH, data_type, data_name) # video_frame/HB/HB_1/
        video_data = skvideo.io.vread(data_path)[20:-20, ...] # NumPy, 20~-20: Fade-in / Fade-out
        os.makedirs(save_dir_path, exist_ok=True) # HB/HB_1/

        for i in range(0, len(video_data), 2):
            frame_name = data_name + f"_{i}.png" # HB_1_i.png
            im = Image.fromarray(video_data[i, ...])
            im.save(os.path.join(save_dir_path, frame_name))