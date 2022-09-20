import os

import skvideo
# skvideo.setFFmpegPath('./FFmpeg/bin')
import skvideo.io
import numpy as np

VIDEO_PATH = '../MemSeg_clips'
SAVE_PATH = '../data/video'

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, 'HB'), exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, 'NB'), exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, 'SB'), exist_ok=True)

with open('./video_names.txt','r') as f:
    v_names = f.read().splitlines()

    for data in v_names:
        print(data)
        if data[0:2] == 'HB':
            video_data = skvideo.io.vread(os.path.join(VIDEO_PATH, f"HB/{data}"))
            np.save(os.path.join(SAVE_PATH, 'HB', f"{data[0:-4]}.npy"), video_data)
        elif data[0:2] == 'NB':
            video_data = skvideo.io.vread(os.path.join(VIDEO_PATH, f"NB/{data}"))
            np.save(os.path.join(SAVE_PATH, 'NB', f"{data[0:-4]}.npy"), video_data)
        elif data[0:2] == 'SB':
            video_data = skvideo.io.vread(os.path.join(VIDEO_PATH, f"SB/{data}"))
            np.save(os.path.join(SAVE_PATH, 'SB', f"{data[0:-4]}.npy"), video_data)
