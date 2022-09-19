import os

import skvideo
# skvideo.setFFmpegPath('./FFmpeg/bin')
import skvideo.io
import numpy as np

VIDEO_PATH = None
# VIDEO_PATH = "../video/MemSeg_clips"

os.makedirs("./data", exist_ok=True)
os.makedirs("./data/HB", exist_ok=True)
os.makedirs("./data/NB", exist_ok=True)
os.makedirs("./data/SB", exist_ok=True)

with open('./video_names.txt','r') as f:
    v_names = f.read().splitlines()

    for data in v_names:
        print(data)
        if data[0:2] == 'HB':
            video_data = skvideo.io.vread(os.path.join(VIDEO_PATH, f"HB/{data}"))
            np.save(f"./data/HB/{data[0:-4]}.npy", video_data)
        elif data[0:2] == 'NB':
            video_data = skvideo.io.vread(os.path.join(VIDEO_PATH, f"NB/{data}"))
            np.save(f"./data/NB/{data[0:-4]}.npy", video_data)
        elif data[0:2] == 'SB':
            video_data = skvideo.io.vread(os.path.join(VIDEO_PATH, f"SB/{data}"))
            np.save(f"./data/SB/{data[0:-4]}.npy", video_data)
