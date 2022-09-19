from PIL import Image
from numpy import asarray
import numpy as np
import os

os.mkdir("C:/project/RNN/preprocessing/frame_data")
os.mkdir("C:/project/RNN/preprocessing/frame_data/recognition")
os.mkdir("C:/project/RNN/preprocessing/frame_data/time_discrimination")

frame_path = "C:/project/RNN/video" # where frame image is saved
data_path = "C:/project/RNN/preprocessing" # name txt file is saved
save_path = "C:/project/RNN/preprocessing/frame_data" # where to save

def frame2npy(frame_path, data_path, save_path, type):
    if type == 'rec':
        with open(f"{data_path}/Fnames_rec.txt","r") as f:
            frame_names = f.read().splitlines()
            frame_dir = "MemSeg_SceneRecogImg"
            save_dir = "recognition"
    elif type == 'time':
        with open(f"{data_path}/Fnames_time.txt","r") as f:
            frame_names = f.read().splitlines()
            frame_dir = "MemSeg_timeDiscrimImg"
            save_dir = "time_discrimination"

    for data in frame_names:
        print(data)
        image = Image.open(f"{frame_path}/{frame_dir}/{data[0:2]}/{data}")
        img2npy = asarray(image)
        np.save(f"{save_path}/{save_dir}/{data[0:-4]}.npy", img2npy)


frame2npy(frame_path, data_path, save_path,'rec')
frame2npy(frame_path, data_path, save_path, 'time')