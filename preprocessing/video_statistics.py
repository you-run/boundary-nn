import cv2
import os
import numpy as np
from PIL import Image, ImageChops, ImageStat
import matplotlib.pyplot as plt
import random

'''
identifing NB, SB, HB with video statistics
'''

def pixel_difference(data_path, data):
    cap1 = cv2.VideoCapture(data_path +'/' + data)
    cap2 = cv2.VideoCapture(data_path + '/' + data)
    _, image2 = cap2.read()
    _, image1 = cap1.read()
    _, image2 = cap2.read()
    frame_length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    diff_mean = np.zeros(frame_length-1)
    diff_var = np.zeros(frame_length-1)
    for frame_num in range(frame_length-1):
        image1_pil = Image.fromarray(image1)
        image2_pil = Image.fromarray(image2)
        diff2 = ImageChops.difference(image2_pil, image1_pil) # pixel by pixel difference
        stat = ImageStat.Stat(diff2)
        diff_mean[frame_num] = sum(stat.mean)
        diff_var[frame_num] = sum(stat.var)
        #print(f"{frame_num}th vs {frame_num+1}th frame pixel difference : mean_sum={diff_mean}, var_sum = {diff_var}")
        _, image1 = cap1.read()
        _, image2 = cap2.read()
        #break
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    return (diff_mean, diff_var)

def frame_average(data_path, data):
    cap = cv2.VideoCapture(data_path + '/' + data)
    _, image = cap.read() # image : (w,h,c) = (540, 960, 3)
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_pix = np.zeros(frame_length)
    frame_var = np.zeros(frame_length)
    for frame_num in range(frame_length):
        frame_pix[frame_num] = np.mean(image)
        frame_var[frame_num] = np.var(image)
        #print(np.mean(image))
        _, image = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    return frame_pix, frame_var

def plot_twin_results(data, label, title):
    num_data = len(data)
    fig,ax = plt.subplots()
    ax.plot(data[0], color='red')
    ax.set_xlabel('frame number')
    ax.set_ylabel(label[0], color='red')
    #ax.set_ylim(0,300)
    ax2 = ax.twinx()
    ax2.plot(data[1], color='blue',alpha=0.5)
    ax2.set_ylabel(label[1], color='blue')
    #ax2.set_ylim(0,12000)
    plt.title(f"{title} in {video}")
    plt.show()


boundary_path = 'C:/project/RNN/video/MemSeg_clips'
num_video = 3
'''
frame_pix, frame_var = frame_average('C:/project/RNN/video/MemSeg_clips/HB', 'HB_23.mp4')
video = 'HB_23.mp4'
plot_twin_results((frame_pix, frame_var), ('mean','variance'), 'frame-by-frame difference')
'''

for boundary in os.listdir(boundary_path):
    #print(boundary)
    path = os.path.join(boundary_path, boundary)
    videos = os.listdir(path)
    random.shuffle(videos)
    for video in videos[0:num_video]:
        print(video)
        #mean, var = pixel_difference(path, video)
        #plot_twin_results((mean, var), ('mean', 'variance'))
        frame_pix, frame_var = frame_average(path, video)
        plot_twin_results((frame_pix, frame_var), ('mean','variance'), 'frame-by-frame difference')
        