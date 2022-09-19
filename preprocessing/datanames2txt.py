import os

# video names into txt file
def videoNames2txt(data_path, save_path):
    print('video2txt')
    dir_list = os.listdir(data_path)
    for dir in dir_list:
        files = os.listdir(data_path + '/' + dir)
        for data in files:
            with open(f"{save_path}/video_names.txt","a+") as f:
                f.write(data+'\n')

# NWB data names into txt file
def NWB2txt(data_path, save_path):
    print('nwb2txt')
    dir_list = os.listdir(data_path)
    for Dir in dir_list:
        #print(Dir)
        if Dir == 'dandiset.yaml':
            continue
        files = os.listdir(data_path + '/' + Dir)
        for Data in files:
            with open(f"{save_path}/nwb_names.txt","a+") as F:
                F.write(Data + '\n')

# frame names into txt file
def frame2txt(data_path, save_path,task):
    print('frame2txt')
    dir_list = os.listdir(data_path)
    for dir in dir_list:
        #print(dir)
        files = os.listdir(data_path + '/' + dir)
        for data in files:
            if task == 'rec':
                with open(f"{save_path}/Fnames_rec.txt","a+") as f:
                    f.write(data + '\n')
            elif task == 'time':
                with open(f"{save_path}/Fnames_rec.txt","a+") as f:
                    f.write(data + '\n')

# video names into txt file
video_path = "C:/project/RNN/video/MemSeg_clips"
txt_path = "C:/project/RNN/preprocessing"
videoNames2txt(video_path, txt_path)

# NWB names into txt file
nwb_path = "C:/project/RNN/000207/"
NWB2txt(nwb_path,txt_path)

# Recognition frame names into txt file
rec_frame_path = "C:/project/RNN/video/MemSeg_SceneRecogImg"
frame2txt(rec_frame_path, txt_path, 'rec')

# TimeDiscrimination frame names into txt file
time_frame_path = "C:/project/RNN/video/MemSeg_timeDiscrimImg"
frame2txt(time_frame_path, txt_path, 'time')