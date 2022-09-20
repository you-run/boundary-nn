import pandas as pd
import os
import numpy as np

csv_path = "C:/project/RNN/preprocessing/behavior_data"
#os.mkdir(f"{csv_path}/behavior_npy")
save_path = "C:/project/RNN/preprocessing/behavior_data/behavior_npy"

# preprocessing behavior data from dataframe to numpy array
def behavior2npy(csv_path, file_name, save_path, type):
    # only run for files of certain type
    if type in file_name:
        print(file_name)
        # get subject number
        if file_name[file_name.find('_')+2] == '.':
            sub_num = file_name[file_name.find('_')+1]
        else:
            sub_num = file_name[file_name.find('_')+1:file_name.find('_')+3]
        
        # process dataframe data
        df = pd.read_csv(f"{csv_path}/{file_name}")

        resp_value = df["resp_value"]  
        resp_bin = df["response_bin"]
        if type == 'rec':
            answer = df["stimuli_type"] + 1 # change 0,1 -> 1,2
        elif type == 'time':
            answer = df["key"]
        
        # make numpy array of resp_value and answer at certain time bin
        beh = np.zeros((2, max(resp_bin)+30))
        for bin, response, ans in zip(resp_bin, resp_value, answer):
            beh[0,bin] = response
            beh[1,bin] = ans

        # save in numpy array for each subjects
        np.save(f"{save_path}/{type}_{sub_num}.npy", beh)
        print('saved')
    else:
        return


files = os.listdir(csv_path)

for data in files:
    # if data is not csv file, skip
    if data[-4:] != '.csv':
        continue
    behavior2npy(csv_path, data, save_path, 'rec')
    behavior2npy(csv_path, data, save_path, 'time')

#behavior2npy(csv_path, "recognition_1.csv", save_path, 'rec')