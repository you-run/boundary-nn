from time import time
import pynwb
import numpy as np
import math

# function for transforming sec to HZ, HZ to bin, bin to sec
def HZ_transform(matrix, column_name, abb):
    if f"{abb}_HZ" in matrix.keys():
        print('Already existing column')
        return
    column_idx = matrix.columns.get_loc(column_name)

    # make _HZ column and reposition it
    matrix[f"{abb}_HZ"] = matrix[column_name]*30 
    HZ_column = matrix.pop(f"{abb}_HZ")
    matrix.insert(column_idx+1,f"{abb}_HZ", HZ_column)

    # make _bin column and reposition it
    matrix[f"{abb}_bin"] = np.array(np.floor(matrix[f"{abb}_HZ"]),dtype=np.int_)
    bin_column = matrix.pop(f"{abb}_bin")
    matrix.insert(column_idx+2,f"{abb}_bin", bin_column)

    # make _regarded column and reposition it
    matrix[f"{abb}_regarded"] = matrix[f"{abb}_bin"] / 30
    reg_column = matrix.pop(f"{abb}_regarded")
    matrix.insert(column_idx+3,f"{abb}_regarded", reg_column)
    
save_path = 'C:/project/RNN/preprocessing/behavior_data'

with open("./nwb_names.txt",'r') as f:
    nwb_names = f.read().splitlines()

    for data in nwb_names:
        #print(data)
        if data[5] == '_': # if subject_num < 10
            subject_num = data[4]
            io = pynwb.NWBHDF5IO(f"C:/project/RNN/000207/{data[0:5]}/{data}","r")
            nwbfile = io.read()
        else: 
            subject_num = data[4:6]
            io = pynwb.NWBHDF5IO(f"C:/project/RNN/000207/{data[0:6]}/{data}","r")
            nwbfile = io.read()

        # make dataframe of encoding
        encoding_ori = nwbfile.intervals['encoding_table'].to_dataframe()
        encoding = encoding_ori[['fixcross_time', 'start_time', 'stop_time', \
            'boundary1_time', 'boundary2_time', 'boundary3_time', 'stimCategory','Clip_name']]
        enc_transform_list = np.array(['fixcross_time', 'start_time', 'stop_time'])
        enc_transform_abb = np.array(['fixation','Vstart','Vstop'])
        
        for names, names_abb in zip(enc_transform_list, enc_transform_abb):
            HZ_transform(encoding, names, names_abb)
        #print(encoding.keys())

        # for recognition_table
        recognition_ori = nwbfile.intervals['recognition_table'].to_dataframe()
        recognition = recognition_ori[['trial_num', 'fixcross_time','start_time','stop_time', \
            'RT','boundary_type','stimuli_type','resp_value','old_new','accuracy','confidence','frameName']]
        recognition['response_time'] = recognition['stop_time'] + recognition['RT']
        rec_transform_list = np.array(['fixcross_time','start_time','stop_time','response_time'])
        rec_transform_list_abb = np.array(['fixation','Fstart','Fstop','response'])

        for names, names_abb in zip(rec_transform_list, rec_transform_list_abb):
            HZ_transform(recognition, names, names_abb)
        recognition = recognition.iloc[:,]
        #print(recognition.keys())

        # for timediscrimination_table
        timeDiscrimination_ori = nwbfile.intervals['timediscrimination_table'].to_dataframe()
        timeDiscrimination = timeDiscrimination_ori[['trial_num','fixcross_time','start_time','stop_time',\
            'RT','boundary_type','key','resp_value','leftright','accuracy','confidence','frameName']]
        timeDiscrimination['response_time'] = timeDiscrimination['stop_time'] + timeDiscrimination['RT']
        time_transform_list = np.array(['fixcross_time','start_time','stop_time','response_time'])
        time_transform_list_abb = np.array(['fixation','Fstart','Fstop','response'])

        for names, names_abb in zip(time_transform_list,time_transform_list_abb):
            HZ_transform(timeDiscrimination, names, names_abb)
        #print(timeDiscrimination.keys())

        # save dataframes into csv file
        print(subject_num)
        encoding.to_csv(f"{save_path}/encoding_{subject_num}.csv", index=True)
        recognition.to_csv(f"{save_path}/recognition_{subject_num}.csv", index=False)
        timeDiscrimination.to_csv(f"{save_path}/timeDiscrimination_{subject_num}.csv", index=False)