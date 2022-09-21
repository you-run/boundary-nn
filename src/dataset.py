import os
import numpy as np
import pandas as pd
import torch

class RecognitionDataLoader:
    def __init__(self, data_dir, batch_size, sub_indexes, block_size=300):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sub_indexes = sub_indexes
        self.block_size = block_size

    def __iter__(self):
        return RecognitionDataLoaderObject(
            self.data_dir,
            self.batch_size,
            self.sub_indexes,
            self.block_size
        ).__iter__()


class RecognitionDataLoaderObject:
    def __init__(self, data_dir, batch_size, sub_indexes, block_size=300):
        self.target_sub = np.random.choice(sub_indexes)
        self.beh_df = pd.read_csv(
            os.path.join(data_dir, "behavior", "behavior_df", f"recognition_{self.target_sub}.csv")
        )[["Fstart_bin", "Fstop_bin", "frameName"]]
        self.beh_df.frameName = self.beh_df.frameName.apply(lambda x: x[2:-1]) # Byte encoded -> UTF-8
        self.beh_y = np.load(
            os.path.join(data_dir, "behavior", "behavior_npy", f"rec_{self.target_sub}.npy")
        )[1, :]
        self.frame_dir = os.path.join(data_dir, "frame", "recognition")

        self.cur_pos = self.beh_df.Fstart_bin.min()
        self.start_pos = self.beh_df.Fstart_bin.min()
        self.end_pos = self.beh_df.Fstop_bin.max()

        self.batch_size = batch_size  #TODO: Batching
        self.block_size = block_size

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cur_pos >= self.end_pos:
            raise StopIteration

        y = self.beh_y[self.cur_pos: self.cur_pos + self.block_size]
        y = torch.from_numpy(y).unsqueeze(0)
        x = torch.zeros(self.batch_size, y.size(1), 3, 540, 960)
        self.cur_pos += self.block_size

        return x, y

    def __len__(self):
        return (self.end_pos - self.start_pos + 1) // self.block_size
