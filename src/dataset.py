import os
from collections import defaultdict

import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import pandas as pd

def get_dirname(video_name):
    dirname = video_name
    if "NB" in video_name:
        dirname = f"NB_0cut_{video_name.split('_')[-1]}"
    return dirname


class RandomFrameDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        train=True,
        train_indices=None,
        eval_mode=0,
        debug=False,
    ):
        """ RandomFrameDataset

        Args:
            root_dir (_type_): _description_
            transform (_type_, optional): _description_. Defaults to None.
            train (bool, optional): _description_. Defaults to True.
            train_indices (list, optional): train data indices, list of length 24
                enabled only when eval_mode is 1
            eval_mode (int, optional): _description_. Defaults to 0.
                0: Split by even/odd-numbered frames across all videos
                1: Stratified video split
            debug (bool, optional): _description_. Defaults to False.
        """
        assert eval_mode in (0, 1), f"Undefined eval_mode of RandomFrameDataset: {eval_mode}, expected 0 or 1."

        super().__init__()
        self.transform = transform

        if eval_mode == 0:
            self.data_path = list(filter(
                lambda x: (int(x[:-4].split("_")[-1])) % 2 == (0 if train else 1),
                glob.glob(root_dir + "/**/*.png", recursive=True)
            ))
        else: # eval_mode == 1:
            self.data_path = []
            for btype in ("HB", "NB", "SB"):
                indices = train_indices
                if not train:
                    indices = list(set(range(30)).difference(set(indices)))

                video_list = sorted(
                    os.listdir(os.path.join(root_dir, btype)),
                    key=lambda x: int(x.split("_")[-1])
                )
                video_list = list(map(video_list.__getitem__, indices))
                for video_name in video_list:
                    self.data_path.extend(glob.glob(root_dir + f"/{btype}/{video_name}/*.png"))
            self.data_path = list(filter(
                lambda x: (int(x[:-4].split("_")[-1])) % 2 == 0,
                self.data_path
            ))

        if debug:
            self.data_path = self.data_path[:(len(self.data_path) // 50)]

    def __getitem__(self, idx):
        img = Image.open(self.data_path[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data_path)


class VideoFrameDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        train=True,
        train_indices=None,
        eval_mode=0,
        debug=False,
    ):
        """ VideoFrameDataset

        Args:
            root_dir (_type_): _description_
            transform (_type_, optional): _description_. Defaults to None.
            train (bool, optional): _description_. Defaults to True.
            train_indices (list, optional): train data indices, list of length 24
                enabled only when eval_mode is 1
            eval_mode (int, optional): _description_. Defaults to 0.
                0: Split by even/odd-numbered frames across all videos
                1: Stratified video split
            debug (bool, optional): _description_. Defaults to False.
        """
        assert eval_mode in (0, 1), f"Undefined eval_mode of VideoFrameDataset: {eval_mode}, expected 0 or 1."

        super().__init__()
        self.transform = transform

        if eval_mode == 0:
            data_path = list(filter(
                lambda x: (int(x[:-4].split("_")[-1])) % 2 == (0 if train else 1),
                glob.glob(root_dir + "/**/*.png", recursive=True)
            ))
        else: # eval_mode == 1:
            data_path = []
            for btype in ("HB", "NB", "SB"):
                indices = train_indices
                if not train:
                    indices = list(set(range(30)).difference(set(indices)))

                video_list = sorted(
                    os.listdir(os.path.join(root_dir, btype)),
                    key=lambda x: int(x.split("_")[-1])
                )
                video_list = list(map(video_list.__getitem__, indices))
                for video_name in video_list:
                    data_path.extend(glob.glob(root_dir + f"/{btype}/{video_name}/*.png"))
            data_path = list(filter(
                lambda x: (int(x[:-4].split("_")[-1])) % 2 == 0,
                data_path
            ))

        if debug:
            data_path = data_path[:(len(data_path) // 50)]

        self.dataset = defaultdict(list)
        for d in data_path:
            self.dataset[d.split('/')[-2]].append(d)
        for k, v in self.dataset.items():
            self.dataset[k] = sorted(v, key=lambda x: int(x[:-4].split("_")[-1]))[::2]
        self.dataset = list(self.dataset.values())

    def __getitem__(self, video_idx):
        pathset = self.dataset[video_idx]
        imageset = []
        for path in pathset:
            img = Image.open(path)
            if self.transform is not None:
                img = self.transform(img)
            imageset.append(np.expand_dims(img, axis=0))
        imageset = np.vstack(imageset)
        return imageset

    def __len__(self):
        return len(self.dataset)


class VideoCollator(object):
    def __init__(self, seq_len, padding_value=0.):
        self.seq_len = seq_len
        self.padding_value = padding_value

    def __call__(self, data): # list[(L, C, H, W)], len(list) == N
        self.input_seq = pad_sequence(
            data,
            batch_first=True,
            padding_value=self.padding_value
        ) # (N, L_max, C, H, W)
        return self.input_seq


class SingleVideoHandler:
    def __init__(self, root_dir, transform=None, debug=False):
        self.root_dir = root_dir # .../video_frame
        self.transform = transform
        self.debug = debug

    def get_video_frames(self, name="HB_1"):
        video_type = name.split("_")[0] # HB
        data_path = sorted(
            glob.glob(self.root_dir + f"/{video_type}/{get_dirname(name)}/*.png", recursive=True),
            key=lambda x: int(x[:-4].split("_")[-1])
        )
        assert len(data_path) != 0, f"Not a valid video name: {name}"

        if self.debug:
            data_path = data_path[:(len(data_path) // 10)]

        frames = []
        for file_path in data_path:
            img = Image.open(file_path)
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)
        frames = torch.stack(frames)

        return frames

    def get_output_with_batch(self, model, batch_size, name="HB_1"):
        device = model.device
        frames = self.get_video_frames(name=name).to(device)
        batch_idx = list(range(0, len(frames) + batch_size + 1, batch_size))

        model.eval()
        mu_set = []
        log_var_set = []
        recon_set = []
        with torch.no_grad():
            for start_idx, end_idx in zip(batch_idx, batch_idx[1:]):
                (mu, log_var), recon = model(frames[start_idx:end_idx, ...])
                mu_set.append(mu.detach().cpu().numpy())
                log_var_set.append(log_var.detach().cpu().numpy())
                recon_set.append(recon.detach().cpu().numpy())
        mu = np.vstack(mu_set)
        log_var = np.vstack(log_var_set)
        recon = np.vstack(recon_set)
        return (mu, log_var), recon

    def __call__(self, name="HB_1"):
        return self.get_video_frames(name)


class SceneRecogDataset:
    def __init__(self, root_dir, transform=None, debug=False):
        self.root_dir = root_dir # .../MemSeg_SceneRecogImg
        self.transform = transform
        self.debug = debug
        self.data_path = glob.glob(self.root_dir + "/**/*.png", recursive=True)

    def get_single_data(self, name="HB_1"):
        target_data_path = sorted([x for x in self.data_path if f"{name}_" in x])
        assert len(target_data_path) == 2, f"Not a valid video name: {name}"

        new_idx = int("foil" in target_data_path[1])
        target_data = []
        for file_path in target_data_path:
            img = Image.open(file_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            target_data.append(img)

        return target_data, new_idx



class TimeDiscriDataset:
    def __init__(self, root_dir, transform=None, debug=False):
        self.root_dir = root_dir # .../MemSeg_timeDiscrimImg
        self.transform = transform
        self.debug = debug
        self.data_path = glob.glob(self.root_dir + "/**/*.png", recursive=True)

    def get_single_data(self, name="HB_1"):
        target_data_path = sorted([x for x in self.data_path if f"{name}_" in x])
        assert len(target_data_path) == 4, f"Not a valid video name: {name}"

        front_idx = [idx for idx, path in enumerate(target_data_path) if "A" in path ]
        assert len(front_idx) == 2, f"Index len should be 2, got {len(front_idx)}"
        target_data = []
        for file_path in target_data_path:
            #print(file_path)
            img = Image.open(file_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            target_data.append(img)
        return target_data, front_idx

def get_recon_dataset(
    root_dir,
    transform=None,
    frames=[
        "HB/HB_1/HB_1_28.png", "HB/HB_20/HB_20_146.png", "HB/HB_30/HB_30_32.png",
        "NB/NB_0cut_12/NB_0cut_12_22.png", "NB/NB_0cut_24/NB_0cut_24_25.png", "NB/NB_0cut_3/NB_0cut_3_162.png",
        "SB/SB_5/SB_5_128.png", "SB/SB_19/SB_19_108.png", "SB/SB_30/SB_30_116.png"
    ]
):
    images = [Image.open(os.path.join(root_dir, frame_path)) for frame_path in frames]
    if transform is not None:
        images = [transform(image) for image in images]
    images = torch.stack(images)
    return images


if __name__ == "__main__":
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([
        T.Resize(size=(270, 480)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # dataset = VideoFrameDataset('../data/video_frame/', transform=transform, train=False)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    # for x in dataloader:
    #     print(x.shape)
    #     assert 0

    # import torch.nn as nn

    # device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    # dataset = SingleVideoHandler('../data/video_frame/', transform=transform)

    # from model import ModuleUtils
    # class MyModule(nn.Module, ModuleUtils):
    #     def __init__(self):
    #         super().__init__()
    #         self.fc = nn.Linear(480, 240)

    #     def forward(self, x):
    #         return (self.fc(x), torch.randn(5, 5)), torch.randn(5, 5)

    # model = MyModule().to(device)
    # batch_size = 4
    # outputs = dataset.get_output_with_batch(model, batch_size, "HB_1")
    # print(outputs.shape)

    #####
    dataset = SceneRecogDataset(root_dir='../data/MemSeg_SceneRecogImg', transform=transform)
    print(dataset.get_single_data(name="HB_1")[0][0].shape)
    print(len(dataset.get_single_data(name="HB_1")[0]))
    print(dataset.get_single_data(name="HB_1")[1])
    # ret = dataset(name="HB_1")
    # print(ret.shape)
    #####

    # tf = T.ToPILImage()
    # for i, r in enumerate(ret):
    #     img = tf(r)
    #     img.show()
    #     if i == 50:
    #         break
