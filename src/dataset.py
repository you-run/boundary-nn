import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, debug=False):
        super().__init__()
        self.root_dir = root_dir # .../video_frame
        self.data_path = list(filter(
            lambda x: (int(x[:-4].split("_")[-1])) % 2 == (0 if train else 1),
            glob.glob(root_dir + "/**/*.png", recursive=True)
        ))
        self.transform = transform
        if debug:
            self.data_path = self.data_path[:(len(self.data_path) // 10)]

    def __getitem__(self, idx):
        img = Image.open(self.data_path[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data_path)


class SequentialVideoFrameDataset:
    def __init__(self, root_dir, transform=None, debug=False):
        self.root_dir = root_dir # .../video_frame
        self.transform = transform
        self.debug = debug

    def get_video_frames(self, name="HB_1"):
        video_type = name.split("_")[0] # HB
        data_path = sorted(
            glob.glob(self.root_dir + f"/{video_type}/{name}/*.png", recursive=True),
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
        outputs = []
        with torch.no_grad():
            for start_idx, end_idx in zip(batch_idx, batch_idx[1:]):
                outputs.append(model(frames[start_idx:end_idx, ...])[0][0].detach().cpu().numpy())
        outputs = np.vstack(outputs)
        return outputs

    def __call__(self, name="HB_1"):
        return self.get_video_frames(name)


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

    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    dataset = SequentialVideoFrameDataset('../data/video_frame/', transform=transform)

    from model import ModuleUtils
    class MyModule(nn.Module, ModuleUtils):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(480, 240)

        def forward(self, x):
            return (self.fc(x), torch.randn(5, 5)), torch.randn(5, 5)

    model = MyModule().to(device)
    batch_size = 4
    outputs = dataset.get_output_with_batch(model, batch_size, "HB_1")
    print(outputs.shape)

    # ret = dataset(name="HB_1")
    # print(ret.shape)

    # tf = T.ToPILImage()
    # for i, r in enumerate(ret):
    #     img = tf(r)
    #     img.show()
    #     if i == 50:
    #         break
