import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, transform=None, debug=False):
        super().__init__()
        self.root_dir = root_dir # .../video_frame
        self.data_path = glob.glob(root_dir + "/**/*.png", recursive=True)
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
        data_path = glob.glob(self.root_dir + f"/{video_type}/{name}/*.png", recursive=True)
        data_path = sorted(data_path, key=lambda x: int(x[:-4].split("_")[-1]))
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
    
    def __call__(self, name="HB_1"):
        return self.get_video_frames(name)
        


if __name__ == "__main__":
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize(size=(270, 480)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # dataset = VideoFrameDataset('../data/video_frame/', transform=transform)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    # for x in dataloader:
    #     print(x.shape)
    #     assert 0 

    dataset = SequentialVideoFrameDataset('../data/video_frame/', transform=transform)
    ret = dataset(name="HB_1")
    print(ret.shape)

    # tf = T.ToPILImage()
    # for i, r in enumerate(ret):
    #     img = tf(r)
    #     img.show()
    #     if i == 50:
    #         break
