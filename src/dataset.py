import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, transform=None, debug=False):
        super().__init__()
        self.root_dir = root_dir # .../video_frame
        self.data_path = glob.glob(root_dir + "/**/*.png", recursive=True)
        self.transform = transform
        self.debug = debug

    def __getitem__(self, idx):
        img = Image.open(self.data_path[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data_path)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(size=(270, 480)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset = VideoFrameDataset('../data/video_frame/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    for x in dataloader:
        print(x.shape)
        assert 0 
