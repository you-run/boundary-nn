import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils import get_system_info, set_figure_options
from dataset import RandomFrameDataset, SceneRecogDataset
from model import MODEL_DICT


DATA_PATH = "../../../data"
MODEL_PATH = "../log/Dec26_03:33:55_resvae-v5-BEST/resvae-v5_last_epoch.pt"
MODEL_NAME = "resvae-v5"


if __name__ == "__main__":
    # Settings
    set_figure_options()
    device, num_workers = get_system_info()
    print(f"Device: {device}")

    transform = transform=transforms.Compose([
        transforms.Resize(size=(256, 512)),
        transforms.ToTensor(),
    ])

    frame_dataset = RandomFrameDataset(
        os.path.join(DATA_PATH, "video_frame"),
        transform=transform,
        train=True,
        eval_mode=0,
    )
    train_dataloader = DataLoader(
        frame_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers
    )

    model = MODEL_DICT[MODEL_NAME](in_channels=32, latent_dim=512)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    scene_recog_dataset = SceneRecogDataset(root_dir=os.path.join(DATA_PATH, "MemSeg_SceneRecogImg"), transform=transform)


    model.eval()
    with torch.no_grad():
        video_mu_set = []
        video_log_var_set = []

        # Video latent
        for x in tqdm(train_dataloader):
            x = x.to(device)
            (mu, log_var), preds = model(x)
            video_mu_set.append(mu.detach().cpu().numpy())
            video_log_var_set.append(log_var.detach().cpu().numpy())
        video_mu_set = np.vstack(video_mu_set)
        video_log_var_set = np.vstack(video_log_var_set)

        # Frame latent
        new_set = []
        old_set = []
        for btype in ("HB", "NB", "SB"):
            for i in tqdm(range(30)):
                video_name = "{btype}_{i}"
                images, new_idx = scene_recog_dataset.get_single_data(name=video_name)
                new_set.append(images[new_idx])
                old_set.append(images[1 - new_idx])
        new_set = torch.stack(new_set)
        old_set = torch.stack(old_set)

        new_set, old_set = new_set.to(device), old_set.to(device)
        (new_mu_set, new_log_var_set), _ = model(new_set)
        (old_mu_set, old_log_var_set), _ = model(old_set)
        new_mu_set = new_mu_set.detach().cpu().numpy()
        old_mu_set = old_mu_set.detach().cpu().numpy()
        # Skip log_var cause it is not necessary for now

    print(video_mu_set.shape, new_mu_set.shape, old_mu_set.shape)

    latent_set = np.stack(video_mu_set, new_mu_set, old_mu_set)
    print(latent_set.shape)
