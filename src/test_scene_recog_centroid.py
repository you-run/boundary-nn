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

def get_video_latent(
    model,
    video_handler,
    n_components=3
):
    for btype in ("HB", "NB", "SB"):
        for i in tqdm(range(30)):
            video_name = f"{btype}_{i}"
            video_frames = video_handler.get_video_frames(name=video_name)
            model_input = torch.vstack((video_frames, image_a.unsqueeze(0), image_b.unsqueeze(0)))
    model.eval()
    with torch.no_grad():
        video_frames = video_handler.get_video_frames(name=video_name)
        model_input = torch.vstack((video_frames, image_a.unsqueeze(0), image_b.unsqueeze(0)))
        (mu, _), _ = model(model_input) # (N+2, 512)
    
    pca = PCA(n_components=n_components)
    denoised_points = pca.fit_transform(mu) # (N+2, 3)

    distance = np.linalg.norm(denoised_points[-1, :] - denoised_points[-2, :])
    return distance

def get_euclidean_dist(
    model,
    video_handler,
    scene_recog_dataset,
    video_name,
    n_components=3
):
    model.eval()
    with torch.no_grad():
        video_frames = video_handler.get_video_frames(name=video_name)
        (image_a, image_b), new_idx = scene_recog_dataset.get_single_data(name=video_name)
        model_input = torch.vstack((video_frames, image_a.unsqueeze(0), image_b.unsqueeze(0)))
        (mu, _), _ = model(model_input) # (N+2, 512)
    
    pca = PCA(n_components=n_components)
    denoised_points = pca.fit_transform(mu) # (N+2, 3)

    distance = np.linalg.norm(denoised_points[-1, :] - denoised_points[-2, :])
    return distance

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
        batch_size=128,
        shuffle=True,
        num_workers=num_workers
    )

    model = MODEL_DICT[MODEL_NAME](in_channels=32, latent_dim=512)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    scene_recog_dataset = SceneRecogDataset(root_dir=os.path.join(DATA_PATH, "MemSeg_SceneRecogImg"), transform=transform)

    model.train()
    mu_set = []
    log_var_set = []
    for x in train_dataloader:
        x = x.to(model.device)
        (mu, log_var), preds = model(x)
        mu_set.append(mu.detach().cpu().numpy())
        log_var_set.append(log_var.detach().cpu().numpy())

    mu = np.vstack(mu_set)
    log_var = np.vstack(log_var_set)
    print(mu.shape, log_var.shape)
