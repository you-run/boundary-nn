import os
import json
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils import get_system_info, set_figure_options
from dataset import SingleVideoHandler, SceneRecogDataset
from model import MODEL_DICT
from loss import LOSS_DICT


def sampling_points(mu, log_var, n=100):
    std = torch.exp(log_var / 2)
    eps = torch.randn((n, *mu.shape))
    z = (mu + std * eps).detach().cpu().numpy() # (N, *mu.shape)
    return z

def plot_scene_recog(
    ax,
    model,
    video_handler,
    scene_recog_dataset,
    video_name,
    n_sampling=30,
    n_components=3
):
    model.eval()
    with torch.no_grad():
        video_frames = video_handler.get_video_frames(name=video_name)
        (image_a, image_b), new_idx = scene_recog_dataset.get_single_data(name=video_name)
        model_input = torch.vstack((video_frames, image_a.unsqueeze(0), image_b.unsqueeze(0)))
        (mu, log_var), _ = model(model_input)
    
    sampled_points = sampling_points(mu, log_var, n=n_sampling) # (n, *mu.shape)
    n_points, n_t, z_dim = sampled_points.shape
    sampled_points = sampled_points.reshape(-1, z_dim) # (n * n_t, z_dim)

    pca = PCA(n_components=n_components)
    reduced_points = pca.fit_transform(sampled_points) # (n * n_t, 3)
    reduced_points = reduced_points.reshape(n_points, n_t, n_components) # (n, n_t, 3)

    for points in reduced_points:
        ax.scatter(points[:-2, 0], points[:-2, 1], points[:-2, 2], c=np.arange(len(points) - 2), s=2, cmap='Greens', alpha=0.2)
        ax.scatter(points[-2, 0], points[-2, 1], points[-2, 2], s=6, c='red' if new_idx == 0 else 'blue', marker='x')
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], s=6, c='red' if new_idx == 1 else 'blue', marker='x')
    
    red_patch = mpatches.Patch(color='red', label='New')
    blue_patch = mpatches.Patch(color='blue', label='Old')
    ax.legend(handles=[red_patch, blue_patch])
    ax.set_title(f"Scene recognition: {video_name}")


if __name__ == "__main__":
    # Settings
    set_figure_options()
    device, num_workers = get_system_info()
    print(f"Device: {device}")

    transform = transform=transforms.Compose([
        transforms.Resize(size=(256, 512)),
        transforms.ToTensor(),
    ])

    model = MODEL_DICT["resvae-v5"](in_channels=32, latent_dim=512)
    model.load_state_dict(
        torch.load("../log/Dec26_03:33:55_resvae-v5-BEST/resvae-v5_last_epoch.pt", map_location=device)
    )
    video_handler = SingleVideoHandler(root_dir="../data/video_frame", transform=transform)
    scene_recog_dataset = SceneRecogDataset(root_dir="../data/MemSeg_SceneRecogImg", transform=transform)

    fig = plt.figure(figsize=(16, 16), dpi=150)
    fig.suptitle("Scene Recognition")
    ax1 = fig.add_subplot(3, 2, 1, projection='3d')
    ax2 = fig.add_subplot(3, 2, 2, projection='3d')
    ax3 = fig.add_subplot(3, 2, 3, projection='3d')
    ax4 = fig.add_subplot(3, 2, 4, projection='3d')
    ax5 = fig.add_subplot(3, 2, 5, projection='3d')
    ax6 = fig.add_subplot(3, 2, 6, projection='3d')

    plot_scene_recog(ax1, model, video_handler, scene_recog_dataset, video_name="HB_1")
    plot_scene_recog(ax2, model, video_handler, scene_recog_dataset, video_name="HB_2")
    plot_scene_recog(ax3, model, video_handler, scene_recog_dataset, video_name="NB_1")
    plot_scene_recog(ax4, model, video_handler, scene_recog_dataset, video_name="NB_2")
    plot_scene_recog(ax5, model, video_handler, scene_recog_dataset, video_name="SB_1")
    plot_scene_recog(ax6, model, video_handler, scene_recog_dataset, video_name="SB_2")
    plt.savefig("result_scene_recog.png")
