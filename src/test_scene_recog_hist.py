import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from torchvision import transforms
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils import get_system_info, set_figure_options
from dataset import SingleVideoHandler, SceneRecogDataset
from model import MODEL_DICT


DATA_PATH = "../../../data"
MODEL_PATH = "../log/Dec26_03:33:55_resvae-v5-BEST/resvae-v5_last_epoch.pt"
MODEL_NAME = "resvae-v5"

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

    model = MODEL_DICT[MODEL_NAME](in_channels=32, latent_dim=512)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    video_handler = SingleVideoHandler(root_dir=os.path.join(DATA_PATH, "video_frame"), transform=transform)
    scene_recog_dataset = SceneRecogDataset(root_dir=os.path.join(DATA_PATH, "MemSeg_SceneRecogImg"), transform=transform)

    use_pickle = True

    if use_pickle:
        import pickle
        with open("dist.pkl", "rb") as f:
            dists = pickle.load(f)
    else:    
        dists = {"HB": [], "NB": [], "SB": []}
        for btype in ("HB", "NB", "SB"):
            for i in tqdm(range(30)):
                video_name = f"{btype}_{i + 1}"
                dist = get_euclidean_dist(model, video_handler, scene_recog_dataset, video_name, 3)
                dists[btype].append(dist)

    sns.histplot(dists["HB"], label="HB", kde=True, linewidth=1, edgecolor='white', binwidth=1)
    sns.histplot(dists["NB"], label="NB", kde=True, linewidth=1, edgecolor='white', binwidth=1)
    sns.histplot(dists["SB"], label="SB", kde=True, linewidth=1, edgecolor='white', binwidth=1)
    plt.legend()
    plt.show()