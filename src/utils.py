import os
import random

import numpy as np
import torch
from torch.backends import cudnn
import argparse
import matplotlib.pyplot as plt

from model import MODEL_DICT
from loss import LOSS_DICT
from optimizer import OPTIM_DICT

def set_figure_options():
    plt.rcParams['font.family'] = "Helvetica"
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['legend.frameon'] = False

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def configure_cudnn(debug=False):
    cudnn.enabled = True
    cudnn.benchmark = True
    if debug:
        cudnn.deterministic = True
        cudnn.benchmark = False

def get_systme_info():
    device = 'cpu'
    num_workers = 1

    if torch.cuda.is_available():
        device = 'cuda'
        num_workers = 4

    return torch.device(device), num_workers

def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        '--model', '-M', type=str,
        choices=tuple(MODEL_DICT.keys()), default='ae'
    )
    parser.add_argument(
        '--loss', '-L', type=str,
        choices=tuple(LOSS_DICT.keys()), default='kld_bce'
    )
    parser.add_argument(
        '--optim', '-OPT', type=str,
        choices=tuple(OPTIM_DICT.keys()), default="adam"
    )
    parser.add_argument(
        '--lr', '-LR', type=float,
        default=3e-4
    )
    parser.add_argument(
        '--weight-decay', type=float,
        default=1e-5
    )
    parser.add_argument(
        '--epochs', '-E', type=int,
        default=100
    )
    parser.add_argument(
        '--batch-size', '-B', type=int,
        default=64
    )
    parser.add_argument(
        '--eval-step', '-ES', type=int,
        default=5
    )
    parser.add_argument(
        '--img-size', nargs=2, type=int,
        default=[270, 480]
    )
    parser.add_argument(
        '--data-dir', '-DD', type=str,
        default='../data/video_frame'
    )
    parser.add_argument(
        '--seed', '-S', type=int,
        default=0
    )
    parser.add_argument(
        '--debug', type=bool,
        default=False
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)
