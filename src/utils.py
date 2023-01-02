import os
import random
from multiprocessing import cpu_count

import numpy as np
import torch
from torch.backends import cudnn
import argparse
import matplotlib.pyplot as plt

from model import MODEL_DICT
from loss import LOSS_DICT
from optimizer import OPTIM_DICT

def set_figure_options():
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

def get_num_workers():
    if cpu_count() > 5:
        num_workers = cpu_count() // 2
    elif cpu_count() < 2:
        num_workers = 0
    else:
        num_workers = 2
        
    return num_workers

def get_system_info():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = get_num_workers()

    return device, num_workers

def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        '--model', '-M', type=str,
        choices=tuple(MODEL_DICT.keys()), default='ae'
    )
    parser.add_argument(
        '--in-channels', '-C', type=int,
        default=64
    )
    parser.add_argument(
        '--latent-dim', '-LD', type=int,
        default=256
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
        '--eval-mode', '-EM', type=int,
        choices=(0, 1), default=0
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
        default='../../../data/video_frame'
    )
    parser.add_argument(
        '--seed', '-S', type=int,
        default=0
    )
    parser.add_argument(
        '--use-amp',
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        '--debug',
        action=argparse.BooleanOptionalAction,
        default=False
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)
