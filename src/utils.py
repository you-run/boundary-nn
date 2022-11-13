import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        '--model', '-M', type=str,
        choices=['ae', 'vae'], default='ae'
    )
    parser.add_argument(
        '--batch-size', '-B', type=int,
        default=64
    )
    parser.add_argument(
        '--lr', '-L', type=float,
        default=3e-4
    )
    parser.add_argument(
        '--weight-decay', '-W', type=float,
        default=1e-5
    )
    parser.add_argument(
        '--epochs', '-E', type=int,
        default=100
    )
    parser.add_argument(
        '--eval-step', '-ES', type=int,
        default=5
    )
    parser.add_argument(
        '--data-dir', '-DD', type=str,
        default='../data/video_frame'
    )
    parser.add_argument(
        '--debug', '-DB', type=bool,
        default=False
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)
