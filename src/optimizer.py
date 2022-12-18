import torch

OPTIM_DICT = {
    'sgd': torch.optim.SGD,
    'adagrad': torch.optim.Adagrad,
    'rmsprop': torch.optim.RMSprop,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW
}