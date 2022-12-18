import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, mu, log_var):
        kld_loss = 0.5 * torch.sum(-1 - log_var + mu.pow(2) + log_var.exp())
        bce_loss = F.binary_cross_entropy(preds, targets, reduction='sum')
        return kld_loss + bce_loss


class KLDivMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, mu, log_var):
        kld_loss = 0.5 * torch.sum(-1 - log_var + mu.pow(2) + log_var.exp())
        mse_loss = F.mse_loss(preds, targets, reduction='sum')
        return kld_loss + mse_loss


LOSS_DICT = {
    'kld_bce': KLDivBCELoss,
    'kld_mse': KLDivMSELoss,
}

# 2)
# kl_divergence = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
# bce_loss = F.binary_cross_entropy(decoder_out, x)
# loss = kl_divergence + bce_loss

# 3)
# kl_loss = 0.5 * (-1 - log_var + mu.pow(2) + log_var.exp()).mean()
# mse_loss = F.binary_cross_entropy(decoder_out, x)
# loss = kl_loss + mse_loss