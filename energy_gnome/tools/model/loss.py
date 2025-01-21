import torch


def loss_MAPE(output, target):
    # MAPE loss
    return torch.mean(torch.abs((target - output) / target))
