import torch.nn as nn
import torch.optim as optim

def get_criterion(configs):
    if configs.criterion == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()

def get_lr_scheduler(optimizer, configs):
    if configs.lr_scheduler == "CosineAnnealingWarmRestarts":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            configs.T_0,
            configs.T_mult,
            configs.eta_min
        )
    elif configs.lr_scheduler == "ExponentialLR":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=configs.gamma)

def get_optimizer(params, configs):
    if configs.optimizer == "SGD":
        return optim.SGD(params=params, lr=configs.lr, momentum=configs.momentum)
    elif configs.optimizer == "Adam":
        return optim.Adam(params=params, lr=configs.lr, betas=configs.betas)