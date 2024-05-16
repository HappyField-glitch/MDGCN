import os
import yaml
import pickle
from scipy import io

import torch

def deal(data):
    if isinstance(data, str):
        data = [int(x) for x in data.split(',')]
    else:
        data = int(data)
    return data

def get_parser(yaml_dir):
    config = yaml.load(open(yaml_dir), Loader=yaml.FullLoader)
    config['in_dims'] = deal(config['in_dims'])
    config['out_dims'] = deal(config['out_dims'])
    config['hid_dims'] = deal(config['hid_dims'])
    config['optimizer'] = [x for x in config['optimizer'].split(',')]
    config['scheduler'] = [x for x in config['scheduler'].split(',')]
    config['active_domain_loss_step'] = deal(config['active_domain_loss_step'])
    config['alpha_weight'] = float(config['alpha_weight'])
    config['beta_weight'] = float(config['beta_weight'])
    config['gamma_weight'] = float(config['gamma_weight'])
    return config

def get_scheduler(config, optimizer):
    if config.scheduler[0] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.1, patience=int(config.scheduler[1]))
    elif config.scheduler[0] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=int(config.scheduler[1]), gamma=float(config.scheduler[2]))
    elif config.scheduler[0] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[int(x) for x in config.scheduler[2].split(' ')], gamma=float(config.scheduler[1]))
    elif config.scheduler[0] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=float(config.scheduler[1]))
    elif config.scheduler[0] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=int(config.scheduler[1]))
    elif config.scheduler[0] == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=int(config.scheduler[1]), T_mult=int(config.scheduler[2]))
    return scheduler

def get_optimizer(config, model):
    if config.optimizer[0] == 'SGD':
        optimizer = torch.optim.SGD(params=model.get_optimizer(), momentum=float(config.optimizer[1]), weight_decay=float(config.optimizer[2]))
    elif config.optimizer[0] == 'Adam':
        optimizer = torch.optim.Adam(params=model.get_optimizer(), betas=(float(config.optimizer[1]), float(config.optimizer[2])), weight_decay=float(config.optimizer[3]))
    else:
        print('Just support SGD and Adam ...')
    return optimizer
