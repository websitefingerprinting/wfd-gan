import torch
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import pandas as pd
import logging
import configparser
import common as cm


'''For common usage'''
def init_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter(cm.LOG_FORMAT)
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger


def read_conf(file):
    cf = configparser.ConfigParser()
    cf.read(file)
    return dict(cf['default'])


'''For GAN'''
def compute_gradient_penalty(D, real_samples, fake_samples, c, Tensor):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, c)
    # d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean(0)
    return gradient_penalty


def loadDataset(dir):
    '''Load Dataset from npy file, normalize it and change it to tensor'''
    '''x: (num x length)
       y: (num, )'''
    data = np.load(dir, allow_pickle=True).item()
    X, y = data['feature'], data['label']
    return X, y

def loadTrace(fdir):
    with open(fdir,'r') as f:
        tmp = f.readlines()
    trace = np.array(pd.Series(tmp).str.slice(0,-1).str.split("\t",expand=True).astype(float))
    return trace