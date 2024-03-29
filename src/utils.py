import os
from os.path import join

import torch
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import pandas as pd
import glob
import logging
import multiprocessing as mp
from os.path import join
from contextlib import contextmanager
import configparser
import joblib
import re

from model import *
import common as cm


'''For common usage'''


def init_logger(name, log_dir=None):
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

    if log_dir is not None:
        ch2 = logging.StreamHandler(open(log_dir, 'w'))
        ch2.setFormatter(formatter)
        logger.addHandler(ch2)
    return logger


def init_directory(dir, tag=""):
    basedir = os.path.split(os.path.split(dir)[0])[0]
    modeldir = join(basedir, tag, 'model')
    checkpointdir = join(basedir, tag, 'checkpoint')
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    if not os.path.exists(checkpointdir):
        os.makedirs(checkpointdir)
    return join(basedir, tag), modeldir, checkpointdir


def read_conf(file, sec='default'):
    cf = configparser.ConfigParser()
    cf.read(file)
    return dict(cf[sec])


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


# deprecated function
def loadDataset(dir):
    '''Load Dataset from npy file, normalize it and change it to tensor'''
    '''x: (num x length)
       y: (num, )'''
    data = np.load(dir, allow_pickle=True).item()
    X, y = data['feature'], data['label']
    return X, y


def load_dataset(dir):
    """Load Dataset from npz file"""
    '''x: (num x length)
       y: (num, )'''
    data = np.load(dir)
    X, y = data['features'], data['labels']
    return X, y


# deprecated
def prepare_dataset(features, labels, config):
    """Pick data from a specific config
    Pick 0 ~ MON_SITE_NUM-1 classes, each one pick MON_INST_NUM instances,
    index starting from MON_INST_START_IND
    """
    MON_SITE_NUM = int(config['monitored_site_num'])
    MON_INST_NUM = int(config['monitored_inst_num'])
    MON_INST_START_IND = int(config['monitored_inst_start_ind'])
    assert (MON_SITE_NUM > 0 and MON_INST_NUM > 0 and MON_INST_START_IND >= 0)
    X, y = [], []
    for mon_site_ind in range(MON_SITE_NUM):
        tmp = labels[labels == mon_site_ind]
        labels_class = list(labels[labels == mon_site_ind][MON_INST_START_IND:MON_INST_START_IND + MON_INST_NUM])
        features_class = list(features[labels == mon_site_ind][MON_INST_START_IND:MON_INST_START_IND + MON_INST_NUM])
        X.extend(features_class)
        y.extend(labels_class)
    return np.array(X), np.array(y)


def loadTrace(fdir, norm_cell=False):
    with open(fdir, 'r') as f:
        tmp = f.readlines()
    trace = np.array(pd.Series(tmp).str.slice(0, -1).str.split("\t", expand=True).astype(float))
    if norm_cell:
        # in case we feed in defended traces that
        # use +-888 to represent a dummy packet
        trace[:, 1] = np.sign(trace[:, 1])
    return trace


@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def load_log_ipt(fdir):
    """Load .ipt file to a list
    The default format is one line comment + std + data
    The returned is the kernel std + ipt in log scale
    """
    with open(fdir, 'r') as f:
        lines = f.readlines()[1:]
        data = np.array(pd.Series(lines).str.slice(0, -1)).astype(float)
    return data
