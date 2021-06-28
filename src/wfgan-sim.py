import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
import os
from os.path import join
import argparse
import joblib
import logging
import glob
import re
import datetime
from time import strftime
import utils
import common as cm
from model import *

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def parse_args():
    logger = utils.init_logger("wfdgan")
    parser = argparse.ArgumentParser(description='Simulation of wfgan.')
    parser.add_argument('--dir',
                        required=True,
                        help='Path of the dataset to be defended.')
    parser.add_argument('--model',
                        required=True,
                        help='Path of the trained GAN model. Only provide the folder which contains the trained '
                             'models and data scaler.')
    parser.add_argument('--ipt',
                        required=True,
                        help='Path of ipt info.')
    parser.add_argument('--format',
                        metavar='<suffix of files>',
                        help='The suffix of files',
                        default=".cell")
    args = parser.parse_args()

    cf = utils.read_conf(cm.confdir)

    return args, logger, cf


def init():
    args, logger, cf = parse_args()
    MON_SITE_NUM = int(cf['monitored_site_num'])
    MON_INST_NUM = int(cf['monitored_inst_num'])
    MON_INST_START_IND = int(cf['monitored_inst_start_ind'])
    if cf['open_world'] == '1':
        UNMON_SITE_NUM = int(cf['unmonitored_site_num'])
        OPEN_WORLD = 1
    else:
        OPEN_WORLD = 0
        UNMON_SITE_NUM = 0

    # initialize output folder
    outputdir = os.path.join(cm.outputdir,
                             "wfgan_{}x{}_{}_".format(MON_SITE_NUM, MON_INST_NUM, UNMON_SITE_NUM) + strftime(
                                 '%m%d_%H%M%S'))
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    flist = []
    for i in range(MON_SITE_NUM):
        for j in range(MON_INST_START_IND, MON_INST_START_IND + MON_INST_NUM):
            fdir = join(args.dir, '{}-{}{}'.format(i, j, args.format))
            if os.path.exists(fdir):
                flist.append(fdir)
    for i in range(UNMON_SITE_NUM):
        fdir = join(args.dir, '{}{}'.format(i, args.format))
        if os.path.exists(fdir):
            flist.append(fdir)

    return args, logger, cf, outputdir, flist


class WFGAN:
    def __init__(self):
        self.scaler = None
        self.model = None
        self.seq_len = 0
        self.class_dim = 0
        self.latent_dim = 0
        self.o2o_list = None
        self.o2i_list = None

    def load_model(self, mdir):
        scaler_path = join(mdir, 'scaler.gz')
        model_path_list = glob.glob(join(mdir, 'generator*.ckpt'))
        assert len(model_path_list) == 1
        model_path = model_path_list[0]
        info = re.match(cm.GENERATOR_NAME_REG, model_path.split('/')[-1])
        scaler = joblib.load(scaler_path)
        model = Generator(info['seqlen'], info['cls'], info['latentdim']).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        self.scaler = scaler
        self.model = model
        self.seq_len = info['seqlen']
        self.class_dim = info['cls']
        self.latent_dim = info['latentdim']

    def load_ipt(self, ipt_dir):
        with open(join(ipt_dir, 'o2o.txt'), 'r') as f:
            o2o_list = pd.Series(f.readlines()).str.slice(0, -1).astype(float)
        self.o2o_list = np.array(o2o_list)
        with open(join(ipt_dir, 'o2i.txt'), 'r') as f:
            o2i_list = pd.Series(f.readlines()).str.slice(0, -1).astype(float)
        self.o2i_list = np.array(o2i_list)

    def get_ref_trace(self):
        assert self.model
        assert self.scaler

        c_ind = np.random.randint(self.class_dim)
        self.model.eval()
        with torch.no_grad():
            z = np.random.randn(1, self.latent_dim).astype('float32')
            z = torch.from_numpy(z).to(device)
            c = torch.zeros(1, self.class_dim)
            c[:, c_ind] = 1
            c = c.to(device)
            synthesized_x = self.model(z, c).cpu().numpy()
            synthesized_x = self.scaler.inverse_transform(synthesized_x).flatten()
            length = min(int(synthesized_x[0]), self.seq_len - 1)
            if length % 2 != 0:
                length -= 1
            assert length % 2 == 0

            synthesized_x = np.round(synthesized_x[1:1 + length]).astype(int)
            return list(synthesized_x)

    def get_ipt(self, which='o2o'):
        assert self.o2i_list and self.o2o_list
        if which == 'o2o':
            return np.random.choice(self.o2o_list)
        elif which == 'o2i':
            return np.random.choice(self.o2i_list)
        else:
            raise ValueError('Wrong option: {}'.format(which))


def process_outgoing(wfgan, last_t, outgoing, ref_outgoing):
    outgoing_defended = []
    to_send = 0
    start_t = -1
    first_in = True
    ref_ind = -1
    while start_t < last_t:
        ref_ind += 1
        if first_in:
            end_t = 0
            first_in = False
        else:
            end_t = start_t + wfgan.get_ipt('o2o')
        to_send = sum(outgoing[np.where((outgoing[:, 0] > start_t) & (outgoing[:, 0] <= end_t))][:, 1])
        need = np.ceil(ref_outgoing[ref_ind] / cm.CELL_SIZE).astype(int)
        if need <= to_send:
            to_send -= need
            for _ in range(need):
                outgoing_defended.append([start_t, 1])
        else:
            for _ in range(to_send):
                outgoing_defended.append([start_t, 1])
            for _ in range(need - to_send):
                outgoing_defended.append([start_t, cm.DUMMY_CODE])
        start_t = end_t
    outgoing_defended = np.array(outgoing_defended)
    assert len(outgoing_defended[outgoing_defended[:, 1] == 1]) == len(outgoing)
    return outgoing_defended


def process_incoming(wfgan, outgoing_defended, incoming, ref_incoming):
    incoming_defended = []
    to_send = 0
    start_t = 0
    ref_ind = -1
    for i in range(len(outgoing_defended)):
        ref_ind += 1
        end_t = outgoing_defended[i] + wfgan.get_ipt('o2i')
        to_send = sum(incoming[np.where((incoming[:, 0] > start_t) & (incoming[:, 0] <= end_t))][:, 1])
        need = np.ceil(ref_incoming[ref_ind] / cm.CELL_SIZE).astype(int)
        if need <= to_send:
            to_send -= need
            for _ in range(need):
                incoming_defended.append([end_t, -1])
        else:
            for _ in range(to_send):
                incoming_defended.append([end_t, -1])
            for _ in range(need - to_send):
                incoming_defended.append([end_t, -cm.DUMMY_CODE])
        start_t = end_t

    incoming_defended = np.array(incoming_defended)
    assert len(incoming_defended[incoming_defended[:, 1] == -1]) == len(incoming)
    return incoming_defended


def simulate(fdir, wfgan, outputdir):
    np.random.seed(datetime.datetime.now().microsecond)
    trace = utils.loadTrace(fdir)
    ref_trace_len = len(trace) * 5
    ref_trace = []
    while ref_trace_len > 0:
        # get several ref trace so as to make sure that we have enough to use in the simulation
        tmp_trace = wfgan.get_ref_trace()
        ref_trace_len -= len(tmp_trace)
        ref_trace.extend(ref_trace)
    outgoing_defended = process_outgoing(wfgan, trace[-1, 0], trace[trace[:, 1] > 0], ref_trace[::2])
    incoming_defended = process_incoming(wfgan, outgoing_defended, trace[trace[:, 1] < 0], ref_trace[1::2])
    trace_defended = np.concatenate((outgoing_defended, incoming_defended), axis=0)
    trace_defended = trace_defended[trace_defended[:, 0].argsort()]
    fname = fdir.split('/')[-1]
    with open(join(outputdir, fname), 'w') as f:
        for time, direction in trace_defended:
            f.write('{:.4f}\t{:.0f}\n'.format(time, direction))


if __name__ == '__main__':
    args, logger, cf, outputdir, flist = init()
    logger.info('To simulate {} files, results are output to {}'.format(len(flist), outputdir))
    wfgan = WFGAN()
    wfgan.load_model(args.model)
    wfgan.load_ipt(args.ipt)
    logger.debug("Model is successfully loaded.")
    with utils.poolcontext(60) as p:
        p.map(partial(simulate, wfgan=wfgan, outputdir=outputdir), flist)
