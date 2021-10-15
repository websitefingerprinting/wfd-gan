import os
from os.path import join
import numpy as np
import torch
import glob
import argparse
import re
import joblib
import datetime
from functools import partial
import utils
import common as cm
from model import Generator
from extract import get_burst

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
DUMMY_CODE = 888

def parse_args():
    logger = utils.init_logger("sim-wt-gan")
    parser = argparse.ArgumentParser(description='Walkie-Talkie with GAN simulation.')
    parser.add_argument('--dir',
                        required=True,
                        help='Path of the simulated dataset.')
    parser.add_argument('--model',
                        required=True,
                        help='Path of the trained GAN generator.')
    parser.add_argument('--format',
                        metavar='<suffix of files>',
                        help='The suffix of files',
                        default=".cell")
    parser.add_argument('--n_cpu',
                        type=int,
                        default=70,
                        help='Core number for multiprocessing.')
    args = parser.parse_args()
    cf = utils.read_conf(cm.confdir)
    return args, logger, cf


class WTGAN:
    def __init__(self, mdir):
        self.scaler = None
        self.model = None
        self.seq_len = 0
        self.class_dim = 0
        self.latent_dim = 0
        self.load_model(mdir)

    def load_model(self, mdir):
        scaler_path = join(mdir, 'scaler.gz')
        model_path_list = glob.glob(join(mdir, 'generator*.ckpt'))
        assert len(model_path_list) == 1
        model_path = model_path_list[0]
        info = re.match(cm.GENERATOR_NAME_REG, model_path.split('/')[-1])
        scaler = joblib.load(scaler_path)
        model = Generator(int(info['seqlen']), int(info['cls']), int(info['latentdim']), scaler_min=scaler.data_min_[0],
                          scaler_max=scaler.data_max_[0], is_gpu=torch.cuda.is_available()).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        self.scaler = scaler
        self.model = model
        self.seq_len = int(info['seqlen'])
        self.class_dim = int(info['cls'])
        self.latent_dim = int(info['latentdim'])

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
            return np.where(synthesized_x == 0, 1, synthesized_x).tolist()


def init():
    args, logger, cf = parse_args()
    MON_SITE_NUM = int(cf['monitored_site_num'])
    MON_INST_NUM = int(cf['monitored_inst_num'])
    MON_SITE_START_IND = int(cf['monitored_site_start_ind'])
    MON_INST_START_IND = int(cf['monitored_inst_start_ind'])
    if cf['open_world'] == '1':
        UNMON_SITE_NUM = int(cf['unmonitored_site_num'])
    else:
        UNMON_SITE_NUM = 0

    # initialize output folder
    outputdir = args.dir.rstrip('/') + '_wt_sim'
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    flist = []
    for i in range(MON_SITE_START_IND, MON_SITE_START_IND + MON_SITE_NUM):
        for j in range(MON_INST_START_IND, MON_INST_START_IND + MON_INST_NUM):
            fdir = join(args.dir, '{}-{}{}'.format(i, j, args.format))
            if os.path.exists(fdir):
                flist.append(fdir)
    for i in range(UNMON_SITE_NUM):
        fdir = join(args.dir, '{}{}'.format(i, args.format))
        if os.path.exists(fdir):
            flist.append(fdir)

    return args, logger, cf, outputdir, flist


def simulate_trace(fdir, wtgan_model, outputdir):
    fname = os.path.split(fdir)[1]
    saved_dir = join(outputdir, fname)
    np.random.seed(datetime.datetime.now().microsecond)
    # np.random.seed(0)
    try:
        trace = utils.loadTrace(fdir)
        real_bursts = abs(get_burst(trace, fdir))
        # we did have some inaccuracy issue here since we cant simulate the
        # timestamps of each cell in a burst
        # we use the same timestamp for one burst here
        real_burst_ind = 0
        fake_burst_ind = 0
        merged_bursts = []
        fake_bursts = wtgan_model.get_ref_trace()
        while real_burst_ind < len(real_bursts):
            if fake_burst_ind >= len(fake_bursts):
                print("Resample a trace")
                fake_bursts = wtgan_model.get_ref_trace()
                fake_burst_ind = 0

            real_time, real_burst = real_bursts[real_burst_ind]
            fake_burst = fake_bursts[fake_burst_ind]
            real_burst_ind += 1
            fake_burst_ind += 1
            merged_bursts.append((real_time, real_burst, fake_burst))
        with open(saved_dir, 'w') as f:
            for i, burst_info in enumerate(merged_bursts):
                sign = 1 if i % 2 == 0 else -1
                time, real, fake = burst_info
                for _ in range(int(real)):
                    f.write('{:.4f}\t{:.0f}\n'.format(time, sign))
                for _ in range(int(fake)):
                    f.write('{:.4f}\t{:.0f}\n'.format(time, sign * DUMMY_CODE))
    except Exception as e:
        print("Error in {}: {}".format(fdir, e))


if __name__ == '__main__':
    args, logger, cf, outputdir, flist = init()
    logger.info('To simulate {} files, results are output to {}'.format(len(flist), outputdir))
    wtgan = WTGAN(args.model)
    with utils.poolcontext(args.n_cpu) as p:
        p.map(partial(simulate_trace, wtgan_model=wtgan, outputdir=outputdir), flist)
