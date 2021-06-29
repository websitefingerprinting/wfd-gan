import pandas as pd
from functools import partial
import os
from os.path import join
import argparse
import joblib
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
    parser.add_argument('--tol',
                        default=0,
                        type=float,
                        help='A parameter of our defense.')
    parser.add_argument('--n_cpu',
                        type=int,
                        default=70,
                        help='Core number for multiprocessing.'
                        )
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
    MON_SITE_START_IND = int(cf['monitored_site_start_ind'])
    MON_INST_START_IND = int(cf['monitored_inst_start_ind'])
    if cf['open_world'] == '1':
        UNMON_SITE_NUM = int(cf['unmonitored_site_num'])
    else:
        UNMON_SITE_NUM = 0

    # initialize output folder
    outputdir = os.path.join(cm.outputdir,
                             "wfgan_{}x{}_{}_".format(MON_SITE_NUM, MON_INST_NUM, UNMON_SITE_NUM) + strftime(
                                 '%m%d_%H%M%S'))
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


class WFGAN:
    def __init__(self, tol):
        self.tol = tol
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
        model = Generator(int(info['seqlen']), int(info['cls']), int(info['latentdim'])).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        self.scaler = scaler
        self.model = model
        self.seq_len = int(info['seqlen'])
        self.class_dim = int(info['cls'])
        self.latent_dim = int(info['latentdim'])

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
        assert (self.o2i_list is not None) and (self.o2o_list is not None)
        if which == 'o2o':
            return np.random.choice(self.o2o_list)
        elif which == 'o2i':
            return np.random.choice(self.o2i_list)
        else:
            raise ValueError('Wrong option: {}'.format(which))


def adjust_burst_size(should_send, to_send, tol):
    lower_bound_send = should_send * (1 - tol)
    upper_bound_send = should_send * (1 + tol)
    if to_send < lower_bound_send:
        return np.ceil(lower_bound_send).astype(int)
    elif to_send < upper_bound_send:
        return to_send
    else:
        # to_send >= upper_bound_send
        return np.ceil(upper_bound_send).astype(int)


def process_outgoing(wfgan, last_t, outgoing, ref_outgoing):
    outgoing_defended = []
    to_send = 0
    start_t = -1
    first_in = True
    ref_ind = -1
    outgoing_timestamps = []
    while start_t < last_t or to_send > 0:
        ref_ind += 1
        if first_in:
            end_t = 0
            first_in = False
        else:
            end_t = start_t + wfgan.get_ipt('o2o')
        to_send += int(sum(outgoing[np.where((outgoing[:, 0] > start_t) & (outgoing[:, 0] <= end_t))][:, 1]))
        should_send = 1.0 * ref_outgoing[ref_ind] / cm.CELL_SIZE
        should_send = adjust_burst_size(should_send, to_send, wfgan.tol)
        if should_send <= to_send:
            to_send -= should_send
            for _ in range(should_send):
                outgoing_defended.append([end_t, 1])
        else:
            for _ in range(to_send):
                outgoing_defended.append([end_t, 1])
            for _ in range(should_send - to_send):
                outgoing_defended.append([end_t, cm.DUMMY_CODE])
            to_send = 0
        start_t = end_t
        outgoing_timestamps.append(end_t)

    outgoing_defended = np.array(outgoing_defended)
    assert len(outgoing_defended[outgoing_defended[:, 1] == 1]) == len(outgoing)
    return outgoing_defended, outgoing_timestamps


def process_incoming(wfgan, outgoing_timestamps, incoming, ref_trace):
    ref_outgoing = ref_trace[::2]
    ref_incoming = ref_trace[1::2]
    incoming_defended = []
    to_send = 0
    start_t = 0
    ref_ind = -1
    for outgoing_timestamp in outgoing_timestamps:
        # one outgoing burst corresponds to one incoming burst
        ref_ind += 1
        end_t = max(outgoing_timestamp + wfgan.get_ipt('o2i'), start_t + 0.0001)
        to_send += -int(sum(incoming[np.where((incoming[:, 0] > start_t) & (incoming[:, 0] <= end_t))][:, 1]))
        should_send = 1.0 * ref_incoming[ref_ind] / cm.CELL_SIZE
        should_send = adjust_burst_size(should_send, to_send, wfgan.tol)
        if should_send <= to_send:
            to_send -= should_send
            for _ in range(should_send):
                incoming_defended.append([end_t, -1])
        else:
            for _ in range(to_send):
                incoming_defended.append([end_t, -1])
            for _ in range(should_send - to_send):
                incoming_defended.append([end_t, -cm.DUMMY_CODE])
            to_send = 0
        start_t = end_t

    # if there are still some remaining incoming ones, make sure to pad the tail
    outgoing_t = outgoing_timestamps[-1]
    incoming_t = incoming_defended[-1][0]
    while to_send > 0:
        ref_ind += 1
        tmp = wfgan.get_ipt('o2o')
        outgoing_t = outgoing_t + tmp
        dummy_outgoing_burst_len = np.ceil(ref_outgoing[ref_ind] / cm.CELL_SIZE).astype(int)
        for _ in range(dummy_outgoing_burst_len):
            incoming_defended.append([outgoing_t, cm.DUMMY_CODE])
        incoming_t = max(outgoing_t + wfgan.get_ipt('o2i'), incoming_t + 0.0001)
        should_send = np.ceil(ref_incoming[ref_ind] / cm.CELL_SIZE).astype(int)
        if should_send <= to_send:
            to_send -= should_send
            for _ in range(should_send):
                incoming_defended.append([incoming_t, -1])
        else:
            for _ in range(to_send):
                incoming_defended.append([incoming_t, -1])
            for _ in range(should_send - to_send):
                incoming_defended.append([incoming_t, -cm.DUMMY_CODE])
            to_send = 0

    incoming_defended = np.array(incoming_defended)
    assert len(incoming_defended[incoming_defended[:, 1] == -1]) == len(incoming)
    return incoming_defended


def simulate(fdir, wfgan, outputdir):
    try:
        np.random.seed(datetime.datetime.now().microsecond)
        trace = utils.loadTrace(fdir)
        ref_trace_len = len(trace) * 4
        ref_trace = []
        while ref_trace_len >= 0:
            # get several ref trace so as to make sure that we have enough to use in the simulation
            tmp_trace = wfgan.get_ref_trace()
            ref_trace_len -= len(tmp_trace)
            ref_trace.extend(tmp_trace)
        outgoing_defended, outgoing_timestamps = process_outgoing(wfgan, trace[-1, 0], trace[trace[:, 1] > 0],
                                                                  ref_trace[::2])
        incoming_defended = process_incoming(wfgan, outgoing_timestamps, trace[trace[:, 1] < 0], ref_trace)
        trace_defended = np.concatenate((outgoing_defended, incoming_defended), axis=0)
        trace_defended = trace_defended[trace_defended[:, 0].argsort()]
        fname = fdir.split('/')[-1]
        with open(join(outputdir, fname), 'w') as f:
            for time, direction in trace_defended:
                f.write('{:.4f}\t{:.0f}\n'.format(time, direction))
    except Exception as e:
        logger.error('Error in {}: {}'.format(fdir, e))


if __name__ == '__main__':
    args, logger, cf, outputdir, flist = init()
    logger.info('To simulate {} files, results are output to {}'.format(len(flist), outputdir))
    wfgan = WFGAN(tol=args.tol)
    wfgan.load_model(args.model)
    wfgan.load_ipt(args.ipt)
    logger.debug("Model is successfully loaded.")
    with utils.poolcontext(args.n_cpu) as p:
        p.map(partial(simulate, wfgan=wfgan, outputdir=outputdir), flist)
