import argparse
import common as cm
from model import *
import os
from os.path import join
from sklearn import preprocessing
import numpy as np
import multiprocessing as mp
import datetime
from time import strftime
import utils

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

latent_dim = 100 # The latent space of the generator


class IPT_SAMPLER:
    def __init__(self, cdfs):
        self.cdfs = cdfs # 4 cdf: o2o, o2i, i2o, i2i

    def sample(self, cdf_type):
        # sample a random number from one type of distribution
        if cdf_type == 'o2o' or cdf_type == 0:
            return self.sample_(0)
        elif cdf_type == 'o2i' or cdf_type == 1:
            return self.sample_(1)
        elif cdf_type == 'i2o' or cdf_type == 2:
            return self.sample_(2)
        elif cdf_type == 'i2i' or cdf_type == 3:
            return self.sample_(3)
        else:
            raise ValueError("Wrong cdf type:{}".format(cdf_type))
            return None

    def sample_(self, cdf_index):
        x, cdf = self.cdfs[cdf_index]
        y = np.random.uniform(0, 1)
        y0, y1 = cdf[cdf <= y][-1], cdf[cdf >= y][0]
        x0, x1 = x[cdf <= y][-1], x[cdf >= y][0]
        if y0 == y1:
            return 10 ** ((x0 + x1) / 2)
        else:
            return 10 ** ((y - y0) / (y1 - y0) * (x1 - x0) + x0)


def parse_args():
    """initialize logger"""
    logger = utils.init_logger("syn")
    '''read in arg'''
    parser = argparse.ArgumentParser(description='Generate synthesized traces based on trained GAN.')
    parser.add_argument('--model',
                        required=True,
                        help='Path of the trained GAN generator.')
    parser.add_argument('--train',
                        required=True,
                        help='The training data, used to reproduce the scaler (saved npy file).')
    parser.add_argument('--cdf',
                        required=True,
                        help='Path of cdf of ipt.')
    parser.add_argument('--format',
                        metavar='<suffix of files>',
                        help='The suffix of files',
                        default=".cell")
    args = parser.parse_args()
    cf = utils.read_conf(cm.confdir)
    return args, logger, cf


def parallel(flist, n_jobs=20):
    pool = mp.Pool(n_jobs)
    pool.map(syn, flist)


def sample_trace(c_ind):
    global model, scaler, latent_dim, class_dim
    n = 1
    model.eval()
    with torch.no_grad():
        z = np.random.randn(n, latent_dim).astype('float32')
        z = torch.from_numpy(z).to(device)
        c = torch.zeros(n, class_dim)
        c[:,c_ind] = 1
        c = c.to(device)
        recon_x = model(z, c).cpu().numpy()
        recon_x = recon_x.mean(axis=0).reshape(1,-1)
        recon_x = scaler.inverse_transform(recon_x)
        recon_x = np.round(recon_x).astype(int).flatten()
    return np.trim_zeros(recon_x)

def expand(trace):
    # 1,2,4,2 -> 1,-1,-1, 1, 1, ...
    expanded_trace = []
    for i, burst in enumerate(trace):
        if i%2 == 0:
            sign = 1
        else:
            sign = -1
        if burst == 0:
            # sometimes you generated a burst of size 0
            # correctify it as at least 1
            burst = 1
        for j in range(burst):
            expanded_trace.append(sign)
    return expanded_trace


def syn(fdir):
    global sampler, MON_SITE_NUM
    np.random.seed(datetime.datetime.now().microsecond)
    fname = os.path.split(fdir)[1].split(format)[0]
    if '-' in fname:
        label = int(fname.split('-')[0])
    else:
        label = MON_SITE_NUM
    sampled_trace = sample_trace(label)
    trace = expand(sampled_trace)
    t = 0.0
    syn_trace = [[t,1]]
    for i in range(len(trace)):
        if i == 0:
            continue
        cur_pkt = trace[i]
        last_pkt = trace[i-1]
        if cur_pkt>0 and last_pkt>0:
            cdf_type = 'o2o'
        elif cur_pkt>0 and last_pkt<0:
            cdf_type = 'i2o'
        elif cur_pkt<0 and last_pkt>0:
            cdf_type = 'o2i'
        elif cur_pkt<0 and last_pkt<0:
            cdf_type = 'i2i'
        ipt = sampler.sample(cdf_type)
        # print("Fname:{} Current idx:{}, sampled ipt:{:.6f}, cdf_type:{}, pktseq:{}->{}".format(fname, i, ipt, cdf_type,last_pkt, cur_pkt ))

        assert ipt >= 0
        t += ipt
        syn_trace.append([t,cur_pkt])
    with open(fdir, 'w') as f:
        for pkt in syn_trace:
            f.write("{:.4f}\t{}\n".format(pkt[0], int(pkt[1])))


if __name__ == '__main__':
    global model, scaler, class_dim, MON_SITE_NUM, sampler

    args, logger, cf = parse_args()
    format = args.format

    # load train data
    data = np.load(args.train, allow_pickle=True).item()
    X, y = data['feature'], data['label']
    seq_len = X.shape[1]
    class_dim = y.max() +1
    logger.info("X shape {}, y shape {}, class num: {}".format(X.shape, y.shape, class_dim))
    assert class_dim > 1
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X)
    logger.info("Prepare scaler.")

    # load model
    model = Generator(seq_len, class_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    logger.info("Model loaded.")

    # load ipt cdf
    cdfs = np.load(args.cdf)
    sampler = IPT_SAMPLER(cdfs)

    # Generate size
    MON_SITE_NUM = int(cf['monitored_site_num'])
    MON_INST_NUM = int(cf['monitored_inst_num'])
    if cf['open_world'] == '1':
        UNMON_SITE_NUM = int(cf['unmonitored_site_num'])
        OPEN_WORLD = 1
    else:
        OPEN_WORLD = 0
        UNMON_SITE_NUM = 0

    # initialize output folder
    outputdir = os.path.join(cm.outputdir, "syn_{}x{}_{}_".format(MON_SITE_NUM,MON_INST_NUM,UNMON_SITE_NUM)+strftime('%m%d_%H%M%S'))
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    flist = []
    for i in range(MON_SITE_NUM):
        for j in range(MON_INST_NUM):
            fdir = join(outputdir, str(i)+'-'+str(j)+args.format)
            flist.append(fdir)
    for i in range(UNMON_SITE_NUM):
        fdir = join(outputdir, str(i)+args.format)
        flist.append(fdir)
    logger.info("Outputdir:{}, generate {} traces".format(outputdir, len(flist)))
    parallel(flist)
