import argparse
import common as cm
import utils
from model import *
import os
from os.path import join
from sklearn import preprocessing
import numpy as np
import multiprocessing as mp
import pandas as pd
import datetime
from time import strftime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

debug_file = 0

def parse_args():
    # initialize logger
    logger = utils.init_logger("def")
    # read in args
    parser = argparse.ArgumentParser(description='Create defended trace based on trained GAN.')
    parser.add_argument('--train',
                        required=True,
                        help='Train dataset path (saved npy file).')
    parser.add_argument('--test',
                        required=True,
                        help='Test dataset directory (raw traces).')
    parser.add_argument('--model', '-m',
                        required=True,
                        help='Path to the saved model.')
    parser.add_argument("--latent_dim", type=int, default=16, help="dimensionality of the latent space")
    parser.add_argument('--format',
                        metavar='<suffix of files>',
                        help='The suffix of files',
                        default=".cell")
    args = parser.parse_args()
    cf = utils.read_conf(cm.confdir)
    return args, logger, cf


def parallel(flist, n_jobs=20):
    pool = mp.Pool(n_jobs)
    data_dict = pool.map(defend, flist)
    return data_dict


def sample(c_ind):
    global model, scaler, latent_dim, class_dim
    n = 1
    model.eval()
    with torch.no_grad():
        z = np.random.randn(n, latent_dim).astype('float32')
        z = torch.from_numpy(z).to(device)
        c = torch.zeros(n, class_dim)
        c[:,c_ind] = 1
        c = c.to(device)
        synthesized_x = model(z, c).cpu().numpy()
        synthesized_x = synthesized_x.mean(axis=0).reshape(1,-1)
        synthesized_x = scaler.inverse_transform(synthesized_x).flatten()
        synthesized_x = np.round(np.trim_zeros(synthesized_x)).astype(int)
        synthesized_x[synthesized_x==0] = 1 # make sure each burst size > 0
        assert (synthesized_x>0).all()
    return synthesized_x

def extract(x):
    '''
    extract burst sequences from raw traces, ignore sign since we assume a positive comes after a negative burst.
    A normal loading starts from outgoing (positive) bursts.
    If encounter incoming (negative) ones first, must be fragment and remove them.
    '''
    start = -1
    for pkt in x:
        start += 1
        if pkt > 0:
            break
    x = x[start:]
    new_x = []
    sign = np.sign(x[0])
    cnt = 0
    for e in x:
        if np.sign(e) == sign:
            cnt += 1
        else:
            new_x.append( cnt)
            cnt = 1
            sign = np.sign(e)
    new_x.append( cnt )
    return new_x

def choose_morphed_class(label):
    global  MON_SITE_NUM
    if label < MON_SITE_NUM:
        return MON_SITE_NUM
    else:
        return np.random.choice(range(MON_SITE_NUM))

def defend(fdir):
    global outputdir, class_dim, format, MON_SITE_NUM
    np.random.seed(datetime.datetime.now().microsecond)
    with open(fdir,'r') as f:
        tmp = pd.Series(f.readlines()).str.slice(0,-1).str.split("\t",expand=True)
    trace = np.array(tmp)[:,1].astype(int)
    x= extract(trace)
    fname = fdir.split("/")[-1].split(format)[0]
    if '-' in fname:
        label = int(fname.split('-')[0])
    else:
        label = MON_SITE_NUM
    c_ind = choose_morphed_class(label)
    logger.debug("Label:{}, chosen label:{}".format(label, c_ind))

    synthesized_x = sample(c_ind)
    while len(synthesized_x) < len(x):
        assert c_ind != label
        logger.debug("synthesized_x {} is shorter than x {}, chosen label:{}".format(len(synthesized_x),len(x), c_ind))
        c_ind = choose_morphed_class(label)
        synthesized_x = np.concatenate((synthesized_x, sample(c_ind)))
    synthesized_x = synthesized_x[:len(x)]

    assert x[0] > 0
    with open(join(outputdir, fname+format),'w') as f:
        for i in range(0, len(x)):
            sign = 1 if i%2 == 0 else -1  # Assume outgoing bursts first
            cur_t = i
            for _ in range(x[i]):
                f.write("{:d}\t{:d}\n".format(cur_t, int(sign*1)))
            for _ in range(synthesized_x[i]-x[i]):
                f.write("{:d}\t{:d}\n".format(cur_t, int(sign*888)))
    if debug_file:
        with open(join(outputdir, fname+".debug"),"w") as f:
            for i in range(0,len(x)):
                f.write("{}\t{}\n".format(x[i],synthesized_x[i]))

if __name__ == '__main__':
    global outputdir, model, scaler, class_dim, latent_dim, format, MON_SITE_NUM

    args, logger, cf = parse_args()
    latent_dim = args.latent_dim
    format = args.format

    # initialize output folder
    outputdir = os.path.join(cm.outputdir, args.test.rstrip('/').split("/")[-1], "defended_trace")
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # load train data
    data = np.load(args.train, allow_pickle=True).item()
    X, y = data['feature'], data['label']
    seq_len = X.shape[1]
    class_dim = max(y)+1
    assert class_dim > 1
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X)
    logger.info("Scaler prepared.")

    # load model
    model = Generator(seq_len, class_dim, args.latent_dim).to(device)
    logger.info(seq_len, class_dim, args.latent_dim)
    model.load_state_dict(torch.load(args.model, map_location=device))
    logger.info("Model loaded.")

    # load test data
    MON_SITE_NUM = int(cf['monitored_site_num'])
    MON_INST_NUM = int(cf['monitored_inst_num'])
    if cf['open_world'] == '1':
        UNMON_SITE_NUM = int(cf['unmonitored_site_num'])
        OPEN_WORLD = 1
    else:
        OPEN_WORLD = 0
        UNMON_SITE_NUM = 0
    flist = []
    for i in range(MON_SITE_NUM):
        for j in range(MON_INST_NUM):
            fdir = join(args.test, str(i)+'-'+str(j)+args.format)
            if os.path.exists(fdir):
                flist.append(fdir)
    for i in range(UNMON_SITE_NUM):
        fdir = join(args.test, str(i)+args.format)
        if os.path.exists(fdir):
            flist.append(fdir)

    parallel(flist)
