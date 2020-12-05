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
import glob
import re
import joblib

DUMMY_CODE = 888 # used to mark dummy packet
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name_pattern = "generator_seqlen([0-9]*)_cls([0-9]*)_latentdim([0-9]*).ckpt"

def parse_args():
    """initialize logger"""
    logger = utils.init_logger("syn")
    '''read in arg'''
    parser = argparse.ArgumentParser(description='Using Gan to generate decoy bursts for Walkie-Talkie')
    parser.add_argument('--dir',
                        required=True,
                        help='Path of the dataset to be defended.')
    parser.add_argument('--model',
                        required=True,
                        type=str,
                        help="Directory of the trained model and scaler")
    parser.add_argument('--format',
                        metavar='<suffix of files>',
                        help='The suffix of files',
                        default=".cell")
    args = parser.parse_args()
    cf = utils.read_conf(cm.confdir)
    return args, logger, cf



def parallel(flist, n_jobs=20):
    pool = mp.Pool(n_jobs)
    pool.map(sim_wt, flist)


def syn_trace(label):
    global scaler, model, latent_dim, cls_num, MON_SITE_NUM
    # for mon, syn an unmon and vice-versa
    if label < MON_SITE_NUM:
        c_ind = MON_SITE_NUM
    else:
        c_ind = np.random.randint(MON_SITE_NUM)

    model.eval()
    with torch.no_grad():
        z = np.random.randn(1, latent_dim).astype('float32')
        z = torch.from_numpy(z).to(device)
        c = torch.zeros(1, cls_num)
        c[:, c_ind] = 1
        c = c.to(device)
        synthesized_x = model(z, c).cpu().numpy()
        synthesized_x = scaler.inverse_transform(synthesized_x).flatten()
        length = int(synthesized_x[0])
        synthesized_x =  synthesized_x[1:1+length].astype(int)
        synthesized_x = np.trim_zeros(synthesized_x,trim='b')
    return synthesized_x


def sim_wt(fdir):
    # logger.debug("Similate {}".format(fdir))
    np.random.seed(datetime.datetime.now().microsecond)
    global outputdir, MON_SITE_NUM
    fname = os.path.split(fdir)[1]
    if '-' in fname:
        label = int(fname.split("-")[0])
    else:
        label = MON_SITE_NUM
    original_trace = utils.loadTrace(fdir)
    syn_burst = syn_trace(label)
    # skip the first few incoming packets
    res_trace = []
    start_ind = -1
    for pkt in original_trace:
        start_ind += 1
        if pkt[1] > 0:
            break
        res_trace.append(pkt)

    cur_sign = 1
    cur_time = original_trace[start_ind,0]
    for i, pkt in enumerate(original_trace[start_ind:]):
        if np.sign(pkt[1]) == cur_sign:
            res_trace.append(pkt)
            cur_time = pkt[0]
        else:
            # pad fake burst
            if len(syn_burst) > 0 :
                dummy_num = syn_burst[0]
                for _ in range(dummy_num):
                    res_trace.append([cur_time, cur_sign*DUMMY_CODE])
            syn_burst = syn_burst[1:]

            cur_sign = np.sign(pkt[1])
            cur_time = pkt[0]
            res_trace.append(pkt)
    if len(syn_burst) > 0:
        dummy_num = syn_burst[0]
        for _ in range(dummy_num):
            res_trace.append([cur_time,cur_sign*DUMMY_CODE])
    res_trace_np = np.array(res_trace)
    assert len(np.where(abs(res_trace_np[:,1]) == 1)[0]) == len(original_trace)

    with open(join(outputdir, fname),"w") as f:
        for pkt in res_trace:
            f.write("{:.4f}\t{:.0f}\n".format(pkt[0],pkt[1]))



if __name__ == '__main__':
    global outputdir, MON_SITE_NUM, format, model, scaler, latent_dim, cls_num

    args, logger, cf = parse_args()
    format = args.format

    # load model
    model_dir = glob.glob(join(args.model,"generator*.ckpt"))[0]
    pattern = re.compile(model_name_pattern)
    m = pattern.match(model_dir.split("/")[-1])
    seqlen, cls_num, latent_dim = int(m.group(1)), int(m.group(2)), int(m.group(3))
    model = Generator(seqlen, cls_num, latent_dim).to(device)
    model.load_state_dict(torch.load(model_dir, map_location=device))

    scaler_dir = glob.glob(join(args.model, "scaler.gz"))[0]
    scaler = joblib.load(scaler_dir)
    logger.info("Model loaded.")

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
    outputdir = os.path.join(cm.outputdir, "wt_{}x{}_{}_".format(MON_SITE_NUM,MON_INST_NUM,UNMON_SITE_NUM)+strftime('%m%d_%H%M%S'))
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    flist = []
    for i in range(MON_SITE_NUM):
        for j in range(MON_INST_NUM):
            fdir = join(args.dir, str(i)+'-'+str(j)+args.format)
            flist.append(fdir)
    for i in range(UNMON_SITE_NUM):
        fdir = join(args.dir, str(i)+args.format)
        flist.append(fdir)
    logger.info("Outputdir:{}, generate {} traces.".format(outputdir, len(flist)))
    parallel(flist[:2])
    # for f in flist[:1]:
    #     sim_wt(f)