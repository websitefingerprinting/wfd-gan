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

DUMMY_CODE = 888 # used to mark dummy packet


def parse_args():
    """initialize logger"""
    logger = utils.init_logger("syn")
    '''read in arg'''
    parser = argparse.ArgumentParser(description='Simulation of using synthesized traces as a tunnel to transport data.')
    parser.add_argument('--dir',
                        required=True,
                        help='Path of the dataset to be defended.')
    parser.add_argument('--decoy',
                        required=True,
                        help='Path of the decoy dataset')
    parser.add_argument('--format',
                        metavar='<suffix of files>',
                        help='The suffix of files',
                        default=".cell")
    args = parser.parse_args()
    cf = utils.read_conf(cm.confdir)
    return args, logger, cf



def parallel(flist, n_jobs=20):
    pool = mp.Pool(n_jobs)
    pool.map(tunnel_defense, flist)


def choose_a_tunnel_trace(label):
    global decoy_dir, MON_SITE_NUM, format
    if label == MON_SITE_NUM:
        # Unmon site, choose a monitored one
        decoy_list = glob.glob(join(decoy_dir,"*-*"+format))
    else:
        # mon site, choose a decoy from different classes
        decoy_list = list(set(glob.glob(join(decoy_dir, "*"+format))) - set(glob.glob(join(decoy_dir,"{}-*".format(label)+format))))
    decoy_fdir = np.random.choice(decoy_list)
    # logger.debug("Choose decoy {}".format(decoy_fdir))
    tunnel_trace = utils.loadTrace(decoy_fdir)
    return tunnel_trace



def tunnel_defense(fdir):
    # logger.debug("Similate {}".format(fdir))
    np.random.seed(datetime.datetime.now().microsecond)
    # Optimistic simulation
    global outputdir, MON_SITE_NUM
    trace = utils.loadTrace(fdir)
    fname = os.path.split(fdir)[1]
    if '-' in fname:
        label = int(fname.split("-")[0])
    else:
        label = MON_SITE_NUM
    original_trace = utils.loadTrace(fdir)
    original_out_trace = original_trace[original_trace[:,1]>0][:,0]
    original_in_trace = original_trace[original_trace[:,1]<0][:,0]
    defended_trace = []
    stop = 0
    base_t = 0
    while len(original_out_trace) > 0 or len(original_in_trace) > 0:
        tunnel_trace = choose_a_tunnel_trace(label)
        for pkt in tunnel_trace:
            t, dir = pkt[0], pkt[1]
            cur_t = t + base_t
            if dir>0:
                # need an outgoing packet
                if len(original_out_trace) == 0 or original_out_trace[0] > cur_t:
                    # no data to send or has not come yet
                    defended_trace.append([cur_t, DUMMY_CODE])
                else:
                    assert original_out_trace[0] <= cur_t
                    defended_trace.append([cur_t, 1])
                    original_out_trace = np.delete(original_out_trace,0)
            else:
                # need an incoming packet
                if len(original_in_trace) == 0 or original_in_trace[0] > cur_t:
                    defended_trace.append([cur_t, -DUMMY_CODE])
                else:
                    assert original_in_trace[0] <= cur_t
                    defended_trace.append([cur_t,-1])
                    original_in_trace = np.delete(original_in_trace, 0)
            if len(original_out_trace) ==0 and len(original_in_trace) == 0:
                # no data to send, after transmit this decoy trace, stop
                stop = 1
                break
        if stop:
            break
        base_t = defended_trace[-1][0]
    assert len(original_in_trace) == 0 and len(original_out_trace) == 0
    with open(join(outputdir, fname), 'w') as f:
        for pkt in defended_trace:
            f.write("{:.4f}\t{}\n".format(pkt[0], pkt[1]))


if __name__ == '__main__':
    global outputdir, decoy_dir, MON_SITE_NUM, format

    args, logger, cf = parse_args()
    format = args.format
    decoy_dir = args.decoy

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
    outputdir = os.path.join(cm.outputdir, "tunnel_{}x{}_{}_".format(MON_SITE_NUM,MON_INST_NUM,UNMON_SITE_NUM)+strftime('%m%d_%H%M%S'))
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
    parallel(flist)
