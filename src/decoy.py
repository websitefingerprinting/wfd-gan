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

DUMMY_CODE = 888 # used to mark dummy packet


def parse_args():
    """initialize logger"""
    logger = utils.init_logger("syn")
    '''read in arg'''
    parser = argparse.ArgumentParser(description='Simulation of Decoy defense using synthesized traces.')
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
    pool.map(decoy_defense, flist)


def decoy_defense(fdirs):
    global outputdir
    fdir, decoydir = fdirs[0], fdirs[1]
    logger.info("{} {}".format(fdir,decoydir))
    fname = os.path.split(fdir)[1]

    trace, decoy_trace = utils.loadTrace(fdir), utils.loadTrace(decoydir)
    decoy_trace[:,1] *= DUMMY_CODE
    defended_trace = np.concatenate((trace,decoy_trace),axis=0)
    defended_trace = defended_trace[defended_trace[:, 0].argsort(kind="mergesort")]

    with open(join(outputdir, fname), 'w') as f:
        for pkt in defended_trace:
            f.write("{:.4f}\t{}\n".format(pkt[0], int(pkt[1])))


if __name__ == '__main__':
    global outputdir

    args, logger, cf = parse_args()
    format = args.format

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
    outputdir = os.path.join(cm.outputdir, "decoy_{}x{}_{}_".format(MON_SITE_NUM,MON_INST_NUM,UNMON_SITE_NUM)+strftime('%m%d_%H%M%S'))
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    flist = []
    decoy_unmon = np.random.permutation(np.arange(UNMON_SITE_NUM))
    decoy_mon = []
    for i in range(MON_SITE_NUM):
        for j in range(MON_INST_NUM):
            decoy_mon.append((i,j))
    decoy_mon = np.random.permutation(decoy_mon)
    decoy_list = []
    for i in range(MON_SITE_NUM):
        for j in range(MON_INST_NUM):
            fdir = join(args.dir, str(i)+'-'+str(j)+args.format)
            flist.append(fdir)
            decoy_page = join(args.decoy, str(decoy_unmon[i*MON_SITE_NUM+j]) + args.format)
            decoy_list.append(decoy_page)
    for i in range(UNMON_SITE_NUM):
        fdir = join(args.dir, str(i)+args.format)
        decoy_id = decoy_mon[i]
        decoy_page = join(args.decoy, str(decoy_id[0])+"-"+str(decoy_id[1])+args.format)
        flist.append(fdir)
        decoy_list.append(decoy_page)
    assert len(decoy_list) == len(flist)
    flist = list(zip(flist, decoy_list))
    logger.info("Outputdir:{}, generate {} traces".format(outputdir, len(flist)))
    parallel(flist)
