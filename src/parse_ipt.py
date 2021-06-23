import numpy as np
import os
from os.path import join
import common as cm
import utils
import argparse
import multiprocessing as mp
import pandas as pd

logger = utils.init_logger('extract')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract burst sequences ipt from raw traces')

    parser.add_argument('--dir',
                        metavar='<traces path>',
                        required=True,
                        help='Path to the directory with the traffic traces to be simulated.')
    parser.add_argument('--format',
                        metavar='<file suffix>',
                        default=".pkt",
                        )
    parser.add_argument('--log',
                        type=str,
                        dest="log",
                        metavar='<log path>',
                        default='stdout',
                        help='path to the log file. It will print to stdout by default.')

    # Parse arguments
    args = parser.parse_args()
    return args


def get_ipt(trace):
    """Get ipt info of packets from one direction,
    Include the o2o time gap and whether two packets are split by some pkts from the other direction.
    Example: 1, -1, 1. This o2o is split
    """
    ipt = []
    for i in range(1, len(trace)):
        cur_pkt = trace.iloc[i]
        last_pkt = trace.iloc[i-1]
        time_gap = cur_pkt.time - last_pkt.time
        index_gap = cur_pkt.name - last_pkt.name
        ipt.append([time_gap, index_gap])
    return ipt


def parallel(flist, n_jobs=70):
    with mp.Pool(n_jobs) as p:
        res = p.map(extract, flist)
        p.close()
        p.join()
    return res


def extract(fdir):
    trace = utils.loadTrace(fdir)
    trace = pd.DataFrame(trace)
    trace.columns = ['time', 'direction']
    outgoing = trace[trace.direction > 0]
    incoming = trace[trace.direction < 0]
    o2o = get_ipt(outgoing)
    i2i = get_ipt(incoming)
    return np.array(o2o), np.array(i2i)


if __name__ == '__main__':
    # parser config and arguments
    args = parse_arguments()
    logger.info("Arguments: %s" % (args))
    outputdir = join(cm.outputdir, os.path.split(args.dir.rstrip('/'))[1], 'feature')
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    cf = utils.read_conf(cm.confdir)
    MON_SITE_NUM = int(cf['monitored_site_num'])
    MON_INST_NUM = int(cf['monitored_inst_num'])
    if cf['open_world'] == '1':
        UNMON_SITE_NUM = int(cf['unmonitored_site_num'])
        OPEN_WORLD = 1
    else:
        OPEN_WORLD = 0
        UNMON_SITE_NUM = 0

    # logger.info('Extracting features...')

    flist = []
    for i in range(MON_SITE_NUM):
        for j in range(MON_INST_NUM):
            if os.path.exists(os.path.join(args.dir, str(i) + "-" + str(j) + args.format)):
                flist.append(os.path.join(args.dir, str(i) + "-" + str(j) + args.format))
    for i in range(UNMON_SITE_NUM):
        if os.path.exists(os.path.join(args.dir, str(i) + args.format)):
            flist.append(os.path.join(args.dir, str(i) + args.format))

    raw_data_dict = parallel(flist)
    o2o, i2i = zip(*raw_data_dict)

    np.savez_compressed(join(outputdir, "o2o_i2i.npz"), o2o=o2o, i2i=i2i)
    logger.info("output to {}".format(join(outputdir, "o2o_i2i.npz")))
