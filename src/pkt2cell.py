import numpy as np
import os
from os.path import join
import common as cm
import utils
import argparse
import pandas as pd
import multiprocessing as mp

logger = utils.init_logger('wfd')
CELL_SIZE = 514

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse cells from pkt files (only implemented the closed world)')

    parser.add_argument('--dir',
                        metavar='<traces path>',
                        required=True,
                        help='Path to the directory with the traffic traces to be simulated.')
    parser.add_argument('--format',
                        metavar='<file suffix>',
                        default=".pkt",
                        )
    parser.add_argument('--proc_num',
                        type=int,
                        metavar='<process num>',
                        default=60,
                        help='The num of parallel workers')

    # Parse arguments
    args = parser.parse_args()
    return args


def parse(fpath):
    global format, dst_dir
    with open(fpath, "r") as f:
        tmp = f.readlines()
    t = pd.Series(tmp).str.slice(0, -1).str.split("\t", expand=True).astype(float)
    pkts = np.array(t)
    cells = []
    fname = fpath.split('/')[-1]
    dst_fname = fname.replace(format, '.cell')
    with open(join(dst_dir, dst_fname), 'w') as f:
        for pkt in pkts:
            cur_t = pkt[0]
            size = pkt[1]
            cur_sign = np.sign(size)
            num_of_cells = int(abs(np.round(size / CELL_SIZE)))
            for _ in range(num_of_cells):
                f.write('{:.6f}\t{:.0f}\n'.format(cur_t, cur_sign))


if __name__ == '__main__':
    global format, dst_dir
    # parser config and arguments
    args = parse_arguments()
    format = args.format
    logger.info("Arguments: %s" % (args))

    cf = utils.read_conf(cm.confdir)
    MON_SITE_NUM = int(cf['monitored_site_num'])
    MON_INST_NUM = int(cf['monitored_inst_num'])

    flist = []
    for i in range(MON_SITE_NUM):
        for j in range(MON_INST_NUM):
            if os.path.exists(os.path.join(args.dir, str(i) + "-" + str(j) + args.format)):
                flist.append(os.path.join(args.dir, str(i) + "-" + str(j) + args.format))

    dst_dir = args.dir.rstrip('/') + '_cell'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    with mp.Pool(processes=args.proc_num) as p:
        p.map(parse, flist)
        p.close()
        p.join()

