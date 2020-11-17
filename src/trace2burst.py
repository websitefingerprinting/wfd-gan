import numpy as np
import os
import common as cm
import utils
import argparse
import pandas as pd
import multiprocessing as mp

logger = utils.init_logger('burst')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generat a distribution for ipt')

    parser.add_argument('--dir',
                        metavar='<traces path>',
                        required=True,
                        help='Path to the directory with the traffic traces to be simulated.')
    parser.add_argument('--format',
                        metavar='<file suffix>',
                        default=".cell",
                        )
    parser.add_argument('--proc_num',
                        type=int,
                        metavar='<process num>',
                        default=2,
                        help='The num of parallel workers')

    # Parse arguments
    args = parser.parse_args()
    return args


def parse(fpath):
    global format
    with open(fpath, "r") as f:
        tmp = f.readlines()
    t = pd.Series(tmp).str.slice(0, -1).str.split("\t", expand=True).astype(np.int64)
    trace = np.array(t)
    trace[:, 0] = (trace[:,0] - trace[0, 0])
    trace = trace.astype(float)
    trace[:,0] /= 1e9
    burst = []
    cursign = np.sign(trace[0, 1])
    curburst = 0
    curtime = trace[0, 0]
    for pkt in trace:
        if np.sign(pkt[1]) == cursign:
            curburst += abs(pkt[1])
        else:
            burst.append([curtime, curburst * cursign])
            curtime = pkt[0]
            curburst = abs(pkt[1])
            cursign = np.sign(pkt[1])
    burst.append([curtime, curburst * cursign])

    burst_outputdir = fpath.split(format)[0] + '.burst'
    with open(burst_outputdir, 'w') as f:
        for b in burst:
            f.write('{:.4f}\t{:.0f}\n'.format(b[0], b[1]))


if __name__ == '__main__':
    global MON_SITE_NUM, length, format
    # parser config and arguments
    args = parse_arguments()
    format = args.format
    logger.info("Arguments: %s" % (args))

    cf = utils.read_conf(cm.confdir)
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
            if os.path.exists(os.path.join(args.dir, str(i) + "-" + str(j) + args.format)):
                flist.append(os.path.join(args.dir, str(i) + "-" + str(j) + args.format))
    for i in range(UNMON_SITE_NUM):
        if os.path.exists(os.path.join(args.dir, str(i) + args.format)):
            flist.append(os.path.join(args.dir, str(i) + args.format))

    with mp.Pool(processes=args.proc_num) as p:
        p.map(parse, flist)
        p.close()
        p.join()

