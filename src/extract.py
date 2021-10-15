import numpy as np
import os
from os.path import join
import common as cm
import utils
import argparse
import multiprocessing as mp
import pandas as pd

logger = utils.init_logger('extract')

# it is possible the trace has a long tail
# if there is a time gap between two bursts larger than CUT_OFF_THRESHOULD
# We cut off the trace here sicne it could be a long timeout or
# maybe the loading is already finished
# Set a very conservative value
CUT_OFF_THRESHOLD = 10


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract burst sequences ipt from raw traces')

    parser.add_argument('--dir',
                        metavar='<traces path>',
                        required=True,
                        help='Path to the directory with the traffic traces to be simulated.')
    parser.add_argument('--length',
                        type=int,
                        default=1400,
                        help='Pad to length.'
                        )
    parser.add_argument('--norm',
                        type=bool,
                        default=False,
                        help='Shall we normalize burst sizes by dividing it with cell size?'
                        )
    parser.add_argument('--norm_cell',
                        type=bool,
                        default=False,
                        help='Shall we ignore the value of a cell (e.g., +-888 -> +-1)?'
                        )
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


def get_burst(trace, fdir):
    # first check whether there are some outlier pkts based on the CUT_OFF_THRESHOLD
    # If the outlier index is within 50, cut off the head
    # else, cut off the tail
    start, end = 0, len(trace)
    ipt_burst = np.diff(trace[:, 0])
    ipt_outlier_inds = np.where(ipt_burst > CUT_OFF_THRESHOLD)[0]

    if len(ipt_outlier_inds) > 0:
        outlier_ind_first = ipt_outlier_inds[0]
        if outlier_ind_first < 50:
            start = outlier_ind_first + 1
        outlier_ind_last = ipt_outlier_inds[-1]
        if outlier_ind_last > 50:
            end = outlier_ind_last + 1

    if start != 0 or end != len(trace):
        print("File {} trace has been truncated from {} to {}".format(fdir, start, end))

    trace = trace[start:end].copy()

    # remove the first few lines that are incoming packets
    start = -1
    for time, size in trace:
        start += 1
        if size > 0:
            break

    trace = trace[start:].copy()
    trace[:, 0] -= trace[0, 0]
    assert trace[0, 0] == 0
    burst_seqs = trace

    # merge bursts from the same direction
    merged_burst_seqs = []
    cnt = 0
    sign = np.sign(burst_seqs[0, 1])
    time = burst_seqs[0, 0]
    for cur_time, cur_size in burst_seqs:
        if np.sign(cur_size) == sign:
            cnt += cur_size
        else:
            merged_burst_seqs.append([time, cnt])
            sign = np.sign(cur_size)
            cnt = cur_size
            time = cur_time
    merged_burst_seqs.append([time, cnt])
    merged_burst_seqs = np.array(merged_burst_seqs)
    assert sum(merged_burst_seqs[::2, 1]) == sum(trace[trace[:, 1] > 0][:, 1])
    assert sum(merged_burst_seqs[1::2, 1]) == sum(trace[trace[:, 1] < 0][:, 1])
    return np.array(merged_burst_seqs)


def extract(trace, fdir):
    global length, norm
    burst_seq = get_burst(trace, fdir)
    times = burst_seq[:, 0]
    bursts = abs(burst_seq[:, 1])
    if norm:
        bursts /= cm.CELL_SIZE
    bursts = list(bursts)
    bursts.insert(0, len(bursts))
    bursts = bursts[:length] + [0] * (length - len(bursts))
    assert len(bursts) == length
    return bursts, times


def parallel(flist, n_jobs=70):
    with mp.Pool(n_jobs) as p:
        res = p.map(extractfeature, flist)
        p.close()
        p.join()
    return res


def extractfeature(fdir):
    global MON_SITE_NUM, norm_cell
    fname = fdir.split('/')[-1].split(".")[0]
    trace = utils.loadTrace(fdir, norm_cell=norm_cell)
    bursts, times = extract(trace, fdir)
    if '-' in fname:
        label = int(fname.split('-')[0])
    else:
        label = int(MON_SITE_NUM)
    return bursts, times, label


if __name__ == '__main__':
    global MON_SITE_NUM, length, norm, norm_cell
    # parser config and arguments
    args = parse_arguments()
    length = args.length + 1  # add another feature as the real length of the trace
    norm = args.norm
    norm_cell = args.norm_cell
    logger.info("Arguments: %s" % (args))
    outputdir = join(cm.outputdir, os.path.split(args.dir.rstrip('/'))[1], 'feature')
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    cf = utils.read_conf(cm.confdir)
    MON_SITE_NUM = int(cf['monitored_site_num'])
    MON_INST_NUM = int(cf['monitored_inst_num'])
    MON_SITE_START_IND = int(cf['monitored_site_start_ind'])
    MON_INST_START_IND = int(cf['monitored_inst_start_ind'])
    # if cf['open_world'] == '1':
    #     UNMON_SITE_NUM = int(cf['unmonitored_site_num'])
    #     OPEN_WORLD = 1
    # else:
    #     OPEN_WORLD = 0
    #     UNMON_SITE_NUM = 0

    # logger.info('Extracting features...')

    flist = []
    for i in range(MON_SITE_START_IND, MON_SITE_START_IND + MON_SITE_NUM):
        for j in range(MON_INST_START_IND, MON_INST_START_IND + MON_INST_NUM):
            if os.path.exists(os.path.join(args.dir, str(i) + "-" + str(j) + args.format)):
                flist.append(os.path.join(args.dir, str(i) + "-" + str(j) + args.format))
    # do not support open world set.
    # for i in range(UNMON_SITE_NUM):
    #     if os.path.exists(os.path.join(args.dir, str(i) + args.format)):
    #         flist.append(os.path.join(args.dir, str(i) + args.format))
    logger.info('In total {} files.'.format(len(flist)))
    raw_data_dict = parallel(flist)
    bursts, times, labels = zip(*raw_data_dict)
    bursts = np.array(bursts)
    labels = np.array(labels)
    logger.info("feature sizes:{}, label size:{}".format(bursts.shape, labels.shape))
    np.savez_compressed(
        join(outputdir, "raw_feature_{}-{}x{}-{}.npz".format(MON_SITE_START_IND, MON_SITE_START_IND + MON_SITE_NUM,
                                                             MON_INST_START_IND, MON_INST_NUM + MON_INST_START_IND)),
        features=bursts, labels=labels)
    logger.info("output to {}".format(join(outputdir, "raw_feature_{}-{}x{}-{}.npz".
                                           format(MON_SITE_START_IND, MON_SITE_START_IND + MON_SITE_NUM,
                                                  MON_INST_START_IND, MON_INST_NUM + MON_INST_START_IND))))

    # save the time information. The even indexes are outgoing timestamps and the odd indexes are incoming ones.
    np.savez(join(outputdir, "time_feature_{}-{}x{}-{}.npz").
             format(MON_SITE_START_IND, MON_SITE_START_IND + MON_SITE_NUM, MON_INST_START_IND,
                    MON_INST_NUM + MON_INST_START_IND), *times)
