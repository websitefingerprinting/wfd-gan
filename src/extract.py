import numpy as np
import os
from os.path import join
import common as cm
import utils
import argparse
import multiprocessing as mp
import pandas as pd

logger = utils.init_logger('extract')
burst_reorder_threshold_t = [0.008, 0.001]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract burst sequences ipt from raw traces')

    parser.add_argument('--dir',
                        metavar='<traces path>',
                        required=True,
                        help='Path to the directory with the traffic traces to be simulated.')
    parser.add_argument('--length',
                        type=int,
                        default=1000,
                        help='Pad to length.'
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


def group_pkts(pkts):
    """Group packets into bursts, if time gap between two packets are less than threshold,
    then group together. The timestamp for a burst is the timestamp of the start packet.
    """
    burst_seq = []
    cnt = pkts[0, 1]
    start_time = pkts[0, 0]
    if cnt > 0:
        threshold = burst_reorder_threshold_t[0]
    else:
        threshold = burst_reorder_threshold_t[1]
    for i in range(1, len(pkts)):
        last_pkt = pkts[i - 1]
        cur_pkt = pkts[i]
        if cur_pkt[0] - last_pkt[0] < threshold:
            cnt += cur_pkt[1]
        else:
            burst_seq.append([start_time, cnt])
            cnt = cur_pkt[1]
            start_time = cur_pkt[0]
    burst_seq.append([start_time, cnt])
    burst_seq = np.array(burst_seq)

    assert sum(burst_seq[:, 1]) == sum(pkts[:, 1])
    return burst_seq


def get_burst(trace):
    # first remove the first few lines that are incoming packets
    start = -1
    for time, size in trace:
        start += 1
        if size > 0:
            break
    trace = trace[start:].copy()
    outgoing_burst_seqs = group_pkts(trace[trace[:, 1] > 0])
    incoming_burst_seqs = group_pkts(trace[trace[:, 1] < 0])
    burst_seqs = np.concatenate((outgoing_burst_seqs, incoming_burst_seqs), axis=0)
    assert len(burst_seqs) == len(outgoing_burst_seqs) + len(incoming_burst_seqs)
    burst_seqs = burst_seqs[burst_seqs[:, 0].argsort()]

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


def extract(trace):
    global length
    burst_seq = get_burst(trace)
    times = burst_seq[:, 0]
    bursts = list(abs(burst_seq[:, 1]))
    bursts.insert(0, len(bursts))
    bursts = bursts[:length] + [0] * (length - len(bursts))
    assert len(bursts) == length
    return bursts, times


def parallel(flist, n_jobs=80):
    with mp.Pool(n_jobs) as p:
        res = p.map(extractfeature, flist)
        p.close()
        p.join()
    return res


def extractfeature(fdir):
    global MON_SITE_NUM
    fname = fdir.split('/')[-1].split(".")[0]
    trace = utils.loadTrace(fdir)
    bursts, times = extract(trace)
    if '-' in fname:
        label = int(fname.split('-')[0])
    else:
        label = int(MON_SITE_NUM)
    return bursts, times, label


if __name__ == '__main__':
    global MON_SITE_NUM, length
    # parser config and arguments
    args = parse_arguments()
    length = args.length + 1  # add another feature as the real length of the trace
    logger.info("Arguments: %s" % (args))
    outputdir = join(cm.outputdir, os.path.split(args.dir.rstrip('/'))[1], 'feature')
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    cf = utils.read_conf(cm.confdir)
    MON_SITE_NUM = int(cf['monitored_site_num'])
    MON_INST_NUM = int(cf['monitored_inst_num'])
    MON_SITE_START_IND = int(cf['monitored_site_start_ind'])
    MON_INST_START_IND = int(cf['monitored_inst_start_ind'])
    if cf['open_world'] == '1':
        UNMON_SITE_NUM = int(cf['unmonitored_site_num'])
        OPEN_WORLD = 1
    else:
        OPEN_WORLD = 0
        UNMON_SITE_NUM = 0

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

    raw_data_dict = parallel(flist)
    bursts, times, labels = zip(*raw_data_dict)
    bursts = np.array(bursts)
    labels = np.array(labels)
    logger.info("feature sizes:{}, label size:{}".format(bursts.shape, labels.shape))
    np.savez_compressed(join(outputdir, "raw_feature.npz"), features=bursts, labels=labels)
    logger.info("output to {}".format(join(outputdir, "raw_feature_{}-{}x{}-{}.npz".
                                           format(MON_SITE_START_IND, MON_SITE_START_IND + MON_SITE_NUM,
                                                  MON_INST_START_IND, MON_INST_NUM + MON_INST_START_IND))))

    # save the time information. The even indexes are outgoing timestamps and the odd indexes are incoming ones.
    np.savez(join(outputdir, "time_feature_{}-{}x{}-{}.npz").
             format(MON_SITE_START_IND, MON_SITE_START_IND + MON_SITE_NUM, MON_INST_START_IND,
                    MON_INST_NUM + MON_INST_START_IND), *times)
