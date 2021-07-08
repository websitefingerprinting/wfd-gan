import numpy as np
import os
from os.path import join
import common as cm
import utils
import argparse
import pandas as pd
from KDEpy import FFTKDE
from KDEpy.bw_selection import silvermans_rule, improved_sheather_jones
from functools import partial


logger = utils.init_logger('ipt')
INST_NUM = 100 * 1000
core_num = 80


def parse_arguments():
    parser = argparse.ArgumentParser(description='Resample ipt from a dataset using FFTKDE modeling')

    parser.add_argument('--tdir',
                        metavar='<ipt path>',
                        default=None,
                        help='Path to the npz file that saves the parsed ipt info.')
    parser.add_argument('-n',
                        metavar='<resampling number>',
                        type=int,
                        default=1000000,
                        help='The number of data points resampled from the original distribution'
                        )

    # Parse arguments
    args = parser.parse_args()
    return args


def parse_ipt(arr):
    if len(arr) % 2 != 0:
        arr = arr[:-1]
    return list(np.diff(arr[::2])), list(arr[1::2] - arr[::2])


def prepare_dataset(data):
    """input: arr_1, arr_2, ...
    For each array, an even position represents the timestamp of an outgoing burst,
    an odd position represents the timestamps of an incoming burst
    """
    cnt = -1
    o2os = []
    o2is = []
    arr_list = []

    cnt = -1
    for key in data.keys():
        cnt += 1
        if cnt >= INST_NUM:
            break
        arr_list.append(data[key])
    with utils.poolcontext(processes=core_num) as p:
        res = p.map(parse_ipt, arr_list)

    for o2o, o2i in res:
        o2os.extend(o2o)
        o2is.extend(o2i)
    return np.array(o2os), np.array(o2is)


if __name__ == '__main__':
    # parser config and arguments
    args = parse_arguments()

    raw_dataset = np.load(args.tdir)
    o2o, o2i = prepare_dataset(raw_dataset)
    ipt_dataset = {'o2o': o2o, 'o2i': o2i}
    logger.info("Have {} o2o and {} o2i".format(len(o2o), len(o2i)))
    prefix = args.tdir.split('.npz')[0]
    np.savez_compressed('{}_ipt.npz'.format(prefix), o2o=o2o, o2i=o2i)
    logger.info('Save ipt info to {}'.format('{}_ipt.npz'.format(prefix)))
    # logger.info("KDE modeling...")
    # pdf = {}
    # for key in ipt_dataset.keys():
    #     data = ipt_dataset[key]
    #     data[data < 1e-6] = 1e-6
    #     log_ipt = np.log10(data)
    #     x, y = FFTKDE(kernel='gaussian', bw='ISJ').fit(log_ipt).evaluate()
    #     pdf[key] = [x, y]
    # # 'The data is in log scale (seconds).'
    # np.savez_compressed(join(outputdir, 'pdf.npz'), o2o=pdf['o2o'], o2i=pdf['o2i'])

    logger.info("Computing Kernel")
    for key in ipt_dataset.keys():
        data = ipt_dataset[key]
        data[data == 0] = 1e-6
        log_ipt = np.log10(data)
        kernel_std = improved_sheather_jones(log_ipt.reshape(-1, 1))  # Shape (obs, dims)
        # sample a part of the original data points
        resampled_data = np.random.choice(log_ipt, size=min(args.n, len(log_ipt)), replace=False)
        # # (1) First resample original data, then (2) add noise from kernel
        # resampled_data = np.random.choice(log_ipt, size=args.n, replace=True)
        # resampled_data = resampled_data + np.random.randn(args.n) * kernel_std
        # resampled_data = 10 ** resampled_data

        with open('{}_{}.ipt'.format(prefix, key), 'w') as f:
            f.write('# Log scale, in seconds. The first is the kernal std. Rest are real log(ipts).\n')
            f.write('{:.6f}\n'.format(kernel_std))
            for data in resampled_data:
                f.write("{:.6f}\n".format(data))
