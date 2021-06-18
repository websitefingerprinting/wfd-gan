import multiprocessing

import numpy as np
import os
from os.path import join
import common as cm
import utils
import argparse
import pandas as pd
from KDEpy import FFTKDE
from KDEpy.bw_selection import silvermans_rule, improved_sheather_jones

logger = utils.init_logger('ipt')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Resample ipt from a dataset using FFTKDE modeling')

    parser.add_argument('--fdir',
                        metavar='<traces path>',
                        default=None,
                        help='Path to the directory of the raw traces')
    parser.add_argument('--tdir',
                        metavar='<ipt path>',
                        default=None,
                        help='Path to the npz file that saves the parsed ipt info.')
    parser.add_argument('--format',
                        metavar='<file suffix>',
                        default=".cell",
                        )
    parser.add_argument('-n',
                        metavar='<resampling number>',
                        default=100000,
                        help='The number of data points resampled from the original distribution'
                        )

    # Parse arguments
    args = parser.parse_args()
    return args


def parse(fpath):
    trace = utils.loadTrace(fpath)
    filtered_trace = [trace[0]]
    last_sign = np.sign(trace[0, 1])
    for time, size in trace[1:]:
        cur_sign = np.sign(size)
        if cur_sign == last_sign:
            continue
        else:
            filtered_trace.append([time, size])
            last_sign = cur_sign
    filtered_trace = np.array(filtered_trace)
    outgoing_filtered_trace = filtered_trace[filtered_trace[:, 1] > 0]
    o2o = np.diff(outgoing_filtered_trace[:, 0])
    assert (o2o >= 0).all()
    o2i = []
    for i in range(1, len(filtered_trace)):
        cur_time, cur_size = filtered_trace[i]
        cur_sign = np.sign(cur_size)
        if cur_sign > 0:
            continue
        last_time = filtered_trace[i - 1][0]
        o2i.append(cur_time - last_time)
    o2i = np.array(o2i)
    assert (o2i >= 0).all()
    return list(o2o), list(o2i)


def prepare_dataset(flist):
    with multiprocessing.Pool(50) as p:
        res = p.map(parse, flist)
    o2o_list = []
    o2i_list = []
    for o2o, o2i in res:
        o2o_list.extend(o2o)
        o2i_list.extend(o2i)
    logger.info('{}, {}'.format(len(o2o_list), len(o2i_list)))
    return {'o2o': np.array(o2o_list), 'o2i': np.array(o2i_list)}


if __name__ == '__main__':
    # parser config and arguments
    args = parse_arguments()
    assert args.fdir or args.tdir

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

    if args.tdir:
        ipt_dataset = np.load(args.tdir)
        outputdir, _ = os.path.split(args.tdir)
    else:
        outputdir = join(cm.outputdir, os.path.split(args.fdir.rstrip('/'))[1], 'kde')
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        flist = []
        for i in range(MON_SITE_NUM):
            for j in range(MON_INST_NUM):
                if os.path.exists(os.path.join(args.fdir, str(i) + "-" + str(j) + args.format)):
                    flist.append(os.path.join(args.fdir, str(i) + "-" + str(j) + args.format))
        for i in range(UNMON_SITE_NUM):
            if os.path.exists(os.path.join(args.fdir, str(i) + args.format)):
                flist.append(os.path.join(args.fdir, str(i) + args.format))
        ipt_dataset = prepare_dataset(flist)
        np.savez_compressed(join(outputdir, 'ipt.npz'), o2o=ipt_dataset['o2o'], o2i=ipt_dataset['o2i'])


    logger.info("KDE modeling...")
    pdf = {}
    for key in ipt_dataset.keys():
        data = ipt_dataset[key]
        data[data == 0] = 1e-9
        log_ipt = np.log10(data)
        x, y = FFTKDE(kernel='gaussian', bw='ISJ').fit(log_ipt).evaluate()
        pdf[key] = [x, y]
    # 'The data is in log scale (seconds).'
    np.savez_compressed(join(outputdir, 'pdf.npz'), o2o=pdf['o2o'], o2i=pdf['o2i'])

    logger.info("Resampling data")
    for key in ipt_dataset.keys():
        data = ipt_dataset[key]
        data[data == 0] = 1e-9
        log_ipt = np.log10(data)
        kernel_std = improved_sheather_jones(log_ipt.reshape(-1, 1))  # Shape (obs, dims)
        # (1) First resample original data, then (2) add noise from kernel
        resampled_data = np.random.choice(log_ipt, size=args.n, replace=True)
        resampled_data = resampled_data + np.random.randn(args.n) * kernel_std
        resampled_data = 10 ** resampled_data
        with open(join(outputdir, '{}.txt'.format(key)), 'w') as f:
            for data in resampled_data:
                f.write("{:.6f}\n".format(data))

