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
    parser = argparse.ArgumentParser(description='Extract feature from raw traces')

    parser.add_argument('--dir',
                        metavar='<traces path>',
                        required=True,
                        help='Path to the directory with the traffic traces to be simulated.')
    parser.add_argument('--length',
                        type=int,
                        default=1200,
                        help='Pad to length.'
                        )
    parser.add_argument('--format',
                        metavar='<file suffix>',
                        default=".burst",
                        )
    parser.add_argument('--save_txt',
                        action='store_true',
                        default=False,
                        help='Whether output to txt file.'
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


def parse(fpath):
    with open(fpath, "r") as f:
        tmp = f.readlines()
    t = pd.Series(tmp).str.slice(0, -1).str.split("\t", expand=True).astype(float)
    t = np.array(t)
    return t[:, 1]


def extract(x):
    global length
    start = -1
    for pkt in x:
        start += 1
        if pkt > 0:
            break
    x = x[start:]
    new_x = []
    sign = np.sign(x[0])
    cnt = 0
    for e in x:
        if np.sign(e) == sign:
            cnt += abs(e)
        else:
            new_x.append(cnt)
            cnt = abs(e)
            sign = np.sign(e)
    new_x.append(cnt)
    new_x.insert(0, len(new_x))
    new_x = new_x[:length] + [0] * (length - len(new_x))
    assert len(new_x) == length
    return new_x


def parallel(flist, n_jobs=20):
    with mp.Pool(n_jobs) as p:
        res = p.map(extractfeature, flist)
        p.close()
        p.join()
    return res


def extractfeature(f):
    global MON_SITE_NUM
    fname = f.split('/')[-1].split(".")[0]
    t = parse(f)
    features = extract(t)
    if '-' in fname:
        label = int(fname.split('-')[0])
    else:
        label = int(MON_SITE_NUM)
    return features, label


if __name__ == '__main__':
    global MON_SITE_NUM, length
    # parser config and arguments
    args = parse_arguments()
    length = args.length + 1 # add another feature as the real length of the trace
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
    data_dict = {'feature': [], 'label': []}

    flist = []
    for i in range(MON_SITE_NUM):
        for j in range(MON_INST_NUM):
            if os.path.exists(os.path.join(args.dir, str(i) + "-" + str(j) + args.format)):
                flist.append(os.path.join(args.dir, str(i) + "-" + str(j) + args.format))
    for i in range(UNMON_SITE_NUM):
        if os.path.exists(os.path.join(args.dir, str(i) + args.format)):
            flist.append(os.path.join(args.dir, str(i) + args.format))

    raw_data_dict = parallel(flist, n_jobs=20)
    features, labels = zip(*raw_data_dict)
    features = np.array(features)
    labels = np.array(labels)
    data_dict['feature'], data_dict['label'] = features, labels
    logger.info("feature sizes:{}, label size:{}".format(features.shape, labels.shape))
    np.save(join(outputdir, "raw_feature"), data_dict)
    logger.info("output to {}".format(outputdir))
    if args.save_txt:
        with open(join(outputdir, 'raw_feature.txt'), 'w') as f:
            for feature in features:
                # feature = feature.argmax(axis=1)
                end = np.where(np.array(feature) > 0)[0][-1]  # last one
                for pkt in feature[:end]:
                    f.write("{:.0f} ".format(pkt))
                f.write("{:.0f}\n".format(feature[end]))
