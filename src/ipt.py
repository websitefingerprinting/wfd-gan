import numpy as np
import os
from os.path import join
import common as cm
import utils
import argparse
import pandas as pd
from KDEpy import FFTKDE

logger = utils.init_logger('ipt')
bandwidth = 0.01

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

    # Parse arguments
    args = parser.parse_args()
    return args


def parse(fpath):
    with open(fpath, "r") as f:
        tmp = f.readlines()
    t = pd.Series(tmp).str.slice(0, -1).str.split("\t", expand=True).astype(float)
    t = np.array(t)
    o2o, o2i, i2o, i2i = [], [], [], []
    for i in range(1,len(t)):
        curpkt = t[i]
        lastpkt = t[i-1]
        ipt = curpkt[0] - lastpkt[0]
        assert ipt >= 0
        if ipt <= 1e-6:
            ipt = 1e-6
        if ipt > 1:
            ipt = 1
        if curpkt[1] > 0 and lastpkt[1] > 0:
            o2o.append( ipt )
        elif curpkt[1] >0 and lastpkt[1] < 0:
            i2o.append( ipt )
        elif curpkt[1] < 0 and lastpkt[1] > 0:
            o2i.append( ipt )
        elif curpkt[1] < 0 and lastpkt[1] < 0:
            i2i.append( ipt )
    return o2o, o2i, i2o, i2i





if __name__ == '__main__':
    global MON_SITE_NUM, length
    # parser config and arguments
    args = parse_arguments()
    outputdir = join(cm.outputdir, os.path.split(args.dir.rstrip('/'))[1], 'kde')
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
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

    res = {'o2o':[],'o2i':[],'i2o':[],'i2i':[]}
    for i, f in enumerate(flist[:2]):
        if i % 1000 == 0:
            logger.info("Processing {}/{}".format(i,len(flist)))
        o2o, o2i, i2o, i2i = parse(f)
        res['o2o'].extend(o2o)
        res['o2i'].extend(o2i)
        res['i2o'].extend(i2o)
        res['i2i'].extend(i2i)
    for key in res.keys():
        res[key] = np.log10(res[key])
    # res['o2o'] = np.random.choice(res['o2o'], min(100000000, len(res['o2o'])), replace=False)
    # res['o2i'] = np.random.choice(res['o2i'], min(100000000, len(res['o2i'])), replace=False)
    # res['i2o'] = np.random.choice(res['i2o'], min(100000000, len(res['i2o'])), replace=False)
    # res['i2i'] = np.random.choice(res['i2i'], min(100000000, len(res['i2i'])), replace=False)
    logger.info("{} {} {} {}".format(res['o2o'].shape,res['o2i'].shape,res['i2o'].shape,res['i2i'].shape))
    np.save(join(outputdir, 'ipt.npy'), res)
    logger.info("KDE modeling...")
    cdf = {}
    for kind in res.keys():
        x, y = FFTKDE(kernel='gaussian', bw='silverman').fit(res[kind]).evaluate()
        cumsum_y = np.cumsum(y)/sum(y)
        cdf[kind] = [x, cumsum_y]
    np.save(join(outputdir, 'cdf.npy'), cdf)


