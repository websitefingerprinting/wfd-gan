import argparse
import datetime
import glob
import multiprocessing as mp
import os
import re
from os import mkdir
from os.path import join, isdir
from time import strftime

import joblib
from KDEpy.bw_selection import silvermans_rule

import common as cm
import utils
from model import *

logger = utils.init_logger('mergepad')
device = torch.device('cpu')
model_name_pattern = "generator_seqlen([0-9]*)_cls([0-9]*)_latentdim([0-9]*).ckpt"


# random.seed(1123)
# np.random.seed(1123)

def load_trace(fname, t=999):
    '''load a trace from fpath/fname up to t time.'''
    '''return trace and its name: cls-inst'''
    label = fname

    pkts = []
    with open(fname, 'r') as f:
        for line in f:
            try:
                timestamp, length = line.strip().split(cm.TRACE_SEP)
                pkts.append([float(timestamp), int(length)])
                if float(timestamp) >= t + 0.5:
                    break
            except ValueError:
                logger.warning("Could not split line: %s in %s", line, fname)
    return label, np.array(pkts)


def weibull(k=0.75):
    return np.random.weibull(0.75)


def dump(trace, fpath):
    '''Write trace packet into file `fpath`.'''
    with open(fpath, 'w') as fo:
        for packet in trace:
            fo.write("{:.4f}".format(packet[0]) + '\t' + "{}".format(int(packet[1])) + '\n')


def merge(this, other, start, cnt=1):
    '''t = 999, pad all pkts, otherwise pad up to t seconds'''
    other[:, 0] -= other[0][0]
    other[:, 0] += start
    other[:, 1] *= cnt
    if this is None:
        this = other
    else:
        this = np.concatenate((this, other), axis=0)
    return this


def est_iat(trace):
    trace_1 = np.concatenate((trace[1:], trace[0:1]), axis=0)
    itas = trace_1[:-1, 0] - trace[:-1, 0]
    return np.random.uniform(np.percentile(itas, 20), np.percentile(itas, 80))


def syn_trace(cind, duration):
    global ipt_sampler
    # first generate a burst sequence
    synthesized_x = query_from_gan(cind)
    cur_t = 0
    res = []
    last_sign = 1
    while cur_t < duration:
        if len(synthesized_x) == 0:
            synthesized_x = query_from_gan(cind)
        cell = synthesized_x.pop(0)
        cur_sign = np.sign(cell)
        if last_sign > 0 and cur_sign > 0:
            ipt_type = 'o2o'
        elif last_sign > 0 and cur_sign < 0:
            ipt_type = 'o2i'
        elif last_sign < 0 and cur_sign > 0:
            ipt_type = 'i2o'
        elif last_sign < 0 and cur_sign < 0:
            ipt_type = 'i2i'
        ipt = ipt_sampler.sample(ipt_type)
        cur_t = cur_t + ipt
        res.append([cur_t, cell])
        last_sign = cur_sign
    res = np.array(res)
    res[:, 0] -= res[0, 0]
    return res

def query_from_gan(cind):
    global model, latent_dim, scaler, cls_num
    model.eval()
    with torch.no_grad():
        z = np.random.randn(1, latent_dim).astype('float32')
        z = torch.from_numpy(z).to(device)
        c = torch.zeros(1, cls_num)
        c[:, cind] = 1
        c = c.to(device)
        synthesized_x = model(z, c).cpu().numpy()
        synthesized_x = scaler.inverse_transform(synthesized_x).flatten()
        length = int(synthesized_x[0])
        synthesized_x = synthesized_x[1:1 + length].astype(int)
        synthesized_x = np.trim_zeros(synthesized_x, trim='b')
        if len(synthesized_x) % 2 == 1:
            synthesized_x = synthesized_x[:-1]
        synthesized_x[synthesized_x<=0] = 1
        assert len(synthesized_x) % 2 == 0
    # expand bursts
    res = []
    for i, burst in enumerate(synthesized_x):
        if i % 2 == 0:
            cur_sign = 1
        else:
            cur_sign = -1
        for _ in range(burst):
            res.append(1 * cur_sign)
    return res



def MergePad2(output_dir, outputname, noise, mergelist=None, waiting_time=10):
    '''mergelist is a list of file names'''
    '''write in 2 files: the merged trace; the merged trace's name'''
    labels = ""
    this = None
    start = 0.0

    for cnt, fname in enumerate(mergelist):
        label, trace = load_trace(fname)
        labels += label + '\t'
        this = merge(this, trace, start, cnt=cnt + 1)
        start = this[-1][0]
        '''pad noise or not'''
        if noise:
            cls_num = np.random.randint(MON_SITE_NUM + OPEN_WORLD)
            # logger.debug("Sampled trace class: {}".format(cls_num))
            if cnt == len(mergelist) - 1:
                ### This is a param in mergepadding###
                '''We assume that the dwell time on the last page d_{max} = uniform(10,15),
                For an attacker, the best he can do is remove the last 10s traffic.  
                '''
                t = np.random.uniform(waiting_time, waiting_time + 5) - waiting_time
            else:
                t = np.random.uniform(1, 10)
            small_time = est_iat(trace)

            # logger.debug("Delta t is %.5f seconds. Dwell time %.4f" % (small_time, t))
            noise_site = syn_trace(cls_num, max(t - small_time, 0.01))
            this = merge(this, noise_site, start + small_time, cnt=999)
            # logger.info("Dwell time is %.2f seconds"%(t))
            start = start + t
        else:
            t = np.random.uniform(1, 10)
            start = start + t

    if noise:
        this = this[this[:, 0].argsort(kind="mergesort")]
    dump(this, join(output_dir, outputname + '.merge'))
    # logger.debug("Merged trace is dumpped to %s.merge" % outputname)
    return labels


def init_directories():
    # Create a results dir if it doesn't exist yet
    if not isdir(cm.outputdir):
        mkdir(cm.outputdir)

    # Define output directory
    timestamp = strftime('%m%d_%H%M')
    output_dir = join(cm.outputdir, 'mergepad_' + timestamp)
    while os.path.exists(output_dir):
        timestamp = strftime('%m%d_%H%M')
        output_dir = join(cm.outputdir, 'mergepad_' + timestamp)

    logger.info("Creating output directory: %s" % output_dir)

    # make the output directory
    mkdir(output_dir)
    return output_dir


def parse_arguments():
    # Read configuration file

    parser = argparse.ArgumentParser(description='It simulates adaptive padding on a set of web traffic traces.')

    parser.add_argument('--dir',
                        metavar='<traces path>',
                        required=True,
                        help='Path to the directory with the traffic traces to be simulated.')
    parser.add_argument('--model',
                        type=str,
                        help='Dir of the trained gan model.(The parent folder)')
    parser.add_argument('--ipt',
                        type=str,
                        help='Dir of the saved IPT file (i.e., ./kde/cdf.npy).')
    parser.add_argument('--noise',
                        type=str,
                        metavar='<pad noise>',
                        default='False',
                        help='Simulate whether pad noise or not')

    parser.add_argument('-n',
                        type=int,
                        metavar='<Instance Number>',
                        help='generate n instances')
    parser.add_argument('-m',
                        type=int,
                        metavar='<Merged Number>',
                        help='generate n instances of m merged traces')
    parser.add_argument('-b',
                        type=int,
                        metavar='<baserate>',
                        default=10,
                        help='baserate')
    parser.add_argument('--mode',
                        type=str,
                        metavar='<mode>',
                        default='fix',
                        help='To generate random-length or fixed-length trace')
    parser.add_argument('--log',
                        type=str,
                        dest="log",
                        metavar='<log path>',
                        default='stdout',
                        help='path to the log file. It will print to stdout by default.')

    args = parser.parse_args()

    return args


def CreateMergedTrace(traces_path, list_names, N, M, BaseRate):
    '''generate length-N merged trace'''
    '''with prob baserate/(baserate+1) a nonsensitive trace is chosen'''
    '''with prob 1/(baserate+1) a sensitive trace is chosen'''
    list_sensitive = glob.glob(join(traces_path, '*-*'))
    list_nonsensitive = list(set(list_names) - set(list_sensitive))

    s1 = len(list_sensitive)
    s2 = len(list_nonsensitive)

    mergedTrace = np.array([])
    for i in range(N):
        mergedTrace = np.append(mergedTrace, np.random.choice(list_sensitive + list_nonsensitive, M, replace=False, \
                                                              p=[1.0 / (s1 * (BaseRate + 1))] * s1 + [
                                                                  BaseRate / (s2 * (BaseRate + 1))] * s2))
    mergedTrace = mergedTrace.reshape((N, M))

    return mergedTrace


def CreateRandomMergedTrace(traces_path, list_names, N, M, BaseRate):
    '''generate random-length merged trace'''
    '''with prob baserate/(baserate+1) a nonsensitive trace is chosen'''
    '''with prob 1/(baserate+1) a sensitive trace is chosen'''
    list_sensitive = glob.glob(join(traces_path, '*-*'))
    list_nonsensitive = list(set(list_names) - set(list_sensitive))

    s1 = len(list_sensitive)
    s2 = len(list_nonsensitive)

    mergedTrace = []
    nums = np.random.choice(range(2, M + 1), N)
    for i, num in enumerate(nums):
        mergedTrace.append(np.random.choice(list_sensitive + list_nonsensitive, num, replace=False, \
                                            p=[1.0 / (s1 * (BaseRate + 1))] * s1 + [
                                                BaseRate / (s2 * (BaseRate + 1))] * s2))
    return mergedTrace, nums


def parallel(output_dir, noise, mergedTrace, n_jobs=20):
    cnt = range(len(mergedTrace))
    l = len(cnt)
    param_dict = zip([output_dir] * l, cnt, [noise] * l, mergedTrace)
    pool = mp.Pool(n_jobs)
    l = pool.map(work, param_dict)
    return l


def work(param):
    np.random.seed(datetime.datetime.now().microsecond)
    output_dir, cnt, noise, T = param[0], param[1], param[2], param[3]
    return MergePad2(output_dir, str(cnt), noise, T)


class IPT_Sampler:
    def __init__(self, cdfs):
        self.cdfs = cdfs # 4 cdf: o2o, o2i, i2o, i2i

    def sample(self, cdf_type):
        # sample a random number from one type of distribution
        if cdf_type == 'o2o' or cdf_type == 0:
            return self.sample_('o2o')
        elif cdf_type == 'o2i' or cdf_type == 1:
            return self.sample_('o2i')
        elif cdf_type == 'i2o' or cdf_type == 2:
            return self.sample_('i2o')
        elif cdf_type == 'i2i' or cdf_type == 3:
            return self.sample_('i2i')
        else:
            raise ValueError("Wrong cdf type:{}".format(cdf_type))
            return None

    def sample_(self, cdf_index):
        x, cdf = self.cdfs[cdf_index]
        y = np.random.uniform(0.0001, 1)  # use 0.0001 to avoid exception next line
        y0, y1 = cdf[cdf <= y][-1], cdf[cdf >= y][0]
        x0, x1 = x[cdf <= y][-1], x[cdf >= y][0]
        if y0 == y1:
            return 10 ** ((x0 + x1) / 2)
        else:
            return 10 ** ((y - y0) / (y1 - y0) * (x1 - x0) + x0)


if __name__ == '__main__':
    global model, scaler, latent_dim, cls_num, ipt_sampler
    # global list_names
    # parser config and arguments
    args = parse_arguments()
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

    # load model
    model_dir = glob.glob(join(args.model, "generator*.ckpt"))[0]
    pattern = re.compile(model_name_pattern)
    m = pattern.match(model_dir.split("/")[-1])
    seqlen, cls_num, latent_dim = int(m.group(1)), int(m.group(2)), int(m.group(3))
    model = Generator(seqlen, cls_num, latent_dim).to(device)
    model.load_state_dict(torch.load(model_dir, map_location=device))

    scaler_dir = glob.glob(join(args.model, "scaler.gz"))[0]
    scaler = joblib.load(scaler_dir)
    logger.info("Model loaded.")

    # build ipt sampler
    ipt_dict = np.load(args.ipt, allow_pickle=True).item()
    ipt_sampler = IPT_Sampler(ipt_dict)

    list_names = glob.glob(join(args.dir, '*'))
    if args.mode == 'fix':
        mergedTrace = CreateMergedTrace(args.dir, list_names, args.n, args.m, args.b)
    elif args.mode == 'random':
        mergedTrace, nums = CreateRandomMergedTrace(args.dir, list_names, args.n, args.m, args.b)
    else:
        logger.error("Wrong mode :{}".format(args.mode))

    # Init run directories
    output_dir = init_directories()
    if args.mode == 'random':
        np.save(join(output_dir, 'num.npy'), nums)

    l = parallel(output_dir, eval(args.noise), mergedTrace, 20)
    # l = []
    # cnt = 0
    # for T in mergedTrace:
    #     l.append(MergePad2(output_dir,  str(cnt) , eval(args.noise), T))
    #     cnt += 1

    with open(join(output_dir, 'list'), 'w') as f:
        [f.write(label + '\n') for label in l]
    print(output_dir)
