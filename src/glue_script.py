import subprocess
from os.path import join
import os
from common import BASE_DIR

src_path = join(BASE_DIR, 'src/gan_glue.py')

for m in range(2,17):
    n = 9900 // m
    # print("m :{}, n: {}".format(m, n))
    cmd = "python3 " + src_path + " --dir /home/homes/jgongac/websiteFingerprinting/data/evaluation/ " + \
          "--model dump/evaluation/model_/ --ipt dump/evaluation/kde/cdf.npy " + \
          "-n "+ str(n)+" -m "+str(m)+ " -b 10 --noise True --mode fix"
    print(cmd)
    # subprocess.call(cmd, shell = True)


