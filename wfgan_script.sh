#!/bin/bash
data_path=~/websiteFingerprinting/data/rimmer_top877_2000_cell/
model_path=./dump/rimmer_top877_2000/training_0629_010626/model/
ipt_path=./dump/rimmer_top877_2000/feature/

for i in $(seq 0.0 0.1 1)
do
  echo $i
  python3 src/wfgan-sim.py --n_cpu 30 --dir $data_path --model $model_path --ipt $ipt_path --tol $i
done