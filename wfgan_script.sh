#!/bin/bash
data_path=/ssd/jgongac/AlexaCrawler/parsed/dataset0_100_0711_103041_filtered/
model_path=./dump/rimmer_top877_2000/training_0629_010626/model/
o2o_path=./dump/rimmer_top877_2000/feature/feature/time_feature_0-100x0-1000_o2o.ipt


for i in $(seq 0.0 0.1 1)
do
  echo $i
  python3 src/wfgan-sim.py --n_cpu 40 --dir $data_path --model $model_path --ipt $o2o_path --tol $i
done