#!/bin/bash
data_path=/ssd/xxx/xxx/parsed/xxx/
model_path=./dump/xxx/xxx/model/
o2o_path=./dump/xxx/feature/time_feature_0-100x0-1000_o2o.ipt


for i in $(seq 0.0 0.1 1)
do
  echo $i
  python3 src/wfgan-sim.py --n_cpu 40 --dir $data_path --model $model_path --ipt $o2o_path --tol $i
done