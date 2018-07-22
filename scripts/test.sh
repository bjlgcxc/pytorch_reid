#!/bin/bash

metric=linear
margin=0.25
scalar=20
name=${metric}_m_${margin}_s_${scalar}
epoch=last
feat_size=1024
batchsize=64

#srun -p VIFrontEnd --gres=gpu:1 \
python examples/test.py \
  --batchsize=${batchsize} \
  --metric=${metric} \
  --margin=${margin} \
  --scalar=${scalar} \
  --name=${name} \
  --save=logs/${name}.mat \
  --which_epoch=${epoch} \
  --feat_size=${feat_size}

