#!/bin/bash

metric=cosface
margin=0.35
scalar=30
name=${metric}_m_${margin}_s_${scalar}_balanced
epoch=99
feat_size=2048
batchsize=32

srun -p VIBackEnd1 --gres=gpu:1 \
  python examples/test.py \
  --batchsize=${batchsize} \
  --metric=${metric} \
  --margin=${margin} \
  --scalar=${scalar} \
  --name=${name} \
  --save=logs/${name}.mat \
  --which_epoch=${epoch} \
  --feat_size=${feat_size} 

