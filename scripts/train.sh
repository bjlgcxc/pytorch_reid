#!/bin/bash

metric=linear
margin=0.35
scalar=30
name=${metric}_m_${margin}_s_${scalar}
optim_type=SGD_Step
dropout=0.5
feat_size=1024
batchsize=32

srun -p VIBackEnd --gres=gpu:1 \
python examples/train.py \
  --train_all \
  --batchsize=${batchsize} \
  --metric=${metric} \
  --margin=${margin} \
  --scalar=${scalar} \
  --name=${name} \
  --optim_type=${optim_type} \
  --dropout=${dropout} \
  --feat_size=${feat_size}
