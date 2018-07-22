#!/bin/bash

metric=linear
margin=0.25
scalar=20
name=${metric}_m_${margin}_s_${scalar}
optim_type=SGD_Step
dropout=0.5
feat_size=1024
batchsize=16

#srun -p VIFrontEnd --gres=gpu:1 \
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
