#!/bin/bash

metric=cosface
margin=0.35
scalar=30
name=${metric}_m_${margin}_s_${scalar}_balanced
optim_type=Adam_Warmup
dropout=0
feat_size=2048
batchsize=64

srun -p VIBackEnd2 --gres=gpu:1 \
python examples/train_balanced.py \
  --train_all \
  --batchsize=${batchsize} \
  --metric=${metric} \
  --margin=${margin} \
  --scalar=${scalar} \
  --name=${name} \
  --optim_type=${optim_type} \
  --dropout=${dropout} \
  --feat_size=${feat_size} 
