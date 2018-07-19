#!/bin/bash

metric=cosface
margin=0.35
scalar=30
name=${metric}_m_${margin}_s_${scalar}
optim_type=SGD_Step
dropout=0
feat_size=1024

python examples/train_balanced.py \
  --train_all \
  --batchsize=24 \
  --metric=${metric} \
  --margin=${margin} \
  --scalar=${scalar} \
  --name=${name} \
  --optim_type=${optim_type} \
  --dropout=${dropout} \
  --feat_size=${feat_size}