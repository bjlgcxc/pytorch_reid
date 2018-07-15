#!/bin/bash

metric=cosface
margin=0.35
scalar=30
name=${metric}_m_${margin}_s_${scalar}
optim_type=SGD_Step

python examples/train.py \
  --train_all \
  --batchsize=32 \
  --erasing_p=0.2 \
  --metric=${metric} \
  --margin=${margin} \
  --scalar=${scalar} \
  --name=${name} \
  --optim_type=${optim_type}
