#!/bin/bash

metric=sphereface
margin=3
scalar=30
name=${metric}_m_${margin}_s_${scalar}
optim_type=SGD_Warmup

python examples/train.py \
  --train_all \
  --batchsize=16 \
  --metric=${metric} \
  --margin=${margin} \
  --scalar=${scalar} \
  --name=${name} \
  --optim_type=${optim_type}
