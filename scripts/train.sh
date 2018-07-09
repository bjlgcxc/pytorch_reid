#!/bin/bash

metric=cosface
margin=0.35
scalar=30
name=${metric}_m_${margin}_s_${scalar}

python examples/train.py \
  --train_all \
  --batchsize=16 \
  --metric=${metric} \
  --margin=${margin} \
  --scalar=${scalar} \
  --name=${name}
