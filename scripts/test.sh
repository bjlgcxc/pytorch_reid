#!/bin/bash

metric=cosface
margin=0.35
scalar=30
name=${metric}_m_${margin}_s_${scalar}

epoch=last

python examples/test.py \
  --batchsize=16 \
  --metric=${metric} \
  --margin=${margin} \
  --scalar=${scalar} \
  --name=${name} \
  --save=logs/${name}.mat \
  --which_epoch=${epoch}
