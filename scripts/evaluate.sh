#!/bin/bash

metric=cosface
margin=0.35
scalar=30
name=${metric}_m_${margin}_s_${scalar}

srun -p VIBackEnd \
  python evaluate/evaluate.py \
    logs/${name}.mat
