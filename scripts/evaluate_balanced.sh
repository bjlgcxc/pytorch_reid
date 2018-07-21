#!/bin/bash

metric=cosface
margin=0.35
scalar=30
name=${metric}_m_${margin}_s_${scalar}_balanced

srun -p VIFrontEnd \
  python evaluate/evaluate.py \
    logs/${name}.mat
