#!/bin/bash

metric=linear
margin=0.25
scalar=20
name=${metric}_m_${margin}_s_${scalar}

#srun -p VIFrontEnd \
  python evaluate/evaluate.py \
    logs/${name}.mat
