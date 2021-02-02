#!/bin/bash
set -e

python -Om benchmarking.project \
  --poisson 5 \
  --trials 4 \
  --width 800 \
  --num-angles 32 \
  --phantom peppers \

python -Om benchmarking.reconstruct \
  --ncore 16 \
  --num-iter 5 \
  --max-iter 50 \
  --phantom peppers \
  --parameters "[{'algorithm': 'mlem', 'num_iter': 5, 'accelerated': True, 'device': 'gpu', 'interpolation': 'NN'}, {'algorithm': 'mlem', 'num_iter': 5, 'accelerated':True, 'device': 'gpu', 'interpolation': 'LINEAR'},{'algorithm': 'mlem', 'num_iter': 5, 'accelerated':True, 'device': 'gpu', 'interpolation': 'CUBIC'},{'algorithm': 'sirt', 'num_iter': 5, 'accelerated':True, 'device': 'gpu', 'interpolation': 'NN'},{'algorithm': 'sirt', 'num_iter': 5, 'accelerated':True, 'device': 'gpu', 'interpolation': 'LINEAR'},{'algorithm': 'sirt', 'num_iter': 5, 'accelerated':True, 'device': 'gpu', 'interpolation': 'CUBIC'}]"

python -Om benchmarking.summarize \
  --phantom peppers \
  --trials 32 \
