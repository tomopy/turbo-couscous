#!/bin/bash
set -e

python -m benchmarking.project \
  --noise \
  --trials 2 \
  --width 1446 \
  --num-angles 1500 \
  --phantom peppers \
  --output-dir gpu \

gdb -ex r --args python benchmarking/reconstruct.py \
  --ncore 4 \
  --num-iter 5 \
  --max-iter 300 \
  --phantom peppers \
  --output-dir gpu \
  --parameters "[{'algorithm': 'sirt', 'num_iter': 5, 'accelerated': True, 'device': 'gpu', 'interpolation': 'LINEAR'}]" \
