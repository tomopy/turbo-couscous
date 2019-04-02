#!/bin/bash
set -e

python -Om benchmarking.project \
  --noise \
  --trials 32 \
  --width 1446 \
  --num_angles 1500 \
  --phantom peppers \

python -Om benchmarking.reconstruct \
  --ncore 1 \
  --num-iter 300 \
  --iter-step 5 \
  --phantom peppers \

python -Om benchmarking.summarize \
  --phantom peppers \
