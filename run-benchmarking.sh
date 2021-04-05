#!/bin/bash
set -e

python -Om benchmarking.project \
  --poisson 500 \
  --trials 8 \
  --width 1446 \
  --num-angles 1500 \
  --phantom peppers \

python -Om benchmarking.reconstruct \
  --ncore 4 \
  --max-iter 50 \
  --phantom peppers \

python -Om benchmarking.summarize \
  --phantom peppers \
  --trials 8 \
