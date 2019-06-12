#!/bin/bash
set -e

python -Om benchmarking.project \
  --noise 500 \
  --trials 32 \
  --width 1446 \
  --num-angles 1500 \
  --phantom peppers \

python -Om benchmarking.reconstruct \
  --ncore 16 \
  --num-iter 5 \
  --max-iter 300 \
  --phantom peppers \

python -Om benchmarking.summarize \
  --phantom peppers \
