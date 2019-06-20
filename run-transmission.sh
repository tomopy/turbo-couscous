#!/bin/bash
set -e

python -Om benchmarking.project \
  --noise 2000 \
  --transmission \
  --trials 32 \
  --width 1446 \
  --num-angles 1500 \
  --phantom peppers \

python -Om benchmarking.reconstruct \
  --ncore 16 \
  --num-iter 5 \
  --max-iter 80 \
  --phantom peppers \

python -Om benchmarking.summarize \
  --phantom peppers \
  --trials 32 \
