#!/bin/bash
set -e

python -Om benchmarking.project \
  --poisson 500 \
  --trials 32 \
  --width 1446 \
  --num-angles 1500 \
  --phantom coins \

python -Om benchmarking.reconstruct \
  --ncore 16 \
  --num-iter 5 \
  --max-iter 50 \
  --phantom coins \

python -Om benchmarking.summarize \
  --phantom coins \
  --trials 8 \
