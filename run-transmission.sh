#!/bin/bash
set -e

python -Om benchmarking.project \
  --noise 2000 \
  --transmission \
  --trials 8 \
  --width 1446 \
  --num-angles 1500 \
  --phantom coins \

python -Om benchmarking.reconstruct \
  --ncore 16 \
  --max-iter 500 \
  --phantom coins \

python -Om benchmarking.summarize \
  --phantom coins \
  --trials 8 \
