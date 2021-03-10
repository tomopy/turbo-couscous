#!/bin/bash
set -e
currentDate=`date +%F`
outputDir=tomopy.github.io/$currentDate/cpu

python -Om benchmarking.project \
  --poisson 500 \
  --trials 32 \
  --width 1446 \
  --num-angles 1500 \
  --phantom peppers \

python -Om benchmarking.reconstruct \
  --ncore 16 \
  --num-iter 5 \
  --max-iter 50 \
  --phantom peppers \

python -Om benchmarking.summarize \
  --phantom peppers \
  --trials 32 \
