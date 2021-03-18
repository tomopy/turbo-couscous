#!/bin/bash
set -e
currentDate=`date +%F`
outputDir=tomopy.github.io/$currentDate/gpu

python -Om benchmarking.project \
  --poisson 5 \
  --trials 4 \
  --width 400 \
  --num-angles 30 \
  --phantom peppers \
  --output-dir $outputDir \

python -Om benchmarking.reconstruct \
  --ncore 16 \
  --max-iter 20 \
  --phantom peppers \
  --parameters "[{'algorithm': 'sirt', 'accelerated':True, 'device': 'gpu', 'interpolation': 'NN'}]" \
  --output-dir $outputDir \

python -Om benchmarking.summarize \
  --phantom peppers \
  --trials 32 \
  --output-dir $outputDir\
