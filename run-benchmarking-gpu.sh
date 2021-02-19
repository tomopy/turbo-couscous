#!/bin/bash
set -e
currentDate=`date +%F`
outputDir=tomopy.github.io/$currentDate/gpu

python -Om benchmarking.project \
  --poisson 5 \
  --trials 4 \
  --width 800 \
  --num-angles 32 \
  --phantom peppers \
  --output-dir $outputDir \

python -Om benchmarking.reconstruct \
  --ncore 16 \
  --max-iter 30 \
  --phantom peppers \
  --parameters "[{'algorithm': 'mlem', 'accelerated': True, 'device': 'gpu', 'interpolation': 'NN'}, {'algorithm': 'mlem', 'accelerated':True, 'device': 'gpu', 'interpolation': 'LINEAR'}]" \
  --output-dir $outputDir \

python -Om benchmarking.summarize \
  --phantom peppers \
  --trials 32 \
  --output-dir $outputDir\
