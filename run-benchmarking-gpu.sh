#!/bin/bash
set -e
currentDate=`date +%F`
outputDir=tomopy.github.io/$currentDate/gpu

python -Om benchmarking.project \
  --poisson 500 \
  --trials 32 \
  --width 1440 \
  --num-angles 1500 \
  --phantom peppers \
  --output-dir $outputDir \

python -Om benchmarking.reconstruct \
  --ncore 16 \
  --max-iter 50 \
  --phantom peppers \
  --parameters "[{'algorithm': 'sirt', 'accelerated':True, 'device': 'gpu', 'interpolation': 'NN'},{'algorithm': 'sirt', 'accelerated':True, 'device': 'gpu', 'interpolation': 'LINEAR'},{'algorithm': 'sirt', 'accelerated':True, 'device': 'gpu', 'interpolation': 'CUBIC'}]" \
  --output-dir $outputDir \

python -Om benchmarking.summarize \
  --phantom peppers \
  --trials 32 \
  --output-dir $outputDir\
