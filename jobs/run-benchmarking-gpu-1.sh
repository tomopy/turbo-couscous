#!/bin/bash
set -e
currentDate=`date +%F`
outputDir=tomopy.github.io/$currentDate/gpu

python -Om benchmarking.project \
  --poisson 500 \
  --trials 8 \
  --width 1446 \
  --num-angles 1500 \
  --phantom peppers \
  --output-dir $outputDir \

python -Om benchmarking.reconstruct \
  --ncore 4 \
  --max-iter 50 \
  --phantom peppers \
  --output-dir $outputDir \
  --parameters "[{'algorithm': 'sirt', 'accelerated':True, 'device': 'gpu', 'interpolation': 'NN'},{'algorithm': 'sirt', 'accelerated':True, 'device': 'gpu', 'interpolation': 'LINEAR'}]" \
  

python -Om benchmarking.summarize \
  --phantom peppers \
  --trials 8 \
  --output-dir $outputDir\
