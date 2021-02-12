#!/bin/bash
set -e
currentDate=`date +%F`
outputDir=tomopy.github.io/$currentDate

python -Om benchmarking.project \
  --poisson 5 \
  --trials 4 \
  --width 800 \
  --num-angles 32 \
  --phantom peppers \
  --output-dir $outputDir \

python -Om benchmarking.reconstruct \
  --ncore 16 \
  --max-iter 50 \
  --phantom peppers \
  --parameters "[{'algorithm': 'mlem', 'num_iter': 5, 'accelerated': True, 'device': 'gpu', 'interpolation': 'NN'}, {'algorithm': 'mlem', 'num_iter': 5, 'accelerated':True, 'device': 'gpu', 'interpolation': 'LINEAR'},{'algorithm': 'mlem', 'num_iter': 5, 'accelerated':True, 'device': 'gpu', 'interpolation': 'CUBIC'},{'algorithm': 'sirt', 'num_iter': 5, 'accelerated':True, 'device': 'gpu', 'interpolation': 'NN'},{'algorithm': 'sirt', 'num_iter': 5, 'accelerated':True, 'device': 'gpu', 'interpolation': 'LINEAR'},{'algorithm': 'sirt', 'num_iter': 5, 'accelerated':True, 'device': 'gpu', 'interpolation': 'CUBIC'}]" \
  --output-dir $outputDir \

python -Om benchmarking.summarize \
  --phantom peppers \
  --trials 32 \
  --output-dir $outputDir\