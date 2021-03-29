#!/bin/bash
set -e
currentDate=`date +%F`
outputDir=tomopy.github.io/$currentDate/cpu

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
  --parameters "[{'algorithm': 'gridrec', 'filter_name': 'shepp'},{'algorithm':'art'}]"

python -Om benchmarking.summarize \
  --phantom peppers \
  --trials 8 \
  --output-dir $outputDir \
