python -Om benchmarking.project \
  --noise \
  --trials 32 \
  --size 1446 \
  --angles 1500 \
  --phantom peppers \
  --output-dir 1.5.x

python -Om benchmarking.reconstruct \
  --ncores 1 \
  --num-iter 300 \
  --iter-step 5 \
  --phantom peppers \
  --output-dir 1.5.x

python -Om benchmarking.summarize \
  --phantom peppers \
  --output-dir 1.5.x
