# Benchmarking of various tokenizers

## Preparation

```
% git submodule update --init
% ./download_resources.sh
% ./compile_all.sh
```

## Measurement

```
% ./run_all.sh 2>&1 | tee ./results
% ./stats.py < ./results
```
