#!/bin/bash

python dataflow_search.py --dnnfile dnns/flowNetC.txt \
        --model_type 2D \
        --bufsize 1572864 \
        --bit_width 16 \
        --memory_bandwidth 25.6 \
        --sa_size 16 \
        --model_type 2D \
        --ifmap 960 576 6 \
        --static True \
        --buffer_partition 3 3 4

python dataflow_search.py --dnnfile dnns/flowNetC.txt \
        --model_type 2D \
        --search_method Constrained \
        --bufsize 1572864 \
        --bit_width 16 \
        --memory_bandwidth 25.6 \
        --sa_size 16 \
        --model_type 2D \
        --ifmap 960 576 6

python dataflow_search.py --dnnfile dnns/flowNetC.txt \
        --model_type 2D \
        --search_method Exhaustive \
        --bufsize 1572864 \
        --bit_width 16 \
        --memory_bandwidth 25.6 \
        --sa_size 16 \
        --model_type 2D \
        --ifmap 960 576 6

python dataflow_search.py --dnnfile dnns/flowNetC.txt \
        --model_type 2D \
        --search_method Constrained \
        --split True\
        --bufsize 1572864 \
        --bit_width 16 \
        --memory_bandwidth 25.6 \
        --sa_size 16 \
        --model_type 2D \
        --ifmap 960 576 6

python dataflow_search.py --dnnfile dnns/flowNetC.txt \
        --model_type 2D \
        --search_method Exhaustive \
        --split True\
        --bufsize 1572864 \
        --bit_width 16 \
        --memory_bandwidth 25.6 \
        --sa_size 16 \
        --model_type 2D \
        --ifmap 960 576 6

python dataflow_search.py --dnnfile dnns/flowNetC.txt \
        --model_type 2D \
        --search_method Constrained \
        --split True \
        --combine True \
        --bufsize 1572864 \
        --bit_width 16 \
        --memory_bandwidth 25.6 \
        --sa_size 16 \
        --model_type 2D \
        --ifmap 960 576 6

python dataflow_search.py --dnnfile dnns/flowNetC.txt \
        --model_type 2D \
        --search_method Exhaustive \
        --split True \
        --combine True \
        --bufsize 1572864 \
        --bit_width 16 \
        --memory_bandwidth 25.6 \
        --sa_size 16 \
        --model_type 2D \
        --ifmap 960 576 6
