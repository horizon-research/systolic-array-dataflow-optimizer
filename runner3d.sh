#!/bin/bash

python dataflow_search.py --dnnfile dnns/test3d.txt \
        --model_type 3D \
        --search_method Constrained \
        --bufsize 1572864 \
        --bit_width 16 \
        --memory_bandwidth 25.6 \
        --sa_size 16 \
        --model_type 3D \
        --ifmap3d 480 288 96 64

python dataflow_search.py --dnnfile dnns/test3d.txt \
        --model_type 3D \
        --search_method Exhaustive \
        --bufsize 1572864 \
        --bit_width 16 \
        --memory_bandwidth 25.6 \
        --sa_size 16 \
        --model_type 3D \
        --ifmap3d 480 288 96 64

python dataflow_search.py --dnnfile dnns/test3d.txt \
        --model_type 3D \
        --search_method Constrained \
        --split True\
        --bufsize 1572864 \
        --bit_width 16 \
        --memory_bandwidth 25.6 \
        --sa_size 16 \
        --model_type 3D \
        --ifmap3d 480 288 96 64

python dataflow_search.py --dnnfile dnns/test3d.txt \
        --model_type 3D \
        --search_method Exhaustive \
        --split True\
        --bufsize 1572864 \
        --bit_width 16 \
        --memory_bandwidth 25.6 \
        --sa_size 16 \
        --model_type 3D \
        --ifmap3d 480 288 96 64

