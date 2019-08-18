# Dataflow optimizer

This is a DNN dataflow optimizer for a particular hardware accelerator, systolic
array. it is able to find a optimal or `approximately` optimal dataflow for a
particular DNN for some hardware constraints, such as bandwidth and SRAM, etc.

## Why

The goal of this optimizer are followings:

* First, this optimizer is to aim to find a close-to optimal configuration in
  order to minimize the latency and reduce the data traffic at the same time.
* Second, this optimizer explores different searching or optimization schemes,
  so that, we can show the trade-off between different optimization schemes.
* Third, this optimizer can automatically apply a special optimization for
  deconvolutions in the DNN pipeline, and we have different levels of opti-
  mizations to explore.

## What's inside

There are two main parts in this framework:
1. The overall framework to drive different optimizers.

* `dnn_optimizer.py`
* `dataflow_search.py`

2. The layer-level optimizers to optimize different

* `layer_base_method.py`
* `layer_optimizer.py`
* `layer_exhaustive_searcher.py`
* `deconv_exhaustive_searcher.py`

You can run the example dataflow optimization by `runner.sh`.

## How to use

To use the dataflow optimizer, you can run for helping info:

```
  $ python dataflow_search.py -h
```

By specify the configuration in the input option and a particular DNN network
you want to optimize, the optimizer will return a dataflow scheme for you. The
sample DNN networks are in the `/dnns` directory.

## Explanation of each option flags

There are three parts consisted all the flags

1. input and output files
  * `--dnnfile` : the actual dnn dataflow file to optimize.
  * `--outfile` : the file to dump all the result.

2. options that are related to search options
  * `--static` :  to set the flag if static partition the buffer enable, this
    flag will statically set the SRAM partition and optimize the entire dnn
    dataflow for that particular dnn dataflow, you also need to specify flag
    `buffer_partition` too.
  * `--split` : enable to apply our special optimization to split a regular
    deconvolution kernel into small sub-kernels and effectively avoid reduntant
    computation.
  * `--combine` : enable to effectively interleave the computation of the split
    sub-kernels during convolution.
  * `--model_type` : DNN model convolution type: 2D or 3D.
  * `--ifmap` : the initial ifmap dimemsion, a.k.a, input image, order: [W H C]
  * `--ifmap3d` : the initial ifmap dimemsion in 3D DNN, order: [W H D C]
  * `--buffer_partition` : the ifmap dimemsion, order: [I O W]
  * `--search_method` : there are three search options: "Constrained",
     "Exhaustive", "Combined", the first one is to use constrained optimization,
     "Exhaustive" is a exhaustive search with the combination of DP.
     "Combine" is to use static partition to set initial guess values for
     constrained optimization and then use constrained optimization.

3. other hardware configurations
  * `--bufsize`, the buffer size, or SRAM size in Bytes, e.g. 1048576.0.
  * `--memory_bandwidth`, the DRAM bandwidth in GB/s, e.g. 25.6.
  * `--sa_size`, the systolic array dimension, e.g. 16 stands for the systolic
    array dimension is 16 by 16.
  * `--bit_width`, Bit Width of each value (typically, 8-bit, 16-bit, 32-bit)

## how to specify a DNN configuration

We provide a simple way to To specify the configuration (or architecture) of
each DNN layer, the example is shown in `/dnns.

The layer parameters are separated by `,`, the order of the specification is:
ofmap channels, kernel height, hernel width, stride, flag to indicate whether
it is a deconvolution layer.


