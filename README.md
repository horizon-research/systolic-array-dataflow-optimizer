# Systolic array dataflow optimizer

This is a DNN dataflow optimizer for a particular hardware accelerator, i.e., systolic
array. It is able to find an optimal or an approximately optimal dataflow for a
particular DNN given hardware constraints, such as bandwidth and SRAM, etc.
This repository is the artifact of our paper [ASV: Accelerated Stereo Vision System](https://www.cs.rochester.edu/horizon/pubs/micro19-tigris.pdf) published at MICRO 2019.

## What's new

The goals of this optimizer are:

* First, this optimizer is to aim to find a close-to optimal configuration in
  order to minimize the latency and reduce the data traffic at the same time.
* Second, this optimizer explores different searching or optimization schemes,
  so that, we can show the trade-off between different optimization schemes.
* Third, this optimizer can automatically apply a special optimization for
  deconvolutions in the DNN pipeline, and we have different levels of optimizations
  to explore.

## What's inside

There are two main parts in this framework:
1. The overall framework to drive different optimizers.

* `dataflow_search.py`
* `dnn_optimizer.py`

2. The layer-level optimizers to optimize different

* `layer(3d)_base_method.py`
* `layer(3d)_static_method.py`
* `layer(3d)_optimizer.py`
* `layer(3d)_exhaustive_searcher.py`
* `deconv_exhaustive_searcher.py`

## How to use

Pre-requisite packages: scipy, numpy, sys, math, pprint

To use the dataflow optimizer, you can run for helping info:

```
  $ python dataflow_search.py -h
```

By specify the configuration in the input option and a particular DNN network
you want to optimize, the optimizer will return a dataflow scheme for you. The
sample DNN networks are in the `/dnns` directory.

A simple example of using this tool on a dnn.
```
  $ python dataflow_search.py --dnnfile dnns/flowNetC.txt \
        --model_type 2D \
        --search_method Constrained \
        --split True\
        --bufsize 1572864 \
        --bit_width 16 \
        --memory_bandwidth 25.6 \
        --sa_size 16 \
        --model_type 2D \
        --ifmap 960 576 6
```
This will load the DNN network from `flowNetC.txt` and search the DNN dataflow
using *Constrained optimization*. You can use `--search_method` option to specify
what kind of search method to use. We provide two different options, one is
`Constrained`, which uses constrained optimization, the other one is `Exhaustive`,
which uses exhaustive search. This command also specifies the ifmap size to be
960-576-6 (width-height-channel). 

In this command, we also need to provide the hardware configuration. `bufsize` specifies that 
the on-chip buffer size is *1572864* bytes. `memory_bandwidth` specifies the memory
bandwidth is *25.6* GB/s. `sa_size` specifies that the systolic array size is *16-by-16*.
The `bitwidth` specifies the number of bits used to represent the numerical precision
for a single number. 

For details of other flags, please see the explanation below.

The dataflow optimization will print the result as a JSON-like format. The example result
is shown below:

```
{'dnn': [{'Deconv?': False,                                  <--------- DNN architecture
          'ifmap': [960, 576, 6],
          'kernel': [7, 7],
          'out_channel': 64,
          'stride': 2,
          'type': '2D'},
          ......
          
         {'Deconv?': True,
          'ifmap': [120.0, 72.0, 128],
          'kernel': [5, 5],
          'out_channel': 64,
          'stride': 1,
          'type': '2D'}],
 'dnn_result': [{'data': {'Deconv?': False,                 <-------- optimization result
                          'ifmap': [960, 576, 6],
                          'kernel': [7, 7],
                          'ofmap': [480.0, 288.0, 64],
                          'out_channel': 64,
                          'stride': 2,
                          'type': '2D'},
                 'result': {'Bound': 'C',
                            'Tile size': [1.0, 8.0, 5.0],               # tiling schedule
                            'buffer_utilization': 0.7890045166015626,
                            'c_0, w_0, h_0': [64, 120, 115],            # tile size
                            'systolic_array_utilization': 1.0,
                            'total_cycle': 40642560,                    # execution cycles
                            'total_transfer': 85029312}},               # DRAM access (in Bytes)
                ......
                
                {'data': {'Deconv?': True,
                          'ifmap': [120.0, 72.0, 128],
                          'kernel': [5, 5],
                          'out_channel': 64,
                          'stride': 1,
                          'type': '2D'},
                 'result': [{'Bound': 'C',
                             'Tile size': [1.0, 2.0, 1.0],              # tiling schedule
                             'buffer_utilization': 0.6793619791666666,
                             'c_0, w_0, h_0': [64, 60, 72],             # tile size
                             'systolic_array_utilization': 1.0,
                             'total_cycle': 6912000,                    # execution cycles
                             'total_transfer': 2690048}]}],             # DRAM access (in Bytes)
 'method': 'Constrained',
 'schedule': {'combine': True, 'split': True, 'static': False},         # optimization flags
 'system_info': {'bit_width': 16.0,
                 'bufsize': 1572864.0,
                 'memory_bandwidth': 25.6,
                 'sa_size': 16.0}}
```

The result shows a couple of fields:
  * `method` : the method you specified for dataflow search.
  * `schedule` : the optimization options you specified for deconvolution.
    Please check out our paper for more details one this.
  * `system_info` : this specifies the hardware configurations.
  * `dnn` : the architecture of you DNN network.
  * `dnn_result` : the optimization result for your dnn. 

You can run more examples of dataflow optimization by `runner.sh`.

### How to specify a DNN configuration

We provide a simple way to specify the configuration (or architecture) of
each DNN layer, the example is shown in `/dnns`.

The layer parameters are separated by `,`, the order of the specification is:
ofmap channels, kernel height, kernel width, stride, flag to indicate whether
it is a deconvolution layer. 

A simple example for 2D DNN is shown below:
```
# ofmap channels, kernel height, kernel width, stride,deconv?
64,7,7,2,False
128,5,5,2,False
...

128,5,5,1,True
64,5,5,1,True
```

Asimple example for 3D DNN is shown here:
```
# ofmap channels, kernel height, kernel width, kernel depth, stride, deconv?
32,3,3,3,1,False
32,3,3,3,1,False
...

64,3,3,3,1,True
32,3,3,3,1,True
```

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

## Citing

This project is a artifact of our 2019 MICRO paper:

Y. Feng,  P. Whatmough, and Y. Zhu, "ASV: Accelerated Stereo Vision System", In Proc. of MICRO, 2019.

Please kindly consider citing this paper in your publications if it helps your research.
```
@inproceedings{yu2019asv,
  title={ASV: Accelerated Stereo Vision System},
  author={Feng, Yu and Whatmough, Paul and Zhu, Yuhao},
  booktitle={Proceedings of the 52th International Symposium on Microarchitecture},
  year={2019},
  organization={ACM}
}
```
