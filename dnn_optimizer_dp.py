#!/usr/bin/python2.7
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
import numpy as np
import scipy;

# if profile in details for one particular network.
detail_profile = True

# if it is static schedule the buffer 
static_schedule = False

# set the switch whether we want to split the Deconv
enable_split = True

# if we combine two different sub-kernels and optimize them 
# together, then, enable this switch
enable_combine = True

# add suffix to every plot for one configuration profiling
suffix = "tmp"

# import my own modules
from dnn_analysis import *

# depends on scheduling, imports different optimizer
if static_schedule:
    from layer_static import optimize, setup_hardware
else:
    from layer_optimizer import optimize, setup_hardware

from layer_dp import optimize_deconv

# a list to store the dnn configuration 
dnn = []

# a list to store all the optimization results
results = []

# import dnn network descrtiption into the system;
# the format for one DNN layer is: 
# (width, height, in_channel, out_channel,
#  kenrel_width, kernel_height, stride, Deconv?)
def import_dnn(filename=None):
    # clear all the previous contents;
    del dnn[:]
    ifmap_dim = [960, 576, 6]
    weight_dim = []

    # The weight input format as follows: 
    # [out_channel,kenrel_width,kernel_height,stride,Deconv?]
    for line in open(filename):
        ls = line.strip().split(",")
        weight_dim.append([int(ls[0]), int(ls[1]), int(ls[2]),\
                             int(ls[3]), ls[4] == 'True'])

    for w in weight_dim:
        # first append the necessary information to compute this Conv layer 
        dnn.append(list(ifmap_dim+w))
        # if it is Deconv;
        if w[-1]:
            # increase the deconv ofmap by two, as default,
            # we only consider stride fo 2
            ifmap_dim = [ifmap_dim[0]*2, ifmap_dim[1]*2, w[0]]
        else: 
            # if it is Conv, scale down the ifmap dimemsion by stride;
            ifmap_dim = [ifmap_dim[0]/w[-2], ifmap_dim[1]/w[-2], w[0]]


# The hardware constraints are:
#   1. the on-chip buffer size; 
#   2. the memory bandwidth; (Unit in bytes/cycle) 
#   3. the systolic array size;
def hardware_constraints(sa_size=16.0, mem_bw=16.0, buf=2097152.0):
    systolic_arr_size = sa_size;
    memory_bandwidth = mem_bw;
    buffer_size = buf;
    return [systolic_arr_size, memory_bandwidth, buffer_size]


def opti_deconv(layer):
    # collect individual result from sub_kernels
    subs = []

    # if the convolution size is odd;
    if layer[5]%2 == 1:
        sub1 = list(layer)
        sub1[4] = (sub1[4]+1)/2
        sub1[5] = (sub1[5]+1)/2
        sub2 = list(layer)
        sub2[4] = (sub2[4]+1)/2
        sub2[5] = (sub2[5]-1)/2
        sub3 = list(layer)
        sub3[4] = (sub3[4]-1)/2
        sub3[5] = (sub3[5]+1)/2
        sub4 = list(layer)
        sub4[4] = (sub4[4]-1)/2
        sub4[5] = (sub4[5]-1)/2
        
        if enable_combine:
            subs.append(optimize_deconv([sub1, sub2, sub3, sub4]))
        else:
            res1 = optimize(sub1)
            subs.append(res1)
            res2 = optimize(sub2)
            subs.append(res2)
            res3 = optimize(sub3)
            subs.append(res3)
            res4 = optimize(sub4)
            subs.append(res4)


    # if the convolution size is even;
    else:
        sub = list(layer)
        sub[4] = sub[4]/2
        sub[5] = sub[5]/2
        if enable_combine:
            # this will consider four same-size sub-kernels 
            # as one sub-kernel with more channels
            sub[3] = sub[3]*4
            subs.append(optimize(sub))
        else:
            # without combining sub-kernels 
            res = optimize(sub)
            # times 4 of each individual sub-kernel's
            # memory traffic and cycles.
            res[0] = res[0]*4
            res[1] = res[1]*4
            subs.append(res)

    ret = [0, 0, 0, 0]
    print(subs)
    if not enable_combine:
        for item in subs:
            ret = [x+y for x,y in zip(ret,item)]

        ret[2] /= len(subs)
        ret[3] /= len(subs)
    else:
        ret = subs[0]

    results.append(ret)
    # sum all the results
    return ret

# the main routine of optimizing the dnn.
def opti_dnn():
    global results
    # clear the result first
    del results[:]

    # optimize for each layer
    for layer in dnn:
        print("[Layer]",layer)

        # check if this layer is Deconv, True == YES
        if layer[-1] == True:
            if enable_split:
                # if split the deconv into smaller ones
                opti_deconv(layer)
            else:
                # start to optimize ordinary Conv layer.
                tmp = list(layer)
                # scale down the ifmap to the ifmap based on the stride size.
                tmp[0] = layer[0]*2
                tmp[1] = layer[1]*2
                results.append(optimize(tmp))
        else:
            # start to optimize ordinary Conv layer.
            tmp = list(layer)
            # scale down the ifmap to the ifmap based on the stride size.
            tmp[0] = layer[0]/layer[-2]
            tmp[1] = layer[1]/layer[-2]
            results.append(optimize(tmp))

    for res in results:
        print(res)

    return results

'''
The functions below are to profile the impacts of different bandwidth,
buffer size, and systolic array size on overall system.
'''
def profile_sa_size(low, high, step):
    arr = []
    # systolic_arr_size, memory_bandwidth, buffer_size
    for size in range(low, high+step, step):
        config = hardware_constraints()
        print("SIZE", size)
        config[0] = float(size)
        print(config)
        setup_hardware(config)
        res = opti_dnn()
        arr.append([item[0:4] for item in res])

    # gather results
    res = {"sa_avg":[], "sa_std":[], "buf_avg":[], "buf_std":[], \
            "cycle_avg": [], "traffic_avg": []}

    for ls in arr:
        res["sa_avg"].append(np.mean([i[2] for i in ls]))
        res["sa_std"].append(np.std([i[2] for i in ls]))
        res["buf_avg"].append(np.mean([i[3] for i in ls]))
        res["buf_std"].append(np.std([i[3] for i in ls]))
        res["cycle_avg"].append(np.mean([i[1] for i in ls]))
        res["traffic_avg"].append(np.mean([i[0] for i in ls]))

    print(res)
    plot_sa_size(res, low, high, step)
    plot_sa_cycle(res, low, high, step)

def profile_bw_size(low, high):
    arr = []
    # systolic_arr_size, memory_bandwidth, buffer_size
    config = hardware_constraints()
    for i in range(low, high):
        config[1] = float(pow(2, i))
        setup_hardware(config)
        res = opti_dnn()
        arr.append([item[0:4] for item in res])

    # gather results
    res = {"sa_avg":[], "sa_std":[], "buf_avg":[], "buf_std":[], \
            "cycle_avg": [], "traffic_avg": []}

    for ls in arr:
        res["sa_avg"].append(np.mean([i[2] for i in ls]))
        res["sa_std"].append(np.std([i[2] for i in ls]))
        res["buf_avg"].append(np.mean([i[3] for i in ls]))
        res["buf_std"].append(np.std([i[3] for i in ls]))
        res["cycle_avg"].append(np.mean([i[1] for i in ls]))
        res["traffic_avg"].append(np.mean([i[0] for i in ls]))

    plot_bw_size(res, low, high)
    plot_bw_cycle(res, low, high)

def profile_buf_size(low, high, step, scale):
    arr = []
    # systolic_arr_size, memory_bandwidth, buffer_size
    for size in range(low, high, step):
        config = hardware_constraints()
        # config[2] = config[2]*0.125*size
        config[2] = config[2]*scale*size
        print(config)
        setup_hardware(config)
        res = opti_dnn()
        arr.append([item[0:4] for item in res])

    # gather results
    res = {"sa_avg":[], "sa_std":[], "buf_avg":[], "buf_std":[], \
            "cycle_avg": [], "traffic_avg": []}

    for ls in arr:
        res["sa_avg"].append(np.mean([i[2] for i in ls]))
        res["sa_std"].append(np.std([i[2] for i in ls]))
        res["buf_avg"].append(np.mean([i[3] for i in ls]))
        res["buf_std"].append(np.std([i[3] for i in ls]))
        res["cycle_avg"].append(np.mean([i[1] for i in ls]))
        res["traffic_avg"].append(np.mean([i[0] for i in ls]))

    print(res)
    plot_buf_size(res, low, high, step, 2, scale)
    plot_buf_cycle(res, low, high, step, 2, scale)

if __name__== '__main__':
    # import the dnn
    import_dnn("dnns/flowNetC.txt")

    # check which characterization you want to proceed
    if detail_profile:
        # set up the hardware configuration
        setup_hardware(hardware_constraints())
        # start the optimization main routine
        res = opti_dnn()
        # plot the result of each layer
        plot_util_dnn(res, suffix)
        profile_layer_cycle(res, suffix)
    else:
        # profile systolic array size impacts
        profile_sa_size(8, 96, 8)
        # profile bandwidth impacts
        # profile_bw_size(-3, 10)
        # profile buffer size impacts
        # profile_buf_size(1, 30, 2, 0.2)




