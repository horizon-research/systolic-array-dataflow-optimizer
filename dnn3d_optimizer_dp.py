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
    from layer3d_static import optimize3d, setup_hardware3d
else:
    from layer_optimizer import optimize, setup_hardware
    from layer3d_optimizer import optimize3d, setup_hardware3d

# from layer3d_optimizer import optimize3d, setup_hardware3d

# a list to store the dnn configuration 
dnn = []
dnn3d = []

# a list to store all the optimization results
results = []

# import dnn network descrtiption into the system;
# the format for one DNN layer is: 
# (width, height, in_channel, out_channel,
#  kenrel_width, kernel_height, stride, Deconv?)
def import_dnn(filename=None):
    # clear all the previous contents;
    del dnn[:]
    switch = False
    ifmap_dim = [960, 576, 6]
    # ifmap3d_dim = [480, 288, 96, 64]  # this is for GC_Net
    ifmap3d_dim = [240, 144, 48, 64]    # this is for PSMNet
    weight_dim = []
    weight3d_dim = []

    # The weight input format as follows: 
    # [out_channel,kenrel_width,kernel_height,stride,Deconv?]
    for line in open(filename):
        if len(line) <= 1:
            switch = True
            continue

        # check if we need to switch to 3d Conv.
        if not switch: 
            ls = line.strip().split(",")
            weight_dim.append([int(ls[0]), int(ls[1]), int(ls[2]),\
                             int(ls[3]), ls[4] == 'True'])
        else:
            ls = line.strip().split(",")
            weight3d_dim.append([int(ls[0]), int(ls[1]), int(ls[2]),\
                             int(ls[3]), int(ls[4]), ls[5] == 'True'])

    # since we know there is no case that 2d Deconv in 3d stereo vision DNN,
    # we don't consider 2D Deconv compute here. 
    for w in weight_dim:
        # first append the necessary information to compute this Conv layer 
        dnn.append(list(ifmap_dim+w))
        # for Conv, scale down the ifmap dimemsion by its stride;
        ifmap_dim = [ifmap_dim[0]/w[-2], ifmap_dim[1]/w[-2], w[0]]

    # next, let's consider 3d dnn.
    for w in weight3d_dim:
        # first append the necessary information to compute this Conv layer 
        dnn3d.append(list(ifmap3d_dim+w))
        # if it is Deconv;
        if w[-1]:
            # now, increase the deconv ofmap by two, as default, 
            # we only consider stride fo 2 in Deconv
            ifmap3d_dim = [ifmap3d_dim[0]*2, ifmap3d_dim[1]*2, ifmap3d_dim[2]*2, w[0]]
        else: 
            # if it is Conv, scale down the ifmap dimemsion by stride;
            ifmap3d_dim = [ifmap3d_dim[0]/w[-2], ifmap3d_dim[1]/w[-2], \
                            ifmap3d_dim[2]/w[-2], w[0]]


# The hardware constraints are:
#   1. the on-chip buffer size; 
#   2. the memory bandwidth; (Unit in bytes/cycle) 
#   3. the systolic array size; 8388608.0
def hardware_constraints(sa_size=16.0, mem_bw=16.0, buf=1048576.0):
    systolic_arr_size = sa_size;
    memory_bandwidth = mem_bw;
    buffer_size = buf;
    return [systolic_arr_size, memory_bandwidth, buffer_size, 0.3, 0.3, 0.4]

# Optimize 3D Deconvlution:
def opti_deconv3d(layer):
    # collect individual result from sub_kernels
    subs = []
    # number of same shape sub-kernels
    num = [1,3,3,1]
    sub_shape = [[2,2,2], [2,2,1], [2,1,1], [1,1,1]]
    # In our case, we only handle 3x3x3 Deconv, 
    # although, we can generalize into other cases;

    # check if we want to combine sub-kernels;
    if enable_combine:

        for i in range(4):
            sub = list(layer)
            # change the kernel shape
            sub[5:8] = sub_shape[i]
        sub[4] = sub[4]*num[i]
        subs.append(sub)
        ret = optimize_deconv3d(subs)
    else:
        for i in range(4):
            sub = list(layer)
            # change the kernel shape
            sub[5:8] = sub_shape[i]

            res = optimize3d(sub)
            res[0] = res[0]*num[i]
            res[1] = res[1]*num[i]
            subs.append(res)

        ret = [0, 0, 0, 0]
        for item in subs:
            print("[sub-kernel]",item)
            ret = [x+y for x,y in zip(ret,item)]
        ret[2] /= len(subs)
        ret[3] /= len(subs)
    
    # sum all the results
    return ret

def opti_dnn3d():
    ret = []
    # optimize for each layer
    for layer in dnn3d:
        # check if this layer is Deconv, True == YES
        if layer[-1] == True:
            # if split the deconv into smaller ones
            if enable_split:
                # split into smaller ones and process in a batch;
                if enable_combine:
                    ret.append(optimize3d(layer))
                else:
                    # split into smaller ones and process individually
                    ret.append(opti_deconv3d(layer))
            else:
                # start to optimize ordinary Conv layer.
                tmp = list(layer)
                # scale down the ifmap to the ifmap based on the stride size.
                tmp[0] = layer[0]*2
                tmp[1] = layer[1]*2
                tmp[2] = layer[2]*2
                ret.append(optimize3d(tmp))
        else:
            # start to optimize ordinary Conv layer.
            tmp = list(layer)
            # scale down the ifmap to the ifmap based on the stride size.
            tmp[0] = layer[0]/layer[-2]
            tmp[1] = layer[1]/layer[-2]
            tmp[2] = layer[2]/layer[-2]
            ret.append(optimize3d(tmp))

    return ret

# the main routine of optimizing the dnn.
def opti_dnn():
    global results
    # clear the result first
    del results[:]

    # optimize for each layer
    for layer in dnn:
        print("[Layer]",layer)
        # start to optimize ordinary Conv layer.
        tmp = list(layer)
        # scale down the ifmap to the ifmap based on the stride size.
        tmp[0] = layer[0]/layer[-2]
        tmp[1] = layer[1]/layer[-2]
        results.append(optimize(tmp))

    opti_3d = None
    # if static_schedule:
    #     res = []
    #     for i in range(1, 3):
    #         config = hardware_constraints()
    #         config[3:6] = [i/(100.0+i*0.3), i/(100.0+i*0.3)*0.3,\
    #                         (100.0-1.3*i)/(100.0+i*0.3)]
    #         config[3:6] = [0.3,0.3,0.4]
    #         setup_hardware3d(config)
    #         print(config)
    #         res.append([(i, (100-i)), opti_dnn3d()])

    #     opti_cycle = sum([ i[1] for i in res[0][1] ])
    #     opti_3d = res[0]
    #     for [k, v] in res:
    #         total_cycle = sum([ i[1] for i in v ])
    #         if (total_cycle < opti_cycle):
    #             opti_3d = v
    #             opti_cycle = total_cycle
    #         print(k, np.mean([ i[3] for i in v ]))
    # else:
    #     results = results + opti_dnn3d()
    results = results + opti_dnn3d()

    if opti_3d != None:
        results = results + opti_3d

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
        setup_hardware3d(config)
        res = opti_dnn()
        for j in range(len(res)):
            if res[j] is None:
                res[j] = arr[-1][j]
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
    for i in range(low, high):
        config = hardware_constraints()
        config[1] = float(pow(2, i))
        setup_hardware(config)
        setup_hardware3d(config)
        res = opti_dnn()
        for j in range(len(res)):
            if res[j] is None:
                res[j] = arr[-1][j]
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
        # print(config[2]*pow(1.2, size)/3.0)
        # config[2] = config[2]*scale*size
        config[2] = config[2]*scale*pow(1.2, size)
        setup_hardware(config)
        setup_hardware3d(config)
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
    plot_buf_size(res, low, high, step, 1, scale)
    plot_buf_cycle(res, low, high, step, 1, scale)


if __name__== '__main__':
    # import the dnn
    # import_dnn("dnns/GC_Net.txt")
    import_dnn("dnns/PSMNet.txt")
    for ln in dnn3d:
        print(ln)
    # exit()
    # check which characterization you want to proceed
    if detail_profile:
        # set up the hardware configuration
        setup_hardware(hardware_constraints())
        # set up the hardware configuration
        setup_hardware3d(hardware_constraints())
        # start the optimization main routine
        res = opti_dnn()
        # plot the result of each layer
        plot_util_dnn(res, suffix)
        profile_layer_cycle(res, suffix)
    else:
        # profile systolic array size impacts
        # profile_sa_size(8, 96, 8)
        # profile bandwidth impacts
        # profile_bw_size(-5, 12)
        # profile buffer size impacts
        profile_buf_size(1, 12, 1, 0.1)
        # profile_buf_size(4, 25, 1, 0.01)




