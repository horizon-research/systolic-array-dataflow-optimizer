#!/usr/bin/python2.7
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
import numpy as np
import scipy
import sys


# import my own modules
import layer_optimizer
import layer_static_method
import layer_exhaustive_searcher
import deconv_exhaustive_searcher

import layer3d_optimizer
import layer3d_exhaustive_searcher

method = None
buffer_partition = None
enable = {
    "static" : False,
    "combine" : False,
    "split" : False,
}

def setup(meta_data, hardware_constraints):
    global enable, method, buffer_partition
    # define the search method
    method = meta_data["method"]

    if meta_data["schedule"]["static"] and \
        "buffer_partition" not in meta_data:
        raise Exception("The static scheduling is not supported"
            " without specifying the buffer partition.")

    if "buffer_partition" in meta_data:
        buffer_partition = meta_data["buffer_partition"]

    # set the schedule policy
    enable["static"] = meta_data["schedule"]["static"]
    enable["combine"] = meta_data["schedule"]["combine"]
    enable["split"] = meta_data["schedule"]["split"]

def single_layer_optimization(data, sys_info):
    global method, enable, buffer_partition
    # if "static" option is enabled, it will be prioritized
    if enable["static"]:
      return layer_static_method.\
          LayerStaticMethod(data, sys_info, buffer_partition).optimize()

    # check the potential method we use here.
    if method == "Constrained":
        if data["type"] == "2D":
            return layer_optimizer.\
                LayerOptimizer(data, sys_info).optimize()
        else:
            return layer3d_optimizer.\
                Layer3dOptimizer(data, sys_info).optimize()
    elif method == "Exhaustive":
        if data["type"] == "2D":
            return layer_exhaustive_searcher.\
                LayerExhaustiveSearcher(data, sys_info).optimize()
        else:
            return layer3d_exhaustive_searcher.\
                Layer3dExhaustiveSearcher(data, sys_info).optimize()
    elif method == "Combined":
        return layer_optimizer.\
            LayerOptimizer(data, sys_info, True).optimize()
    else:
        raise Exception("Unknown search method: {}".format(method))

def single_combine_optimization(data, sys_info):
    global method
    if method == "Constrained":
      return layer_optimizer.\
            LayerOptimizer(data, sys_info).optimize()
    elif method == "Exhaustive":
        return deconv_exhaustive_searcher.\
            DeconvExhaustiveSearcher(data, sys_info).optimize()
    elif method == "Combined":
        return layer_optimizer.\
            LayerOptimizer(data, sys_info, True).optimize()
    else:
        raise Exception("Unknown search method: {}".format(method))

def sub_kernel_sizes(layer):
    add_one = [(i+1)/2 for i in layer["kernel"]]
    sub_one = [i/2 for i in layer["kernel"]]

    sizes = [[]]
    for i in range(len(layer["kernel"])):
      tmp = []
      for j in sizes:
        e1 = list(j) + [add_one[i]]
        e2 = list(j) + [sub_one[i]]
        tmp += [e1, e2]

      sizes = tmp

    return sizes

def single_split_optimization(layer, sys_info):
    subs = []

    # iterate all possible sub_kernels
    for sub_size in sub_kernel_sizes(layer):
        sub = dict(layer)
        sub["kernel"] = sub_size
        subs.append(single_layer_optimization(sub, sys_info))

    return subs

def opti_deconv(layer, sys_info):
    global method, enable
    # collect individual result from sub_kernels
    subs = []

    # if the convolution size is odd;
    if layer["kernel"][0]%2 == 1:
        if enable["combine"]:
            subs.append(single_combine_optimization(layer, sys_info))
        else:
            subs = single_split_optimization(layer, sys_info)
    # if the convolution size is even;
    else:
        sub = dict(layer)
        sub["kernel"][0] = sub["kernel"][0]/2
        sub["kernel"][1] = sub["kernel"][1]/2
        if enable_combine:
            # this will consider four same-size sub-kernels
            # as one sub-kernel with more channels
            sub["out_channel"] = sub["out_channel"]*4
            subs.append(single_layer_optimization(sub4, sys_info))
        else:
            # without combining sub-kernels
            res = single_layer_optimization(sub, sys_info)
            # times 4 of each individual sub-kernel"s
            # memory traffic and cycles.
            res["total_traffic"] = res["total_traffic"]*4
            res["total_cycle"] = res["total_cycle"]*4
            subs.append(res)

    return subs

# the main routine of optimizing the dnn.
def opti_dnn(meta_data, hardware_constraints):
    # set up the configurations;
    setup(meta_data, hardware_constraints)
    dnn = meta_data["dnn"]
    sys_info = meta_data["system_info"]

    results = []

    # optimize for each layer
    for i in range(len(dnn)):
        layer = dnn[i]
        # start to optimize ordinary Conv layer.
        data = dict(layer)

        # check if this layer is Deconv, True == YES
        if layer["Deconv?"] == True:
            if enable["split"]:
                # if split the deconv into smaller ones
                results.append({
                        "data" : data,
                        "result" : opti_deconv(layer, sys_info)
                        })
            else:
                data["ofmap"] = [0] * len(data["ifmap"])
                # scale up the ifmap to the ifmap based on the stride size.
                for i in range(len(data["ifmap"])-1):
                    data["ifmap"][i] = layer["ifmap"][i]*2/layer["stride"]
                    data["ofmap"][j] = layer["ifmap"][j]/layer["stride"]

                # the last element is ofmap channel, so treat it separately
                data["ofmap"][-1] = data["out_channel"]

                # add the result
                results.append({
                        "data" : data,
                        "result" : single_layer_optimization(data, sys_info)
                        })
        else:
            data["ofmap"] = [0] * len(data["ifmap"])
            # scale down the ifmap to the ifmap based on the stride size.
            for j in range(len(data["ifmap"])-1):
                data["ofmap"][j] = layer["ifmap"][j]/layer["stride"]

            # the last element is ofmap channel, so treat it separately
            data["ofmap"][-1] = data["out_channel"]

            results.append({
                        "data" : data,
                        "result" : single_layer_optimization(data, sys_info)
                        })

    return results
