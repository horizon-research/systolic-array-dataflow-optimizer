
import argparse
import numpy as np
import scipy
import sys
import pprint
import json

# import my own modules
import dnn_optimizer

# setup the argument parser
argparser = argparse.ArgumentParser("dataflow_search.py")
# input dnn file and output result
argparser.add_argument("--dnnfile", required=True)
argparser.add_argument("--outfile", help="output file to dump all the results")

# other search options
argparser.add_argument("--static", type=bool, default=False,
                        help="static partition the buffer without dynamically changing")
argparser.add_argument("--split", type=bool, default=False,
                        help="enable to split the convolution kernel into small sub-kernel")
argparser.add_argument("--combine", type=bool, default=False,
                        help="enable to combine the sub-kernels durting compute")
argparser.add_argument("--model_type", default="2D", choices=["2D", "3D"],
                        help="DNN model convolution type: 2D or 3D.")
argparser.add_argument("--ifmap", nargs="+", type=int, required=False,
                        help="the ifmap dimemsion, order: [W H C]")
argparser.add_argument("--ifmap3d", nargs="+", type=int, required=False,
                        help="the ifmap dimemsion, order: [W H D C]")
argparser.add_argument("--buffer_partition", nargs="+", type=float,
                        help="the ifmap dimemsion, order: [I O W]")
argparser.add_argument("--search_method", default="Constrained",
                        choices=["Constrained", "Exhaustive", "Combined"],
                        help="Dataflow search methods: constraint optoimization"
                        ", exhaustive search or combining both.")

# other hardware configurations
argparser.add_argument("--bufsize", type=float, default=1048576.0*1.5,
                        help="in Btyes")
argparser.add_argument("--memory_bandwidth", type=float, default=6.4*4,
                        help="in GB/s")
argparser.add_argument("--sa_size", type=float, default=16,
                        help="Systolic array size")
argparser.add_argument("--bit_width", type=float, default=16,
                        help="Bit Width of each value (typically, 8-bit, 16-bit, 32-bit)")


args = argparser.parse_args()

# import dnn network descrtiption into the system;
# the format for one 2D DNN layer is:
# (width, height, in_channel, out_channel,
#  kenrel_width, kernel_height, stride, Deconv?)
#
# for 3D DNN layer is
# (width, height, disparity, in_channel, out_channel,
#  kenrel_width, kernel_height, kernel_disp, stride, Deconv?)
def import_dnn(filename, ifmap_dim, ifmap3d_dim):
    # a list to store the dnn configuration
    dnn = []
    weight_dim = []

    is_2d_layer = True

    # The weight input format as follows:
    # [out_channel,kenrel_width,kernel_height,stride,Deconv?]
    for line in open(filename):
        if len(line) <= 1:
            continue
        ls = line.strip().split(",")

        if len(ls) == 5:
            dnn.append({"ifmap" : ifmap_dim,
                        "out_channel" : int(ls[0]),
                        "kernel" : [int(ls[1]), int(ls[2])],
                        "stride" : int(ls[3]),
                        "Deconv?" : ls[4] == "True",
                        "type" : "2D"})

            prev_layer = dnn[-1]

            if prev_layer["Deconv?"]:
                # increase the deconv ofmap by two, as default,
                # we only consider stride of 1
                ifmap_dim = [ifmap_dim[0]*2/prev_layer["stride"], \
                             ifmap_dim[1]*2/prev_layer["stride"], \
                             prev_layer["out_channel"]]
            else:
                # if it is Conv, scale down the ifmap dimemsion by stride;
                ifmap_dim = [ifmap_dim[0]/prev_layer["stride"], \
                            ifmap_dim[1]/prev_layer["stride"], \
                            prev_layer["out_channel"]]

        else:
            dnn.append({"ifmap" : ifmap3d_dim,
                        "out_channel" : int(ls[0]),
                        "kernel" : [int(ls[1]), int(ls[2]), int(ls[3])],
                        "stride" : int(ls[4]),
                        "Deconv?" : ls[5] == "True",
                        "type" : "3D"})

            prev_layer = dnn[-1]

            if prev_layer["Deconv?"]:
                # increase the deconv ofmap by two, as default,
                # we only consider stride of 1
                ifmap3d_dim = [ifmap3d_dim[0]*2/prev_layer["stride"], \
                               ifmap3d_dim[1]*2/prev_layer["stride"], \
                               ifmap3d_dim[2]*2/prev_layer["stride"], \
                               prev_layer["out_channel"]]
            else:
                # if it is Conv, scale down the ifmap dimemsion by stride;
                ifmap3d_dim = [ifmap3d_dim[0]/prev_layer["stride"], \
                               ifmap3d_dim[1]/prev_layer["stride"], \
                               ifmap3d_dim[2]/prev_layer["stride"],  \
                               prev_layer["out_channel"]]

    return dnn

# The hardware constraints are:
#   1. the on-chip buffer size;
#   2. the memory bandwidth; (Unit in bytes/cycle)
#   3. the systolic array size;
def hardware_constraints(sa_size=16.0, mem_bw=6.4*4, buf=1048576.0*1.5, bit_width=16.0):
    systolic_arr_size = sa_size;
    memory_bandwidth = mem_bw;
    buffer_size = buf;
    return [systolic_arr_size, memory_bandwidth, buffer_size, bit_width]

def system_config(args, meta_data):
    # set up the search methods
    meta_data["method"] = args.search_method
    # setup the system configuration;
    meta_data["schedule"] = {}
    meta_data["schedule"]["static"] = args.static
    meta_data["schedule"]["split"] = args.split
    meta_data["schedule"]["combine"] = args.combine
    if args.buffer_partition:
        meta_data["buffer_partition"] = args.buffer_partition

    # setup the system;
    meta_data["system_info"] = {}
    meta_data["system_info"]["bufsize"] = args.bufsize
    meta_data["system_info"]["memory_bandwidth"] = args.memory_bandwidth
    meta_data["system_info"]["sa_size"] = args.sa_size
    meta_data["system_info"]["bit_width"] = args.bit_width

    return meta_data

def calculate_overall_performance(meta_data):
    res = {
        "total_cycle" : 0.0,
        "total_transfer" : 0.0
    }
    for data in meta_data["dnn_result"]:
        if isinstance(data['result'], list):
            for item in data['result']:
                res["total_cycle"] += item["total_cycle"]
                res["total_transfer"] += item["total_transfer"]
        else:
            res["total_cycle"] += data['result']["total_cycle"]
            res["total_transfer"] += data['result']["total_transfer"]

    return res


if __name__== "__main__":
    # initialize the result data;
    meta_data = {}

    # setup system configuration;
    meta_data = system_config(args, meta_data)

    # import the dnn
    dnn = import_dnn(args.dnnfile, args.ifmap, args.ifmap3d)
    meta_data["dnn"] = dnn
    hw_constraints = hardware_constraints(sa_size=args.sa_size,
                                          mem_bw=args.memory_bandwidth,
                                          buf=args.bufsize,
                                          bit_width=args.bit_width)

    # start the optimization main routine
    meta_data["dnn_result"] = dnn_optimizer.opti_dnn(meta_data, hw_constraints)

    meta_data["overall_result"] = calculate_overall_performance(meta_data)

    pprint.pprint(meta_data)
