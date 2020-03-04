#!/usr/bin/python2.7

# public library
import math
import numpy as np

class MultiLayerPerceptron(object):
    """docstring for MultiLayerPerceptron"""
    # info for systolic array
    A = None       # systolic array dimension

    # memory bandwith number of bytes can be transferred.
    B = None

    # on-chip buffer size
    buf_size = None

    # input layer dimension
    N = None        # numbers of feature (NumberOfPoints x NumberOfFeature)
    Ci = None       # channels for ifmap
    Co = None       # channels for ofmap

    # on-chip buffer size
    bufi_size = None
    bufo_size = None
    bufw_size = None

    """docstring for MultiLayerPerceptron"""
    def __init__(self, data, sys_info):
        self.data = data
        self.sys_info = sys_info
        self.A = sys_info["sa_size"]
        self.B = sys_info["memory_bandwidth"]/(sys_info["bit_width"]/8)
        self.buf_size = sys_info["bufsize"]

    def init_setup(self):
        layer_info = self.data
       
        # set up the new layer information
        [self.N, self.Ci] = layer_info["ifmap"]
        self.Co = layer_info["out_channel"]

        self.bufw_size = self.Co * self.Ci

    ###############################################################
    #                       general process                       #
    ###############################################################

    # compute buffer utilization
    def buffer_utilization(self, x):
        # buffer = ofmap + weights + ifmap
        return (x*self.Co + self.Ci*self.Co + x*self.Ci)

    # (ofmap + ifmap)*total_batch + (ofmap+weights)*Co/c_0
    def data_transfer(self, x):
        # calculate the total batch
        total_batch = math.ceil(float(self.N) / x)

        # ofmap, ifmap and kernel tile size
        ofmap_tile_size = self.Co * x
        ifmap_tile_size = self.Ci * x
        kernel_tile_size = self.Co*self.Ci

        # ofmap + ifmap transfer
        total_transfer = (ofmap_tile_size + ifmap_tile_size) * total_batch

        # add additional data transfer
        total_transfer += kernel_tile_size

        return total_transfer

    def systolic_array_utilization(self, x):
        A = self.A
        A_w_uiti = math.ceil(self.Co/math.ceil(float(self.Co)/A))

        total_usage = x * self.Co
        round_up_val = math.ceil(float(x/A))*A \
                     * math.ceil(float(self.Co)/A)*A

        # the pct of extra delay due to output-stationary
        delay_pct = float(self.Ci)/(self.Ci+A_w_uiti)

        return delay_pct * total_usage / round_up_val

    def compute_bound_cycle(self, util_rate):
        # total number of ops
        total_computation = (self.N*self.Ci*self.Co)

        # systolic array calculation capacity
        comp_cap = (self.A*self.A) * util_rate

        return total_computation / comp_cap

    def process_parameter(self, x):

        x = math.floor(x)
        bound = "C"
        # make the tile size even for every batch
        x_0 = min(self.N/math.ceil(self.N/round(x)), self.N)

        # (ofmap + ifmap)*total_batch + weights
        total_transfer = self.data_transfer(x_0)

        # compute the utilization of systolic array
        util_sys_arr = self.systolic_array_utilization(x_0)

        # compute the utilization of buffer
        util_buf = float(self.buffer_utilization(x_0))/self.buf_size

        if util_buf > 1.01:
            print("ERROR: the utilization of buffer is over 100%")
            exit()

        # calculate the amount of cycles of computing all elements.
        if self.compute_bound_cycle(util_sys_arr) > total_transfer/self.B:
            bound = "C"
            total_cycle = self.compute_bound_cycle(util_sys_arr)
        else:
            bound = "M"
            total_cycle = total_transfer/self.B

        ret = {
            "total_transfer": round(total_transfer),
            "total_cycle": round(total_cycle),
            "systolic_array_utilization": util_sys_arr,
            "buffer_utilization": util_buf,
            "x_0": x_0,
            "Bound" : bound
        }

        return ret

    # optimize one layer
    def optimize(self):
        self.init_setup()

        # if sum of bufi and bufw is over the self.buf_size
        # we should skip it.
        if self.bufw_size > self.buf_size:
            print("FAIL: the entire weight cannot be stored in buffer")
            exit()

        self.bufi_size = (self.buf_size - self.bufw_size)*self.Ci/(self.Ci+self.Co)
        self.bufo_size = (self.buf_size - self.bufw_size)*self.Co/(self.Ci+self.Co)

        # set the initial guess;
        x0 = self.A

        # let's see what percentage of ifmap can we fit into the buffer.
        while x0 < self.N and (x0+self.A)*self.Ci < self.bufi_size:
            x0 = x0 + self.A

        return self.process_parameter(x0)
