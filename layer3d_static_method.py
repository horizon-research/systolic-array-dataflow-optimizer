#!/usr/bin/python2.7

# public library
import math
import numpy as np

# my own module
from layer3d_base_method import *

###############################################################
#                       general process                       #
###############################################################
class Layer3dStaticMethod(Layer3dBaseMethod):

    # array to store the result from the four different results
    res = []

    def __init__(self, data, sys_info, buffer_partition=None):
        super(Layer3dStaticMethod, self).__init__(data, sys_info)
        self.rets = []
        if buffer_partition:
            # calculate buffer sizes
            self.bufi_size = self.buf_size* \
                buffer_partition[0]/sum(buffer_partition)
            self.bufo_size = self.buf_size* \
                buffer_partition[1]/sum(buffer_partition)
            self.bufw_size = self.buf_size* \
                buffer_partition[2]/sum(buffer_partition)

    # the main optimization routine;
    def opti_buffer(self):
        # set the initial guess;
        x0 = [self.A, self.A]

        # check if the initial configuration can hold the minimum requirements
        if ((x0[0]*self.K_h*self.K_w*self.K_d*self.Ci > self.bufw_size) or
            (self.S*self.S*self.S*x0[1]*self.Ci > self.bufi_size)):
            return

        # first, let's find the number of kernel we can put into buffer.
        while (x0[0]+self.A)*self.K_h*self.K_w*self.K_d*self.Ci < self.bufw_size:
            x0[0] = x0[0]+self.A

        # next let's see what percentage of ifmap can we fit into the buffer.
        if self.K_h*self.K_w*(2*self.S+self.W) >= self.bufi_size:
            while self.K_h*self.K_w*(self.S+x0[1]+self.A)*self.Ci < self.bufi_size:
                x0[1] = x0[1]+self.A
            # no need to optimize the buffer for ofmap, because it is
            # bounded ifmap.
            x = [x0[0], x0[1], 1]
        else:
            d_0 = 1
            while (self.K_h*self.K_w)*(self.S+self.W)*(self.S+self.H) \
                * (self.S+d_0+1)*self.Ci < self.bufi_size:
                d_0 += 1
            # no need to optimize the buffer for ofmap, because it is
            # bounded ifmap.
            x = [x0[0], self.W, self.H, d_0]

        self.process_parameter(x, False, False)
        self.process_parameter(x, False, True)
        self.process_parameter(x, True, False)
        self.process_parameter(x, True, True)

    def optimize_one_buffer_partition(self):
        # if sum of bufi and bufw is over the self.buf_size
        # we should skip it.
        if (self.bufi_size + self.bufw_size) > self.buf_size:
            return

        self.bufo_size = self.buf_size - self.bufi_size - self.bufw_size
        # both cases are possible;
        self.opti_buffer()

        if len(self.res) == 0:
            return

        # choose the larger value as the bottleneck
        row_major_res = None
        if (self.res[0]["total_cycle"] < self.res[1]["total_cycle"]):
            row_major_res = self.res[1]
        else:
            row_major_res = self.res[0]

        # choose the larger value as the bottleneck
        channel_major_res = None
        if (self.res[2]["total_cycle"] < self.res[3]["total_cycle"]):
            channel_major_res = self.res[3]
        else:
            channel_major_res = self.res[2]

        # return the shortest value as the perferred compute ordering.
        ret = None
        if (row_major_res["total_cycle"] < channel_major_res["total_cycle"]):
            ret = dict(row_major_res)
        else:
            ret = dict(channel_major_res)

        self.rets.append(ret)

    # optimize one layer
    def optimize(self):
        self.init_setup()

        # start the optimization
        self.res = []
        self.optimize_one_buffer_partition()

        # find the best result
        ret  = dict(self.rets[0])

        for item in self.rets:
            if ret["total_cycle"] > item["total_cycle"]:
                ret = dict(item)
            if ret["total_cycle"] == item["total_cycle"] and \
                ret["total_transfer"] > item["total_transfer"]:
                ret = dict(item)

        return ret
