#!/usr/bin/python2.7

# public library
import math
import numpy as np

# my own module
from layer3d_static_method import *

###############################################################
#                       general process                       #
###############################################################
class Layer3dExhaustiveSearcher(Layer3dStaticMethod):

    # array to store the result from the four different results
    res = []

    """docstring for LayerExhaustiveSearcher"""
    def __init__(self, data, sys_info):
        super(Layer3dExhaustiveSearcher, self).__init__(data, sys_info, None)
        self.rets = []

    # optimize one layer
    def optimize(self):
        self.init_setup()

        for i in range(1, 20):
            self.bufi_size = self.buf_size*i/20.0
            for j in range(1, 20):
                self.bufw_size = self.buf_size*j/20.0
                # optimize one buffer partition
                self.res = []
                self.optimize_one_buffer_partition()

        ret  = dict(self.rets[0])

        for item in self.rets:
            if ret["total_cycle"] > item["total_cycle"]:
                ret = dict(item)
            if ret["total_cycle"] == item["total_cycle"] and \
                ret["total_transfer"] > item["total_transfer"]:
                ret = dict(item)

        return ret

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

        # next let's see how much ifmap can we fit into the buffer.
        while self.S*self.S*self.S*(x0[1]+self.A)*self.Ci < self.bufi_size:
            x0[1] = x0[1]+self.A

        # no need to optimize the buffer for ofmap, because it is
        # bounded ifmap.
        x = [x0[0], min(round(x0[1]**(1.0/3)), self.W),
             min(round(x0[1]**(1.0/3)), self.H), min(round(x0[1]**(1.0/3)), self.D)]
        self.process_parameter(x, False, False)
        self.process_parameter(x, False, True)
        self.process_parameter(x, True, False)
        self.process_parameter(x, True, True)

