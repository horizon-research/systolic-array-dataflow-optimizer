#!/usr/bin/python2.7

# public library
import math
import numpy as np

# info for systolic array
SysArr = 16.0      # systolic array dimension

# memory bandwith number of bytes can be transferred.
Bandwith = 16.0/4

# on-chip buffer size
BufferSize = 1.0*1024.0*1024.0


def setup_hardware(config):
    global SysArr, Bandwith, BufferSize
    SysArr = config[0]
    Bandwith = config[1]/(config[3]/8)
    BufferSize = config[2]


###############################################################
#                       general process                       #
###############################################################

class LayerBaseMethod(object):
    """docstring for LayerBaseMethod"""
    # info for systolic array
    A = None      # systolic array dimension

    # memory bandwith number of bytes can be transferred.
    B = None

    # on-chip buffer size
    buffer_size = None
    # info for weights
    K_w = None       # kernel width
    K_h = None       # kernel height
    S = None         # stride size

    # input layer dimension
    H = None        # height of ofmap
    W = None        # width of ifmap
    Ci = None      # channels for weights
    Co = None      # channels for ofmap

    # on-chip buffer size
    bufi_size = None
    bufo_size = None
    bufw_size = None


    # array to store the result from the four different results
    res = []

    """docstring for LayerExhaustiveSearcher"""
    def __init__(self, data):
        global SysArr, Bandwith, BufferSize
        self.data = data
        self.A = SysArr
        self.B = Bandwith
        self.buffer_size = BufferSize
        self.res = []


    # compute buffer utilization
    def buffer_utilization(self, x):
        # buffer = ofmap + weights + ifmap
        return x[0]*x[1]*x[2]+self.Ci*self.K_h*self.K_w*x[0]+ \
        self.Ci*(self.S*x[1]+2)*(self.S*x[2]+2)

    def process_parameter(self, x, row_major, comp_bound):

        x = list(map(lambda i: math.floor(i), x))
        bound = "C"
        # make the tile size even for every batch
        c_0 = min(self.Co/math.ceil(self.Co/round(x[0])), self.Co)
        w_0 = min(self.W/math.ceil(self.W/round(x[1])), self.W)
        h_0 =min(self.H/math.ceil(self.H/round(x[2])), self.H)
        # check the result
        # print(c_0, w_0, h_0, self.Co/c_0, self.W/w_0, self.H/h_0)
        # compute the total number of elements needed to be updated 
        # if it is row-major.
        if row_major:
            # (ofmap + ifmap)*total_batch + (ofmap+weights)*Co/c_0
            total_transfer = (h_0*w_0*c_0+(self.S*h_0+2)*(self.S*w_0+2)*self.Ci) \
                                *self.H*self.W*self.Co/(h_0*w_0*c_0) \
                                +(h_0*w_0*c_0+self.K_h*self.K_w*self.Ci*c_0)*self.Co/c_0
        # compute the total number of elements needed to be updated 
        # if it is channel-major.
        else:
            # (ofmap + weights)*total_batch + (ofmap+ifmap)*(H*W)/(h_0*w_0)
            total_transfer = (h_0*w_0*c_0+self.K_h*self.K_w*self.Ci*c_0) \
                                *self.H*self.W*self.Co/(h_0*w_0*c_0) \
                                +(h_0*w_0*c_0+(self.S*h_0+2)*(self.S*w_0+2)*self.Ci) \
                                *self.H*self.W/(h_0*w_0)

        # compute the utilization of systolic array
        util_sys_arr = x[0]/(math.ceil(x[0]/self.A)*self.A) \
                            *x[1]*x[2]/(math.ceil(x[1]*x[2]/self.A)*self.A)

        # compute the utilization of systolic array
        util_buf = self.buffer_utilization([c_0, w_0, h_0])/self.buffer_size

        if util_buf > 1.01:
            return
        # calculate the amount of cycles of computing all elements.
        if comp_bound:
            bound = "C"
            total_cycle = (self.H*self.W*self.Co)*(self.Ci*self.K_h*self.K_w)\
                            /(self.A*self.A)/util_sys_arr 
        else:
            bound = "M"
            total_cycle = total_transfer/self.B

        # print(x[0],(math.ceil(x[0]/A)*A), x[1]*x[2], (math.ceil(x[1]*x[2]/A)*A))
        ret = {
            "total_transfer": round(total_transfer),
            "total_cycle": round(total_cycle), 
            "systolic_array_utilization": util_sys_arr,
            "buffer_utilization": util_buf,
            "c_0, w_0, h_0": [c_0, w_0, h_0],
            "Tile size" : [self.Co/c_0, self.W/w_0, self.H/h_0],
            "Bound" : bound
        }
        self.res.append(ret)
        return
