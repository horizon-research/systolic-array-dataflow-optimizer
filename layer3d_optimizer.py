#!/usr/bin/python2.7

# public library
import math
import numpy as np
from scipy.optimize import minimize

# my own library
from layer3d_base_method import *
from layer_optimizer import *

# threshold for bounds
# if the constraint result is negative but within this threshold,
# it is still consider a valid result.
Threshold = 10.0

class Layer3dOptimizer(Layer3dBaseMethod, LayerOptimizer):
    """docstring for Layer3dOptimizer"""
    def __init__(self, data, sys_info, combined=False):
        super(Layer3dOptimizer, self).__init__(data, sys_info)

    # variables for optimization
    # this two has been encodes as x[3] = {c_0, h_0, w_0, d_0};
    # c_0  # number of channels per batch;
    # h_0, w_0, d_0 # the dimensions of tile per batch;

    ###############################################################
    #                       general process                       #
    ###############################################################

    def init_guess(self):
        x0 = [min(self.A, self.Co), \
              min(math.floor(math.sqrt(self.A)), self.H), \
              min(math.floor(math.sqrt(self.A)), self.W), 1]
        return x0

    def variable_boundary(self):
        return ((min(self.A, self.Co), self.Co),
                (min(math.floor(math.sqrt(self.A)), self.H), self.H), \
                (min(math.floor(math.sqrt(self.A)), self.W), self.W), \
                (1, self.D))

    ###############################################################
    #                     general constraints                     #
    ###############################################################
    # the low bound of buffer size;
    # make sure the buffer utilization is always larger than 0
    def buffer_constraint1(self, x):
        return self.buffer_utilization(x)

    # the upper bound of the buffer size;
    # make sure the buffer utilization is
    # always smaller than buffer size;
    def buffer_constraint2(self, x):
        return self.buf_size - self.buffer_constraint1(x)

    ###############################################################
    #       row-major constraint solving obj and constraints      #
    ###############################################################

    # the minimization objective of row-major
    # this objective is a simplified expression of
    # [h_0*w_0*d_0*c_0+(S*h_0+2)(S*w_0+2)(S*d_0+2)*Ci]*(H*W*D*Co)/(h_0*w_0*d_0*c_0)
    # + [K^3*Ci+h_0*w_0*d_0*c_0]*Co/c_0
    def row_major_mem_obj(self, x):
      return (self.ofmap_tile(x) + self.ifmap_tile(x)) \
          * (self.total_ofmap_size()/self.ofmap_tile(x) - self.Co/x[0]) \
          + self.total_weight_size()/x[0] + x[1]*x[2]*x[3]*self.Co

    def row_major_comp_obj(self, x):
        return self.total_ofmap_size() / self.ofmap_tile(x)

    # make sure the load for row-major is always less than
    # load for channel-major, range : [0, +inf]
    def row_major_constraint(self, x):
        S_2 = (self.K_h+1)/2
        # simplified from K^3*Ci*c_0 > C*(S^3*h_0*w_0*d_0)
        return self.K_h*self.K_w*self.K_d*x[0] - \
            (self.S*x[1]+S_2)*(self.S*x[2]+S_2)*(self.S*x[3]+S_2);

    # make sure the process is always memory-bound;
    # which is the latency for memory access is always
    # greater than lantecy of compute;
    # (c_0*(h_0*w_0*d_0)+C*((S*h_0+2)*(S*w_0+2)*(S*d_0+2))/B
    # >= (K^3*Ci/A^2)*c_0*w_0*d_0*h_0
    # range : [0, +inf]
    def row_major_mem_bound_constraint(self, x):
      return (self.ofmap_tile(x) + self.ifmap_tile(x)) / self.B \
          - self.weight_tile(1)/(self.A*self.A)*self.ofmap_tile(x)

    # make sure the process is always compute-bound;
    # which is the latency for compute is always
    # greater than lantecy of memory access;
    # (c_0*(h_0*w_0*d_0)+Ci*((S*h_0+2)*(S*w_0+2)*(S*d_0+2))/B
    # <= (K^3*Ci/A^2)*c_0*w_0*h_0*d_0
    # range : [0, +inf]
    def row_major_comp_bound_constraint(self, x):
        return self.weight_tile(1) / (self.A*self.A)*self.ofmap_tile(x) \
            - (self.ofmap_tile(x) + self.ifmap_tile(x)) / self.B

    ###############################################################
    #     channel-major constraint solving obj and constraints    #
    ###############################################################

    # the minimization objective of channel-major
    # this is the simplified expression of
    # (K^3*Ci*c_0+h_0*w_0*d_0*c_0)*(H*W*D*Co)/(h_0*w_0*d_0*c_0)
    # + [(S*h_0+2)(S*w_0+2)(S*d_0+2)*Ci + h_0*w_0*d_0*c_0]*(H*W*D)/(h_0*w_0*d_0)
    def channel_major_mem_obj(self, x):
        S_2 = (self.K_h+1)/2
        return (self.total_weight_size)/(x[1]*x[2]*x[3]) + \
                (self.S*x[1]+S_2)*(self.S*x[2]+S_2)*(self.S*x[3]+S_2)/\
                (x[1]*x[2]*x[3])

    def channel_major_comp_obj(self, x):
        return self.total_ofmap_size()/(x[1]*x[2]*x[0]*x[3])

    # make sure the load for channel-major is always less than
    # load for row-major, range : [0, +inf]
    def channel_major_constraint(self, x):
        S_2 = (self.K_h+1)/2
        # simplified from K^3*Ci*c_0 <= Ci*((S*h_0+2)*(S*w_0+2))
        return (self.S*x[1]+S_2)*(self.S*x[2]+S_2)*(self.S*x[3]+S_2) \
            - self.K_h*self.K_w*self.K_d*x[0];

    # make sure the process is always memory-bound;
    # which is the latency for memory access is always
    # greater than lantecy of compute;
    # c_0*(h_0*w_0+K^3*C)/B >= (K^3*C/A^2)*c_0*(h_0*w_0)
    # range : [0, +inf]
    def channel_major_mem_bound_constraint(self, x):
        return (x[1]*x[2]*x[3]+self.weight_tile(1)) / self.B \
            - self.weight_tile(1)/(self.A*self.A)*x[1]*x[2]*x[3]


    # make sure the process is always memory-bound;
    # which is the latency for memory access is always
    # greater than lantecy of compute;
    # c_0*(h_0*w_0+K^3*C)/B >= (K^3*C/A^2)*c_0*(h_0*w_0*d_0)
    # range : [0, +inf]
    def channel_major_comp_bound_constraint(self, x):
        return (self.K_h*self.K_w*self.K_d*self.Co) \
            / (self.A*self.A)*x[1]*x[2]*x[3] \
            - (x[1]*x[2]+self.K_h*self.K_w*self.K_d*self.Co)/self.B


