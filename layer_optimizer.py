#!/usr/bin/python2.7

# public library
import math
import numpy as np
from scipy.optimize import minimize

# my own module
from layer_base_method import *

# threshold for bounds
# if the constraint result is negative but within this threshold,
# it is still consider a valid result.
Threshold = 500.0

class LayerOptimizer(LayerBaseMethod):
    """docstring for LayerOptimizer"""
    def __init__(self, data):
        super(LayerOptimizer, self).__init__(data)

    # variables for optimization
    # this two has been encodes as x[3] = {c_0, h_0, w_0};
    # c_0  # number of channels per batch;
    # h_0xw_0 # size of tile per batch;

    # calculate the latency for compute and memory;
    # l_com = (K_h*K_w*c_0*h_0*w_0)/(R*R)
    # # if row-major
    # l_mem_r = (c_0*h_0*w_0 + C*(h_0+2)*(w_0+2))/B
    # # if channel-major
    # l_mem_c = (c_0*h_0*w_0 + C*K_h*K_w*c_0)/B

    ###############################################################
    #                       general process                       #
    ###############################################################

    def optimize(self):
        self.res = []
        layer_info = self.data
        # set up the new layer information
        [self.W, self.H, self.Ci] = layer_info["ifmap"]
        self.Co = layer_info["out_channel"]
        [self.K_w, self.K_h] = layer_info["kernel"]
        self.S = layer_info["stride"]

        # print("##[LAYER]##", self.W, self.H, self.Ci, self.Co, self.K_w, self.K_h)
        # both cases are possible;
        # opti_mem()
        self.opti_comp()

        if len(self.res) == 0:
            self.opti_mem()

        if len(self.res) == 0:
            return None

        ret  = dict(self.res[0])

        for item in self.res:
            if ret["total_cycle"] > item["total_cycle"]:
                ret = dict(item)
            if ret["total_cycle"] == item["total_cycle"] and \
                ret["total_transfer"] > item["total_transfer"]:
                ret = dict(item)

        return ret

    def opti_mem(self):
        # print("=========================  Memory Bound  ==========================")
        # optimization for row-major;
        self.opti_mem_row_major();
        # optimization for channel-major;
        self.opti_mem_channel_major();
        # print("\n")

    def opti_comp(self):
        # print("=========================  Compute Bound  =========================")
        # optimization for row-major;
        self.opti_comp_row_major();
        # optimization for channel-major;
        self.opti_comp_channel_major();
        # print("\n")

    # the main optimization of memory-bound and row-major case;
    def opti_mem_row_major(self):
        # set the initial guess;
        x0 = [min(self.A, self.Co), min(math.floor(math.sqrt(self.A)), self.H), \
                min(math.floor(math.sqrt(self.A)), self.W)]
        # for row_major_constraint1
        con1 = {'type': 'ineq', 'fun': self.row_major_constraint}
        # for mem_bound_constraint
        con2 = {'type': 'ineq', 'fun': self.row_major_mem_bound_constraint}
        # for the buffer_constraint
        con3 = {'type': 'ineq', 'fun': self.buffer_constraint1}
        con4 = {'type': 'ineq', 'fun': self.buffer_constraint2}

        # summery all the bounds and constraints
        bnds = ((min(self.A, self.Co),self.Co), (min(math.floor(math.sqrt(self.A)), self.H), self.H), \
                (min(math.floor(math.sqrt(self.A)), self.W), self.W))
        cons= ([con1, con2, con3, con4])

        # call the external solver to solve the solution
        solution = minimize(self.row_major_mem_obj, x0, method='SLSQP',\
                        bounds=bnds, constraints=cons)

        passed = True
        if np.any(np.isnan(solution.x)):
            passed = False
            # print("Solution with NaN, abort!")
        # check the validation
        if passed and self.row_major_constraint(solution.x) < -Threshold:
            passed = False
            # print("row major constraint", self.row_major_constraint(solution.x), "NOT PASSED.")
        if passed and self.buffer_constraint2(solution.x) < -Threshold:
            passed = False
            # print("buffer size", self.buffer_constraint1(solution.x), "is OVER limit!")
            # print("buffer constraint", buffer_constraint2(solution.x))
        if passed and self.row_major_mem_bound_constraint(solution.x) < -Threshold:
            passed = False
            # print("row-major memory-bound", self.row_major_mem_bound_constraint(solution.x), \
            #      " no longer bounded!")
        
        if passed:
            # print("Row-major memory-bound case PASSED!")
            self.process_parameter(solution.x, True, False)
        else:
            return None

    # the main optimization of compute-bound and row-major case;
    def opti_comp_row_major(self):
        # set the initial guess;
        x0 = [min(self.A, self.Co), min(math.floor(math.sqrt(self.A)),self.H), \
                min(math.floor(math.sqrt(self.A)), self.W)]
        # for row_major_constraint1
        con1 = {'type': 'ineq', 'fun': self.row_major_constraint}
        # for mem_bound_constraint
        con2 = {'type': 'ineq', 'fun': self.row_major_comp_bound_constraint}
        # for the buffer_constraint
        con3 = {'type': 'ineq', 'fun': self.buffer_constraint1}
        con4 = {'type': 'ineq', 'fun': self.buffer_constraint2}

        # summery all the bounds and constraints
        bnds = ((min(self.A, self.Co), self.Co), (min(math.floor(math.sqrt(self.A)), self.H), self.H), \
                (min(math.floor(math.sqrt(self.A)), self.W), self.W))
        cons= ([con1, con2, con3, con4])

        # call the external solver to solve the solution
        solution = minimize(self.row_major_comp_obj, x0, method='SLSQP',\
                        bounds=bnds, constraints=cons)

        passed = True
        if np.any(np.isnan(solution.x)):
            passed = False
            # print("Solution with NaN, abort!")
        # check the validation
        if passed and self.row_major_constraint(solution.x) < -Threshold:
            passed = False
            # print("row major constraint", self.row_major_constraint(solution.x), "NOT PASSED.")
        if passed and self.buffer_constraint2(solution.x) < -Threshold:
            passed = False
            # print("buffer size", self.buffer_constraint1(solution.x), "is OVER limit!")
        if passed and self.row_major_comp_bound_constraint(solution.x) < -Threshold:
            passed = False
            # print("Row-major compute-bound", self.row_major_comp_bound_constraint(solution.x), \
            #     " no longer bounded!")

        if passed:
            # print("Row-major compute-bound case PASSED!")
            self.process_parameter(solution.x, True, True)
        else:
            return None



    # the main optimization of memory-bound and channel-major case;
    def opti_mem_channel_major(self):
        # set the initial guess;
        x0 = [min(self.A, self.Co), min(math.floor(math.sqrt(self.A)), self.H), \
                min(math.floor(math.sqrt(self.A)), self.W)]
        # for row_major_constraint1
        con1 = {'type': 'ineq', 'fun': self.channel_major_constraint}
        # for mem_bound_constraint
        con2 = {'type': 'ineq', 'fun': self.channel_major_mem_bound_constraint}
        # for the buffer_constraint
        con3 = {'type': 'ineq', 'fun': self.buffer_constraint1}
        con4 = {'type': 'ineq', 'fun': self.buffer_constraint2}

        # summery all the bounds and constraints
        bnds = ((min(self.A, self.Co), self.Co), (min(math.floor(math.sqrt(self.A)), self.H),self.H), \
                (min(math.floor(math.sqrt(self.A)), self.W), self.W))
        cons= ([con1, con2, con3, con4])

        # call the external solver to solve the solution
        solution = minimize(self.channel_major_mem_obj, x0, method='SLSQP',\
                        bounds=bnds, constraints=cons)

        passed = True
        if np.any(np.isnan(solution.x)):
            passed = False
            # print("Solution with NaN, abort!")
        # check the validation
        if passed and self.channel_major_constraint(solution.x) < -Threshold:
            passed = False
            # print("channel major constraint", self.channel_major_constraint(solution.x), "NOT PASSED.")
        if passed and self.buffer_constraint2(solution.x) < -Threshold:
            passed = False
            # print("buffer size", self.buffer_constraint1(solution.x), "is OVER limit!")
        if passed and self.channel_major_mem_bound_constraint(solution.x) < -Threshold:
            passed = False
            # print("Channel-major memory-bound", self.channel_major_mem_bound_constraint(solution.x), \
            #     " no longer bounded!")

        if passed:
            # print("Channel-major memory-bound case PASSED!")
            self.process_parameter(solution.x, False, False)
        else:
            return None


    # the main optimization of compute-bound and channel-major case;
    def opti_comp_channel_major(self):
        # set the initial guess;
        x0 = [min(self.A, self.Co), min(math.floor(math.sqrt(self.A)), self.H), \
                min(math.floor(math.sqrt(self.A)), self.W)]
        # for row_major_constraint1
        con1 = {'type': 'ineq', 'fun': self.channel_major_constraint}
        # for mem_bound_constraint
        con2 = {'type': 'ineq', 'fun': self.channel_major_comp_bound_constraint}
        # for the buffer_constraint
        con3 = {'type': 'ineq', 'fun': self.buffer_constraint1}
        con4 = {'type': 'ineq', 'fun': self.buffer_constraint2}

        # summery all the bounds and constraints
        bnds = ((min(self.A, self.Co), self.Co), (min(math.floor(math.sqrt(self.A)), self.H), self.H), \
                (min(math.floor(math.sqrt(self.A)), self.W), self.W))
        cons= ([con1, con2, con3, con4])

        # call the external solver to solve the solution
        solution = minimize(self.channel_major_comp_obj, x0, method='SLSQP',\
                        bounds=bnds, constraints=cons)

        passed = True
        if np.any(np.isnan(solution.x)):
            passed = False
            # print("Solution with NaN, abort!")
        # check the validation
        if passed and self.channel_major_constraint(solution.x) < -Threshold:
            passed = False
            # print("channel major constraint", self.channel_major_constraint(solution.x), "NOT PASSED.")
        if passed and self.buffer_constraint2(solution.x) < -Threshold:
            passed = False
            # print("buffer size", self.buffer_constraint1(solution.x), "is OVER limit!")
        if passed and self.channel_major_comp_bound_constraint(solution.x) < -Threshold:
            passed = False
            # print("Channel-major compute-bound", self.channel_major_comp_bound_constraint(solution.x), \
            #     " no longer bounded!")

        if passed:
            # print("Channel-major compute-bound case PASSED!")
            self.process_parameter(solution.x, False, True)
        else:
            return None



    ###############################################################
    #                     general constraints                     #
    ###############################################################
    # the low bound of buffer size;
    # make sure the buffer utilization is always larger than 0
    def buffer_constraint1(self, x):
        # buffer = ofmap + weights + ifmap
        return x[0]*x[1]*x[2]+self.Ci*self.K_h*self.K_w*x[0]+self.Ci*(self.S*x[1]+2)*(self.S*x[2]+2)

    # the upper bound of the buffer size;
    # make sure the buffer utilization is
    # always smaller than buffer size;
    def buffer_constraint2(self, x):
        return self.buffer_size - (x[0]*x[1]*x[2]+self.Ci*self.K_h*self.K_w*x[0]+\
                self.Ci*(self.S*x[1]+2)*(self.S*x[2]+2))


    ###############################################################
    #       row-major constraint solving obj and constraints      #
    ###############################################################

    # the minimization objective of row-major
    # this objective is a simplified expression of 
    # [h_0*w_0*c_0+(h_0+2)(w_0+2)*Ci]*(H*W*Co)/(h_0*w_0*c_0)
    # + [K^2*Ci+h_0*w_0*c_0]*Co/c_0
    # this expression can be finally reduce to:
    #   (H*W*Co/c_0 + 2(h_0+w_0)Ci*H*W*Co/(h_0*w_0*c_0)+h_0*w_0*Co/c_0
    def row_major_mem_obj(self, x):
        return (x[0]*x[1]*x[2] + (self.S*x[1]+2)*(self.S*x[2]+2)*self.Ci)*\
                (self.H*self.W*self.Co/(x[0]*x[1]*x[2])-self.Co/x[0]) \
                + (self.K_h*self.K_w*self.Ci)*self.Co/x[0] + x[1]*x[2]*self.Co
        # return H*W*Co/x[0]*(1+2*S*(x[1]+x[2])*Ci/(x[1]*x[2])) + x[1]*x[2]/x[0]

    def row_major_comp_obj(self, x):
        return self.H*self.W*self.Co/(x[1]*x[2]*x[0])

    # make sure the load for row-major is always less than 
    # load for channel-major, range : [0, +inf]
    def row_major_constraint(self, x):
        # simplified from K^2*C*c_0 > C*(S^2*h_0*w_0)
        return self.K_h*self.K_w*x[0] - (self.S*x[1]+2)*(self.S*x[2]+2);

    # make sure the process is always memory-bound;
    # which is the latency for memory access is always 
    # greater than lantecy of compute;
    # (c_0*(h_0*w_0)+C*((S*h_0+2)*(S*w_0+2))/B >= (K^2*C/A^2)*c_0*w_0*h_0 
    # range : [0, +inf]
    def row_major_mem_bound_constraint(self, x):
        return (x[0]*x[1]*x[2] + self.Ci*(self.S*x[1]+2)*(self.S*x[2]+2))/self.B \
                    - self.K_h*self.K_w*self.Ci/(self.A*self.A)*x[0]*x[1]*x[2]


    # make sure the process is always compute-bound;
    # which is the latency for compute is always 
    # greater than lantecy of memory access;
    # (c_0*(h_0*w_0)+C*((S*h_0+2)*(S*w_0+2))/B <= (K^2*C/A^2)*c_0*w_0*h_0 
    # range : [0, +inf]
    def row_major_comp_bound_constraint(self, x):
        return self.K_h*self.K_w*self.Ci/(self.A*self.A)*x[0]*x[1]*x[2] \
                - (x[0]*x[1]*x[2] + self.Ci*(self.S*x[1]+2)*(self.S*x[2]+2))/self.B


    ###############################################################
    #     channel-major constraint solving obj and constraints    #
    ###############################################################

    # the minimization objective of channel-major
    # this is the simplified expression of 
    # (K^2*Ci*c_0+h_0*w_0*c_0)*(H*W*Co)/(h_0*w_0*c_0)
    # + [(h_0+2)(w_0+2)*Ci + h_0*w_0*c_0]*(H*W)/(h_0*w_0)
    def channel_major_mem_obj(self, x):
        return  (self.K_h*self.K_w*self.Ci*self.Co)/(x[1]*x[2])+\
                2*(self.S*x[1]+self.S*x[2])*self.Co/(x[1]*x[2])+1/x[0]

    # the minimization functions is to moinimize the 
    #
    def channel_major_comp_obj(self, x):
        return self.H*self.W*self.Co/(x[1]*x[2]*x[0])

    # make sure the load for channel-major is always less than 
    # load for row-major, range : [0, +inf]
    def channel_major_constraint(self, x):
        # simplified from K^2*C*c_0 <= C*((S*h_0+2)*(S*w_0+2))
        return (self.S*x[1]+2)*(self.S*x[2]+2) - self.K_h*self.K_w*x[0];

    # make sure the process is always memory-bound;
    # which is the latency for memory access is always 
    # greater than lantecy of compute;
    # c_0*(h_0*w_0+K^2*C)/B >= (K^2*C/A^2)*c_0*(h_0*w_0)
    # range : [0, +inf]
    def channel_major_mem_bound_constraint(self, x):
        return (x[1]*x[2]+self.K_h*self.K_w*self.Ci)/self.B \
        - self.K_h*self.K_w*self.Ci/(self.A*self.A)*x[1]*x[2]

    # make sure the process is always memory-bound;
    # which is the latency for memory access is always 
    # greater than lantecy of compute;
    # c_0*(h_0*w_0+K^2*C)/B >= (K^2*C/A^2)*c_0*(h_0*w_0) 
    # range : [0, +inf]
    def channel_major_comp_bound_constraint(self, x):
        return self.K_h*self.K_w*self.Co/(self.A*self.A)*x[1]*x[2] \
        - (x[1]*x[2]+self.K_h*self.K_w*self.Co)/self.B




