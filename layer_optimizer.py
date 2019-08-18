#!/usr/bin/python2.7

# public library
import math
import numpy as np
from scipy.optimize import minimize

# my own module
from layer_base_method import *
import layer_exhaustive_searcher

# threshold for bounds
# if the constraint result is negative but within this threshold,
# it is still consider a valid result.
Threshold = 500.0

class LayerOptimizer(LayerBaseMethod):
    """docstring for LayerOptimizer"""
    def __init__(self, data, sys_info, combined=False):
        super(LayerOptimizer, self).__init__(data, sys_info)
        self.combined = combined

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
        self.init_setup()

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
        x0 = self.init_guess()
        # for row_major_constraint1
        con1 = {'type': 'ineq', 'fun': self.row_major_constraint}
        # for mem_bound_constraint
        con2 = {'type': 'ineq', 'fun': self.row_major_mem_bound_constraint}
        # for the buffer_constraint
        con3 = {'type': 'ineq', 'fun': self.buffer_constraint1}
        con4 = {'type': 'ineq', 'fun': self.buffer_constraint2}

        # summery all the bounds and constraints
        bnds = self.variable_boundary()
        cons = ([con1, con2, con3, con4])

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
        x0 = self.init_guess()
        # for row_major_constraint1
        con1 = {'type': 'ineq', 'fun': self.row_major_constraint}
        # for mem_bound_constraint
        con2 = {'type': 'ineq', 'fun': self.row_major_comp_bound_constraint}
        # for the buffer_constraint
        con3 = {'type': 'ineq', 'fun': self.buffer_constraint1}
        con4 = {'type': 'ineq', 'fun': self.buffer_constraint2}

        # summery all the bounds and constraints
        bnds = self.variable_boundary()
        cons = ([con1, con2, con3, con4])

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
        x0 = self.init_guess()
        # for row_major_constraint1
        con1 = {'type': 'ineq', 'fun': self.channel_major_constraint}
        # for mem_bound_constraint
        con2 = {'type': 'ineq', 'fun': self.channel_major_mem_bound_constraint}
        # for the buffer_constraint
        con3 = {'type': 'ineq', 'fun': self.buffer_constraint1}
        con4 = {'type': 'ineq', 'fun': self.buffer_constraint2}

        # summery all the bounds and constraints
        bnds = self.variable_boundary()
        cons = ([con1, con2, con3, con4])

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
        x0 = self.init_guess()
        # for row_major_constraint1
        con1 = {'type': 'ineq', 'fun': self.channel_major_constraint}
        # for mem_bound_constraint
        con2 = {'type': 'ineq', 'fun': self.channel_major_comp_bound_constraint}
        # for the buffer_constraint
        con3 = {'type': 'ineq', 'fun': self.buffer_constraint1}
        con4 = {'type': 'ineq', 'fun': self.buffer_constraint2}

        # summery all the bounds and constraints
        bnds = self.variable_boundary()
        cons = ([con1, con2, con3, con4])

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
    #                     general computations                    #
    ###############################################################

    def ofmap_tile(self, x):
        return x[0]*x[1]*x[2]

    def weight_tile(self, num):
        return self.Ci*self.K_h*self.K_w*num

    def ifmap_tile(self, x):
        return self.Ci*(self.S*x[1]+2)*(self.S*x[2]+2)

    def total_ofmap_size(self):
        return self.H*self.W*self.Co

    def total_weight_size(self):
        return self.weight_tile(self.Co)

    ###############################################################
    #                     general constraints                     #
    ###############################################################
    # the low bound of buffer size;
    # make sure the buffer utilization is always larger than 0
    def buffer_constraint1(self, x):
        # buffer = ofmap + weights + ifmap
        return (self.ofmap_tile(x) +
                self.weight_tile(x[0]) +
                self.ifmap_tile(x))

    # the upper bound of the buffer size;
    # make sure the buffer utilization is
    # always smaller than buffer size;
    def buffer_constraint2(self, x):
        return (self.buf_size - self.buffer_constraint1(x))

    # set initial guess for constrained optimization
    def init_guess(self):
        # set the initial guess;
        x0 = [min(self.A, self.Co), \
              min(math.floor(math.sqrt(self.A)), self.H), \
              min(math.floor(math.sqrt(self.A)), self.W)]
        if self.combined:
          result = layer_static_method.\
              LayerStaticMethod(data, sys_info, [3.0, 3.0, 4.0]).optimize()
          x0 = result["c_0, w_0, h_0"]

        return x0

    # set constraints for the variables in the optimization
    def variable_boundary(self):
      return ((min(self.A, self.Co), self.Co),
              (min(math.floor(math.sqrt(self.A)), self.H), self.H),
              (min(math.floor(math.sqrt(self.A)), self.W), self.W))


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
      return (self.ofmap_tile(x) + self.ifmap_tile(x)) \
          * (self.total_ofmap_size()/self.ofmap_tile(x) - self.Co/x[0]) \
          + self.total_weight_size()/x[0] + x[1]*x[2]*self.Co

    def row_major_comp_obj(self, x):
      return self.total_ofmap_size() / self.ofmap_tile(x)

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
      return (self.ofmap_tile(x) + self.ifmap_tile(x)) / self.B \
          - self.weight_tile(1)/(self.A*self.A)*self.ofmap_tile(x)

    # make sure the process is always compute-bound;
    # which is the latency for compute is always
    # greater than lantecy of memory access;
    # (c_0*(h_0*w_0)+C*((S*h_0+2)*(S*w_0+2))/B <= (K^2*C/A^2)*c_0*w_0*h_0
    # range : [0, +inf]
    def row_major_comp_bound_constraint(self, x):
        return self.weight_tile(1)/(self.A*self.A)*self.ofmap_tile(x) \
            - (self.ofmap_tile(x) + self.ifmap_tile(x)) / self.B

    ###############################################################
    #     channel-major constraint solving obj and constraints    #
    ###############################################################

    # the minimization objective of channel-major
    # this is the simplified expression of
    # (K^2*Ci*c_0+h_0*w_0*c_0)*(H*W*Co)/(h_0*w_0*c_0)
    # + [(h_0+2)(w_0+2)*Ci + h_0*w_0*c_0]*(H*W)/(h_0*w_0)
    def channel_major_mem_obj(self, x):
        return (self.total_weight_size)/(x[1]*x[2]) + \
            2*(self.S*x[1]+self.S*x[2])*self.Co/(x[1]*x[2])+1/x[0]

    # the minimization functions is to moinimize the
    # channel major compute-bound objective
    def channel_major_comp_obj(self, x):
        return self.total_ofmap_size()/(x[1]*x[2]*x[0])

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
        return (x[1]*x[2] + self.weight_tile(1)) / self.B \
            - self.weight_tile(1)/(self.A*self.A)*x[1]*x[2]

    # make sure the process is always memory-bound;
    # which is the latency for memory access is always
    # greater than lantecy of compute;
    # c_0*(h_0*w_0+K^2*C)/B >= (K^2*C/A^2)*c_0*(h_0*w_0)
    # range : [0, +inf]
    def channel_major_comp_bound_constraint(self, x):
        return self.K_h*self.K_w*self.Co/(self.A*self.A)*x[1]*x[2] \
            - (x[1]*x[2]+self.K_h*self.K_w*self.Co)/self.B


