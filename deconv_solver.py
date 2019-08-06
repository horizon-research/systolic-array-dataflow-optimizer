#!/usr/bin/python2.7
# own module
# from memory_controller import *
# from systolic_array import *
# from onchip_buffer import *

# public library
import cv2 as cv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# info for systolic array
A = 16.0      # systolic array dimension

# info for weights
K = 3.0       # kernel size;
K_arr = []    # notated as an array of sub-kernel size;

# input layer dimension
H = 128.0     # height
W = 256.0     # width
C = 512.0     # channel

# memory bandwith 
B = 1.0

# buffer size
buffer_size = 2.0*1024.0*1024.0

# variables for optimization
# this two has been encodes as x[2]; = {c_0, h_0xw_0};
# c_0  # number of channels per batch;
# h_0xw_0 # size of tile per batch;

# calculate the latency for compute and memory;
# l_com = (K*K*c_0*h_0xw_0)/(R*R)
# # if row-major
# l_mem_r = B*(c_0*h_0xw_0 + C*h_0xw_0)
# # if channel-major
# l_mem_c = B*(c_0*h_0xw_0 + C*K*K*h_0xw_0)


#########################################################
#                 general process                       #
#########################################################

def process_parameter(x, row_major, comp_bound):

    res = [math.ceil(C/x[0]), C/math.ceil(C/x[0]), \
            math.ceil(W*H/x[1]), H*W/math.ceil(W*H/x[1])]

    print(math.ceil(C/x[0]), C/math.ceil(C/x[0]))
    print(math.ceil(W*H/x[1]), H*W/math.ceil(W*H/x[1]))

    x[0] = A*math.floor(x[0]/A)
    x[1] = A*math.floor(x[1]/A)

    print(math.ceil(C/x[0]), C/math.ceil(C/x[0]))
    print(math.ceil(W*H/x[1]), H*W/math.ceil(W*H/x[1]))

    if (row_major):
        total_transfer = (res[1]*res[3]+res[3]*C)*res[2]*res[0]\
                            +(res[1]*res[3]+K*K*C*res[3])*res[0]
    else:
        total_transfer = (res[1]*res[3]+K*K*C*res[1])*res[0]*res[2]\
                            +(res[1]*res[3]+res[3]*C)*res[2]

    if comp_bound:
        total_cycle = (res[0]/A*res[1]/A)*(C*K*K)*res[2]*res[3]
    else:
        total_cycle = total_transfer/B

    print("total_transfer", total_transfer)
    print("total_cycle", total_cycle)
    return [res, total_transfer]


# this function is to verifer if a given hardware
# configuration is able to realize in given hardware 
# constraints. 
# return the result and total 

# def verifier(x, row_major):

#########################################################
#               general constraints                     #
#########################################################
# the low bound of buffer size
def buffer_constraint1(x):
    return C*(K_arr[0]*K_arr[0]*x[0] + K_arr[1]*K_arr[1]*x[1]) \
                + (x[0]+x[1])*x[2] + C*x[2]

# the upper bound of the buffer size
def buffer_constraint2(x):
    return buffer_size - ((x[0]+x[1])*x[2] + C*x[2] + \
                C*(K_arr[0]*K_arr[0]*x[0] + K_arr[1]*K_arr[1]*x[1]))

# the low bound of buffer size
def buffer_constraint3(x):
    return C*K_arr[1]*K_arr[1]*x[3] + x[3]*x[4] + C*x[4]

# the upper bound of the buffer size
def buffer_constraint4(x):
    return buffer_size - (C*K_arr[1]*K_arr[1]*x[3] + x[3]*x[4] + C*x[4])

# make sure x[0] always greater than x[1]
def var_constraint1(x):
    return x[0]-x[1]

# make sure x[1] always greater than x[2]
def var_constraint2(x):
    return x[1]-x[2]

# make sure x[2] always greater than x[3]
def var_constraint3(x):
    return x[2]-x[3]

#########################################################
#   row-major constraint solving obj and constraints    #
#########################################################
# Attention to following notes:
# variables for optimization has always been 
# encoded as x[n]; = {c_0, ..., c_n-1, h_0xw_0};

# the minimization objective of row-major compute-bound
# since we don't consider boundary effects on total latency
# therefore, we try to make total round of computing be less.
def row_major_obj(x):
    return (H*W/x[0] + K_arr[0]*K_arr[0]/x[2]) + \
           (H*W/x[0] + K_arr[1]*K_arr[1]/x[2])*(x[1]/x[0]) + \
           (H*W/x[3] + K_arr[1]*K_arr[1]/x[4])*(1-x[1]/x[0])

# make sure the load for row-major is always less than 
# load for channel-major, range : [0, +inf]
def row_major_constraint1(x):
    # simplified from sum{K[i]*K[i]*C*x[i]} - C*x[2]
    return K_arr[0]*K_arr[0]*x[0]\
                +K_arr[1]*K_arr[1]*x[1]-x[2];

# make sure the load for row-major is always less than 
# load for channel-major, range : [0, +inf]
def row_major_constraint2(x):
    # simplified from K*K*C*x[3] - C*x[4]
    return K_arr[1]*K_arr[1]*x[3]-x[4]

# make sure the process is always memory-bound;
# range : [0, +inf]
def row_major_mem_bound_constraint1(x):
    return (C+x[0]+x[1])/B - \
            (K_arr[0]*K_arr[0]*x[0]+K_arr[1]*K_arr[1]*x[1])*C/(A*A)

# make sure the process is always memory-bound;
# range : [0, +inf]
def row_major_mem_bound_constraint2(x):
    return (C+x[3])/B - K_arr[1]*K_arr[1]*C/(A*A)*x[3]
 
# make sure the process is always compute-bound;
# range : [0, +inf]
def row_major_comp_bound_constraint1(x):
    return (K_arr[0]*K_arr[0]*x[0]+K_arr[1]*K_arr[1]*x[1])*C/(A*A)\
                - (C+x[0]+x[1])/B

# make sure the process is always compute-bound;
# range : [0, +inf]
def row_major_comp_bound_constraint2(x):
    return K_arr[1]*K_arr[1]*C/(A*A)*x[3] - (x[3]+C)/B

# The constraints related to the two sub-kernel optimization routine;
def row_major_dual_constraints():
    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': row_major_constraint1}
    # for row_major_comp_bound_constraint1
    con2 = {'type': 'ineq', 'fun': row_major_comp_bound_constraint1}
    # for the buffer_constraint
    con3 = {'type': 'ineq', 'fun': buffer_constraint1}
    con4 = {'type': 'ineq', 'fun': buffer_constraint2}
    # add additional variable constraints
    con5 = {'type': 'ineq', 'fun': var_constraint1}
    return [con1, con2, con3, con4, con5]

# The constraints related to the left sub-kernel optimization routine;
def row_major_rest_constraints():
    # for row_major_constraint2
    con6 = {'type': 'ineq', 'fun': row_major_constraint2}
    # for row_major_comp_bound_constraint2
    con7 = {'type': 'ineq', 'fun': row_major_comp_bound_constraint2}
    # for the buffer_constraint
    con8 = {'type': 'ineq', 'fun': buffer_constraint3}
    con9 = {'type': 'ineq', 'fun': buffer_constraint4}
    return [con6, con7, con8, con9]

# major routine for optimizing compute-bound row-major sequence;
def opti_comp_row_major():
    # set the initial guess;
    x0 = [A, A, A, A, A]

    # Dual sub-kernel constraint optimization;
    [con1, con2, con3, con4, con5] = row_major_dual_constraints();

    ## start to optimize the rest of the smaller sub-kernel ##
    [con6, con7, con8, con9] = 

    # summery all the bounds and constraints
    bnds = ((A, C), (A, C), (A, H*W), (A, C), (A, H*W))
    cons = ([con1, con2, con3, con4, con5, con6, con7, con8, con9])
    
    # call the external solver to solve the solution
    solution = minimize(row_major_obj, x0, method='SLSQP',\
                    bounds=bnds, constraints=cons)

    print("row major", solution.x, row_major_obj(solution.x))
    print(row_major_constraint(solution.x))
    print("buffer size", buffer_constraint1(solution.x))
    print(buffer_constraint2(solution.x))
    print(row_major_comp_bound_constraint(solution.x))

    process_parameter(solution.x, True, True)


def opti_comp_row_major(C_arr=None, K_arr=None):
    x0 = []
    # check if optimizing regular Conv.
    if C_arr is None:
        # set the initial guess;
        x0 = [A, A]
    else:
        # set the initial guess;
        x0 = [A]*(len(C_arr)+1)

    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': row_major_constraint}
    # for mem_bound_constraint
    con2 = {'type': 'ineq', 'fun': row_major_comp_bound_constraint}
    # for the buffer_constraint
    con3 = {'type': 'ineq', 'fun': buffer_constraint1}
    con4 = {'type': 'ineq', 'fun': buffer_constraint2}

    # initial bounds and conditions constraints array.
    bnds = []
    cons = []

    # summery all the bounds and constraints
    if len(x) == 2:
        bnds = ((A, C), (A, H*W))
        cons = ([con1, con2, con3, con4])
    elif len(x) == 3:
        # add additional variable constraints
        con5 = {'type': 'ineq', 'fun': var_constraint1}
        # summery all the bounds and constraints
        bnds = ((A, C), (A, C), (A, H*W))
        cons = ([con1, con2, con3, con4, con5])
    elif len(x) == 4:
        # add additional variable constraints
        con5 = {'type': 'ineq', 'fun': var_constraint1}
        con6 = {'type': 'ineq', 'fun': var_constraint2}
        # summery all the bounds and constraints
        bnds = ((A, C), (A, C), (A, C), (A, H*W))
        cons = ([con1, con2, con3, con4, con5, con6])
    else:
        # add additional variable constraints
        con5 = {'type': 'ineq', 'fun': var_constraint1}
        con6 = {'type': 'ineq', 'fun': var_constraint2}
        con7 = {'type': 'ineq', 'fun': var_constraint3}
        # summery all the bounds and constraints
        bnds = ((A, C), (A, C), (A, C), (A, C), (A, H*W))
        cons = ([con1, con2, con3, con4, con5, con6, con7])
    
    # call the external solver to solve the solution
    solution = minimize(row_major_obj, x0, method='SLSQP',\
                    bounds=bnds, constraints=cons)

    print("row major", solution.x, row_major_obj(solution.x))
    print(row_major_constraint(solution.x))
    print("buffer size", buffer_constraint1(solution.x))
    print(buffer_constraint2(solution.x))
    print(row_major_comp_bound_constraint(solution.x))

    process_parameter(solution.x, True, True)

def opti_mem_row_major(C_arr=None, K_arr=None):
    x0 = []
    # check if optimizing regular Conv.
    if C_arr is None:
        # set the initial guess;
        x0 = [A,A]
    else:
        # set the initial guess;
        x0 = [A]*(len(C_arr)+1)

    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': row_major_constraint}
    # for mem_bound_constraint
    con2 = {'type': 'ineq', 'fun': row_major_mem_bound_constraint}
    # for the buffer_constraint
    con3 = {'type': 'ineq', 'fun': buffer_constraint1}
    con4 = {'type': 'ineq', 'fun': buffer_constraint2}

    # initial bounds and conditions constraints array.
    bnds = None
    cons = None

    # summery all the bounds and constraints
    if len(x) == 2:
        bnds = ((A, C), (A, H*W))
        cons = ([con1, con2, con3, con4])
    elif len(x) == 3:
        # add additional variable constraints
        con5 = {'type': 'ineq', 'fun': var_constraint1}
        # summery all the bounds and constraints
        bnds = ((A, C), (A, C), (A, H*W))
        cons = ([con1, con2, con3, con4, con5])
    elif len(x) == 4:
        # add additional variable constraints
        con5 = {'type': 'ineq', 'fun': var_constraint1}
        con6 = {'type': 'ineq', 'fun': var_constraint2}
        # summery all the bounds and constraints
        bnds = ((A, C), (A, C), (A, C), (A, H*W))
        cons = ([con1, con2, con3, con4, con5, con6])
    else:
        # add additional variable constraints
        con5 = {'type': 'ineq', 'fun': var_constraint1}
        con6 = {'type': 'ineq', 'fun': var_constraint2}
        con7 = {'type': 'ineq', 'fun': var_constraint3}
        # summery all the bounds and constraints
        bnds = ((A, C), (A, C), (A, C), (A, C), (A, H*W))
        cons = ([con1, con2, con3, con4, con5, con6, con7])

    # call the external solver to solve the solution
    solution = minimize(row_major_obj, x0, method='SLSQP',\
                    bounds=bnds, constraints=cons)

    print("row major", solution.x, row_major_obj(solution.x))
    print(row_major_constraint(solution.x))
    print("buffer size", buffer_constraint1(solution.x))
    print(buffer_constraint2(solution.x))
    print(row_major_mem_bound_constraint(solution.x))

        process_parameter(solution.x, True, False)

########################################################
# channel-major constraint solving obj and constraints #
########################################################

# the minimization objective of channel-major compute-bound
# since we don't consider boundary effects on total latency
# therefore, we try to make total round of computing be less.
def channel_major_obj(x):
    # simplified from H*W*C/x[0] + K*K*C*C*W*H/x[1]
    return  1/x[0] + K*K*C/x[-1]

# make sure the load for channel-major is always less than 
# load for row-major, range : [0, +inf]
def channel_major_constraint(x):
    # simplified from C*x[1] - K*K*C*x[0]
    return x[1]-K*K*x[0];

# make sure the process is always memory-bound;
# range : [0, +inf]
def channel_major_mem_bound_constraint(x):
    # simplified from (x[0]*x[1] + K^2*C*x[0])/B - (K^2*C*x[1])/A^2
    return (x[1]+K*K*C)/B-K*K*C*x[1]/(A*A)

# make sure the process is always compute-bound;
# range : [0, +inf]
def row_major_comp_bound_constraint(x):
    # simplified from (K^2*C*x[1])/A^2 - (x[0]*x[1] + K^2*C*x[0])/B
    return K*K*C*x[1]/(A*A)-(x[1]+K*K*C)/B


def opti_mem_channel_major(C_arr=None, K_arr=None):
    # check if optimizing regular Conv.
    if C_arr is None:
        # set the initial guess;
        x0 = [A,A]
        # for row_major_constraint1
        con1 = {'type': 'ineq', 'fun': channel_major_constraint}
        # for mem_bound_constraint
        con2 = {'type': 'ineq', 'fun': channel_major_mem_bound_constraint}
        # for the buffer_constraint
        con3 = {'type': 'ineq', 'fun': buffer_constraint1}
        con4 = {'type': 'ineq', 'fun': buffer_constraint2}

        # summery all the bounds and constraints
        bnds = ((A, C), (A, H*W))
        cons= ([con1, con2, con3, con4])

        # call the external solver to solve the solution
        solution = minimize(channel_major_obj,x0,method='SLSQP',\
                        bounds=bnds,constraints=cons)

        print("channel major",solution.x, channel_major_obj(solution.x))
        print(channel_major_constraint(solution.x))
        print("buffer size", buffer_constraint1(solution.x))
        print(buffer_constraint2(solution.x))
        print(channel_major_mem_bound_constraint(solution.x))

        process_parameter(solution.x, False, False)


def opti_comp_channel_major(C_arr=None, K_arr=None):
    if C_arr is None:
        # set the initial guess;
        x0 = [A,A]
        # for row_major_constraint1
        con1 = {'type': 'ineq', 'fun': channel_major_constraint}
        # for mem_bound_constraint
        con2 = {'type': 'ineq', 'fun': channel_major_comp_bound_constraint}
        # for the buffer_constraint
        con3 = {'type': 'ineq', 'fun': buffer_constraint1}
        con4 = {'type': 'ineq', 'fun': buffer_constraint2}

        # summery all the bounds and constraints
        bnds = ((A, C), (A, H*W))
        cons= ([con1, con2, con3, con4])

        # call the external solver to solve the solution
        solution = minimize(channel_major_obj,x0,method='SLSQP',\
                        bounds=bnds,constraints=cons)

        print("channel major",solution.x, channel_major_obj(solution.x))
        print(channel_major_constraint(solution.x))
        print("buffer size", buffer_constraint1(solution.x))
        print(buffer_constraint2(solution.x))
        print(comp_bound_constraint(solution.x))

        process_parameter(solution.x, False, True)
    else:
        return 

def opti_comp_channel_major(C_arr=None, K_arr=None):
    # set the initial guess;
    x0 = [A, A, A]
    # for row_major_constraint1
    con1 = {'type': 'ineq', 'fun': channel_major_constraint}
    # for mem_bound_constraint
    con2 = {'type': 'ineq', 'fun': channel_major_comp_bound_constraint}
    # for the buffer_constraint
    con3 = {'type': 'ineq', 'fun': buffer_constraint1}
    con4 = {'type': 'ineq', 'fun': buffer_constraint2}

    # summery all the bounds and constraints
    bnds = ((A, C), (A, H*W))
    cons= ([con1, con2, con3, con4])

    # call the external solver to solve the solution
    solution = minimize(channel_major_obj,x0,method='SLSQP',\
                    bounds=bnds,constraints=cons)

    print("channel major",solution.x, channel_major_obj(solution.x))
    print(channel_major_constraint(solution.x))
    print("buffer size", buffer_constraint1(solution.x))
    print(buffer_constraint2(solution.x))
    print(comp_bound_constraint(solution.x))

    process_parameter(solution.x, False, True)


def opti_mem(C_arr=None, K_arr=None):
    print("==================================================================")
    print("========================  Memory Bound  ==========================")
    # optimization for row-major;
    opti_mem_row_major(C_arr, K_arr);
    # optimization for channel-major;
    opti_mem_channel_major(C_arr, K_arr);
    print("==================================================================\n")

def opti_comp(C_arr=None, K_arr=None):
    print("==================================================================")
    print("========================  Compute Bound  =========================")
    # optimization for row-major;
    opti_comp_row_major(C_arr, K_arr);
    # optimization for channel-major;
    opti_comp_channel_major(C_arr, K_arr);
    print("==================================================================\n")


#########################################################
#                  general routines                     #
#########################################################


def opti_deconv(K_arr):
    # initial an array of counter with length of K_arr
    C_arr = [C]*len(K_arr)
    res = []

    while (len(C_arr) != 0):
        # optimize on both cases;
        opti_mem(C_arr, K_arr)
        opti_comp(C_arr, K_arr)

    return


def optimizeLayer(height, width, channel, w_number):
    opti_mem() 
    return 
    if K == 3:
        K_arr = [(2,2),(2,1),(1,2),(1,1)]
        # enter optimize deconv routine;
        opti_deconv(K_arr)
        # done!!
    else:
        if K % 2 == 0:
            sub_K = K/2     # optimize regular Conv layer
            # if it is possible to be memory-bound only;
            if (sub_K*sub_K*C)/(A*A) < B or \
                B/((sub_K*sub_K*C)/(A*A) - B) > 1:
                opti_mem()  # only possible to be memory-bound;
            else:
                # both cases are possible;
                opti_mem()
                opti_comp()
            # done with optimization with same size sub-kernel;
        else:
            # first split the kernel into two different sub-kernels;
            K_arr = [((K+1)/2, (K+1)/2), ((K-1)/2, (K-1)/2)]
            # enter optimize deconv routine;
            opti_deconv(K_arr)
            # done with optimize deconv with different size sub-kernels;


if __name__== '__main__':
    optimizeLayer(H, W, C, C)


