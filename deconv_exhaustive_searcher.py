#!/usr/bin/python2.7

# public library
import math
import numpy as np

# my own module
from layer_base_method import *

###############################################################
#                       general process                       #
###############################################################
class DeconvExhaustiveSearcher(LayerBaseMethod):

    # array to store the result from the four different results
    rets = []

    """docstring for LayerExhaustiveSearcher"""
    def __init__(self, data, sys_info):
        super(DeconvExhaustiveSearcher, self).__init__(data, sys_info)
        self.rets = []


    # compute buffer utilization
    def buffer_utilization(self, x, area):
        # buffer = ofmap + weights + ifmap
        total_buffer = self.Ci*(self.S*area[0]+2)*(self.S*area[1]+2)
        for i in range(len(x)):
            total_buffer += x[i]*area[0]*area[1]+ \
                    self.Ci*self.Subs[i][0]*self.Subs[i][1]*x[i]

        return total_buffer

    def compute_bound_cycle(self, i, util_rate, c_0):
        # total number of ops
        total_computation = (self.H*self.W*c_0)*\
            (self.Ci*self.Subs[i][0]*self.Subs[i][1])

        # systolic array calculation capacity
        comp_cap = (self.A*self.A) * util_rate

        return total_computation / comp_cap


    def process_parameter(self, x, area):
        area = list(map(lambda i: math.floor(i), area))
        w_0 = min(self.W/math.ceil(self.W/round(area[0])), self.W)
        h_0 = min(self.H/math.ceil(self.H/round(area[1])), self.H)

        total_cycle = 0

        # calculate the total data transfer
        ifmap_tile_size = (self.S*h_0+2)*(self.S*w_0+2)*self.Ci

        # calculate the total batch
        total_batch = self.H*self.W/(h_0*w_0)

        # ifmap transfer
        total_transfer = ifmap_tile_size * total_batch

        util_sys_arr = 0
        util_cnt = 0

        for i in range(len(x)):
            if (round(x[i]) == 0):
                continue

            # compute the total number of elements needed to be updated in row-major.
            # ofmap and ifmap tile size
            ofmap_tile_size = h_0*w_0*x[i]

            # weight tile size
            kernel_tile_size = self.Subs[i][0]*self.Subs[i][0]*self.Ci*x[i]
            total_transfer += kernel_tile_size + ofmap_tile_size

            # compute the utilization of systolic array
            util_sys_arr += self.systolic_array_utilization(x[i], area)
            util_cnt += 1

            # compute the cycle for compute-/memory-bound
            comp_bound_cycle = self.compute_bound_cycle(i, util_sys_arr, x[i])
            mem_bound_cycle = total_transfer/self.B

            # pick up the greater value as the actual cycle
            total_cycle += max(comp_bound_cycle, mem_bound_cycle)

        if (util_cnt > 0):
            util_sys_arr = util_sys_arr/util_cnt

        return (total_cycle, total_transfer, util_sys_arr)

    def fill_bufw(self, remain_subkernels):
        x0 = [0]*len(self.data["sub-kernels"])
        sum_subs = 0
        for i in range(len(self.data["sub-kernels"])):
            sub_size = self.Subs[i][0]*self.Subs[i][1]
            # first, let's find the number of kernel we can put into buffer.
            while sum_subs < self.bufw_size \
                and x0[i] < remain_subkernels[i]:
                x0[i] = x0[i]+self.A
                sum_subs += self.A*sub_size*self.Ci

            if x0[i] > remain_subkernels[i]:
                x0[i] = remain_subkernels[i]

        return x0

    # heuristically decide the area dimenion. [W, H]
    def area_dimension(self, area):
        if area >= self.W * self.H:
          return [self.W, self.H]

        if math.sqrt(area) > self.H:
          tile_w = math.ceil(self.W/math.sqrt(area))
          return [self.W/tile_w, self.H]

        tile_w = math.ceil(self.W/math.sqrt(area))
        tile_h = math.ceil(self.H/math.sqrt(area))
        return [self.W/tile_w, self.H/tile_h]

    # the main optimization routine;
    def opti_buffer(self):
        # check if the initial configuration can hold the minimum requirements
        if ((self.A*self.K_h*self.K_w*self.Ci > self.bufw_size) or
            (self.S*self.S*self.A*self.Ci > self.bufi_size)):
            return

        total_cycle = 0
        total_transfer = 0
        remain_subkernels = [self.data["out_channel"]]*len(self.data["sub-kernels"])

        # set tile area;
        area = 0
        # next let's see how much ifmap can we fit into the buffer.
        while self.S*self.S*(area+self.A)*self.Ci < self.bufi_size:
            area = area+self.A

        round_result = []
        result_cache = {}

        # no need to optimize the buffer for ofmap, because it is
        # bounded ifmap.
        x1 = self.area_dimension(area)

        while not all([sub <= 0.0 for sub in remain_subkernels]):
    
            # set the initial guess;
            x0 = self.fill_bufw(remain_subkernels)

            util_buf = self.buffer_utilization(x0, x1)/self.buf_size

            # print(util_buf, x1, x0)
            if util_buf > 1.01:
                return

            (cycle, transfer, util_rate) = self.process_parameter(x0, x1) \
                if str(x0 + x1) not in result_cache else result_cache[str(x0 + x1)]

            result_cache[str(x0 + x1)] = (cycle, transfer, util_rate)

            if cycle == -1 or transfer == -1:
                return

            total_transfer += transfer
            total_cycle += cycle

            remain_subkernels = np.subtract(remain_subkernels, x0)

            round_result.append({"kernels" :x0,
                                 "tiles" : x1,
                                 "systolic array utilization" : util_rate})

        ret = {
            "total_transfer": round(total_transfer),
            "total_cycle": round(total_cycle),
            "partition" :  {
                "bufi_size" : round(self.bufi_size),
                "bufw_size" : round(self.bufw_size),
                "bufo_size" : round(self.bufo_size),
            },
            "round_result" : round_result,
        }
        self.rets.append(ret)

    # optimize one layer
    def optimize(self):
        self.init_setup()
  
        layer_info = self.data
        add_one = [(i+1)/2 for i in layer_info["kernel"]]
        sub_one = [i/2 for i in layer_info["kernel"]]
        self.data["sub-kernels"] = [
            [add_one[0], add_one[1]],
            [add_one[0], sub_one[1]],
            [sub_one[0], add_one[1]],
            [sub_one[0], sub_one[1]]]

        self.Subs = self.data["sub-kernels"]

        # print("##[LAYER]##", self.W, self.H, self.Ci, self.Co, self.K_w, self.K_h)

        for i in range(1, 20):
            self.bufi_size = self.buf_size*i/20.0
            for j in range(1, 20):
                self.bufw_size = self.buf_size*j/20.0

                self.res = []
                # if sum of bufi and bufw is over the self.buf_size
                # we should skip it.
                if (self.bufi_size + self.bufw_size) > self.buf_size:
                    continue

                # set ofmap size
                self.bufo_size = self.buf_size - self.bufi_size - self.bufw_size
                # both cases are possible;
                self.opti_buffer()


        ret  = dict(self.rets[0])

        for item in self.rets:
            if ret["total_cycle"] > item["total_cycle"]:
                ret = dict(item)
            if ret["total_cycle"] == item["total_cycle"] and \
                ret["total_transfer"] > item["total_transfer"]:
                ret = dict(item)

        return ret
