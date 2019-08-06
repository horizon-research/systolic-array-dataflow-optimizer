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
    def __init__(self, data):
        super(DeconvExhaustiveSearcher, self).__init__(data)
        self.rets = []


    # compute buffer utilization
    def buffer_utilization(self, x, area):
        # buffer = ofmap + weights + ifmap
        total_buffer = self.Ci*(self.S*area[0]+2)*(self.S*area[1]+2)
        for i in range(len(x)):
            total_buffer += x[i]*area[0]*area[1]+ \
                    self.Ci*self.Subs[i][0]*self.Subs[i][1]*x[i]

        return total_buffer

    # (ofmap + ifmap)*total_batch + (ofmap+weights)
    def data_transfer(self, i, h_0, w_0, c_0):
      # calculate the total batch
      total_batch = self.Subs[i][0]*self.Subs[i][0]/(h_0*w_0)

      # ofmap and ifmap tile size
      ofmap_tile_size = h_0*w_0*c_0
      ifmap_tile_size = (self.S*h_0+2)*(self.S*w_0+2)*self.Ci
      # ofmap + ifmap transfer
      total_transfer = (ofmap_tile_size + ifmap_tile_size) * total_batch

      # weight tile size
      kernel_tile_size = self.Subs[i][0]*self.Subs[i][0]*self.Ci*c_0

      # add the rest
      total_transfer += (ofmap_tile_size + kernel_tile_size)

      return total_transfer

    def systolic_array_utilization(self, xi, area):
      area_size = area[0] * area[1]
      A = self.A
      total_usage = xi * area_size
      round_up_val = math.ceil(xi/self.A)*self.A \
          * math.ceil(area[0]*area[1]/self.A)*self.A
      return xi*area_size/round_up_val

    def compute_bound_cycle(self, i, util_rate, c_0):
      # total number of ops
      total_computation = (self.H*self.W*c_0)*\
          (self.Ci*self.Subs[i][0]*self.Subs[i][0])

      # systolic array calculation capacity
      comp_cap = (self.A*self.A) * util_rate

      return total_computation / comp_cap


    def process_parameter(self, x, area):
        area = list(map(lambda i: math.floor(i), area))
        w_0 = min(self.W/math.ceil(self.W/round(area[0])), self.W)
        h_0 = min(self.H/math.ceil(self.H/round(area[1])), self.H)

        total_cycle = 0

        for i in range(len(x)):
            if (round(x[i]) == 0):
                continue
            # make the tile size even for every batch
            c_0 = min(self.Co/math.ceil(self.Co/round(x[i])), self.Co)

            # check the result
            # compute the total number of elements needed to be updated
            # if it is row-major.
            total_transfer = self.data_transfer(i, h_0, w_0, c_0)

            # compute the utilization of systolic array
            util_sys_arr = self.systolic_array_utilization(x[i], area)

            # compute the cycle for compute-/memory-bound
            comp_bound_cycle = self.compute_bound_cycle(i, util_sys_arr, c_0)
            mem_bound_cycle = total_transfer/self.B

            # pick up the greater value as the actual cycle
            total_cycle += max(comp_bound_cycle, mem_bound_cycle)

        return (total_cycle, total_transfer)

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

        return x0

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
        while not all([sub == 0 for sub in remain_subkernels]):
            # set the initial guess;
            x0 = self.fill_bufw(remain_subkernels)

            # no need to optimize the buffer for ofmap, because it is
            # bounded ifmap.
            x1 = [math.sqrt(area), math.sqrt(area)]

            util_buf = self.buffer_utilization(x0, x1)/self.buffer_size

            # print(util_buf, x1, x0)
            if util_buf > 1.01:
                return

            (cycle, transfer) = self.process_parameter(x0, x1) \
                if str(x0 + x1) not in result_cache else result_cache[str(x0 + x1)]

            result_cache[str(x0 + x1)] = (cycle, transfer)

            if cycle == -1 or transfer == -1:
                return

            total_transfer += transfer
            total_cycle += cycle

            remain_subkernels = np.subtract(remain_subkernels, x0)

            round_result.append(x0)

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
        global SysArr, Bandwith, BufferSize

        self.res = []
        layer_info = self.data
        # set up the new layer information
        [self.W, self.H, self.Ci] = layer_info["ifmap"]
        self.Co = layer_info["out_channel"]
        [self.K_w, self.K_h] = layer_info["kernel"]
        self.S = layer_info["stride"]

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
            self.bufi_size = BufferSize*i/20.0
            for j in range(1, 20):
                self.bufw_size = BufferSize*j/20.0

                self.res = []
                # if sum of bufi and bufw is over the BufferSize
                # we should skip it.
                if (self.bufi_size + self.bufw_size) > BufferSize:
                    continue

                # set ofmap size
                self.bufo_size = BufferSize - self.bufi_size - self.bufw_size
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
