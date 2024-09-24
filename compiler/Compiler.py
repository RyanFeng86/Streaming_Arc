import platform
import os
import numpy
import math
import re
import HW_Resources

if platform.system().lower() == "windows":
    PATH = ".\\Network.txt"
else:
    PATH = "./Network.txt"


def attach_multiple(tmp, a, b, c):
    tmp.append(int(a))
    tmp.append(int(b))
    tmp.append(int(c))


def find_seq_max_index(input_list, seq):
    tmp = input_list[:]
    i = 1
    while i <= seq:
        if i == seq:
            return tmp.index(max(tmp))
        else:
            index = tmp.index(max(tmp))
            tmp[index] = -1
        i = i + 1


class Compiler:
    __ifm_data_width__ = 0  # byte
    __filter_data_width__ = 0  # byte
    __weight_list__ = []
    __ifm_list__ = []
    __ifm_with_padding_list__ = []
    __ofm_list__ = []
    __ofm_with_pooling_list__ = []
    __mul_opt_list__ = []

    __total_mul_opt__ = 0
    __dsp_allocation_list__ = []
    __pe_list__ = []
    __simd_list__ = []
    __c_r_list__ = []

    __pre_store_column = []
    __pre_store_filter = []
    __weight_stream_depth = []
    __IFM_depth = []
    __IFM_after_gen_depth = []
    __conv_result_depth = []
    __pooling_result_depth = []

    __used_memory = []
    __used_bandwidth = []

    # __balance_model__ = False

    def __init__(self, ifm, weight):
        self.__ifm_data_width__ = ifm
        self.__filter_data_width__ = weight
        # self.__balance_model__ = model

    def read_network(self, path=PATH):
        network = open(path, "r")
        if network:
            print("File exists")
        else:
            print("File doesn't exist")

        line = network.readline()
        pattern = re.compile(r'\d+')
        regEx = pattern.findall(line)
        regEx = [int(data) for data in regEx]
        self.__ifm_list__.append(regEx)
        for line in network:
            tmp = []
            if "FL" in line:
                tmp.append("fl_en")
            else:
                tmp.append("fl_dis")

            if "PL" in line:
                tmp.append("pl_en")
            else:
                tmp.append("pl_dis")

            if "PD" in line:
                tmp.append("pd_en")
            else:
                tmp.append("pd_dis")
            pattern = re.compile(r'\d+')
            regEx = pattern.findall(line)
            regEx = [int(data) for data in regEx]
            tmp.extend(regEx)
            self.__weight_list__.append(tmp)
        print("ifm_list:", self.__ifm_list__)
        print("weight_list", self.__weight_list__)

    def layer_info_calculation(self):
        i = 0
        while i < len(self.__weight_list__):
            tmp = []
            w = self.__ifm_list__[i][0]
            h = self.__ifm_list__[i][1]
            c = self.__ifm_list__[i][2]
            if self.__weight_list__[i][2] == "pd_en":
                w = w + 2
                h = h + 2
            attach_multiple(tmp, w, h, c)
            self.__ifm_with_padding_list__.append(tmp)
            w = (w - self.__weight_list__[i][3] + 1) / self.__weight_list__[i][7]
            h = (h - self.__weight_list__[i][4] + 1) / self.__weight_list__[i][7]
            c = self.__weight_list__[i][6]
            operation = w * h * c * self.__weight_list__[i][3] * self.__weight_list__[i][4] * self.__weight_list__[i][5]
            self.__mul_opt_list__.append(operation)
            self.__total_mul_opt__ = self.__total_mul_opt__ + operation
            tmp = []
            attach_multiple(tmp, w, h, c)
            self.__ofm_list__.append(tmp)
            if self.__weight_list__[i][1] == "pl_en":
                w = int(w / 2)
                h = int(h / 2)
            tmp = []
            attach_multiple(tmp, w, h, c)
            self.__ofm_with_pooling_list__.append(tmp)
            tmp = []
            attach_multiple(tmp, w, h, c)
            self.__ifm_list__.append(tmp)
            i = i + 1
        print("ifm_list:", self.__ifm_list__)
        print("ifm_with_padding_list:", self.__ifm_with_padding_list__)
        print("ofm_list:", self.__ofm_list__)
        print("ofm_with_pooling_list:", self.__ofm_with_pooling_list__)
        print("mul_opt_list:", self.__mul_opt_list__)

    def pe_simd_dsp_allocation(self):
        i = 0
        simd_pre = []
        simd_index = []
        while i < len(self.__weight_list__):
            simd_index.append(0)
            tmp = [1]
            j = 2
            while j <= self.__weight_list__[i][5]:
                if self.__weight_list__[i][5] % j == 0:
                    tmp.append(j)
                if j < 32:
                    j = j + 1
                else:
                    j = j + 32
            simd_pre.append(tmp)
            i = i + 1
        print(simd_pre)
        # initial pe,simd,dsp for each layer
        i = 0
        while i < len(self.__weight_list__):
            self.__dsp_allocation_list__.append(1)
            self.__simd_list__.append(simd_pre[i][0])
            self.__pe_list__.append(1)
            i = i + 1
        # initial c_r list
        self.__c_r_list__ = [a / b for a, b in zip(self.__mul_opt_list__, self.__dsp_allocation_list__)]
        index = 0
        seq = 1
        while sum(self.__dsp_allocation_list__) < HW_Resources.DSP_NUM:
            index = find_seq_max_index(self.__c_r_list__, seq)
            ###################################################################
            '''
            tmp_simd_1 = self.__simd_list__[:]
            tmp_simd_index_1 = simd_index[:]
            tmp_pe_1 = self.__pe_list__[:]

            tmp_simd_2 = self.__simd_list__[:]
            tmp_pe_2 = self.__pe_list__[:]

            if simd_index[index] + 1 < len(simd_pre[index]):
                if self.__pe_list__[index] == self.__weight_list__[index][6]:
                    # option 1 increase simd
                    simd_index[index] = simd_index[index] + 1
                    self.__simd_list__[index] = simd_pre[index][simd_index[index]]
                else:
                    # option 1 increase simd
                    tmp_simd_index_1[index] = tmp_simd_index_1[index] + 1
                    tmp_simd_1[index] = simd_pre[index][tmp_simd_index_1[index]]
                    tmp_dsp_1 = [a * b for a, b in zip(tmp_simd_1, tmp_pe_1)]
                    tmp_c_r_1 = [a / b for a, b in zip(self.__mul_opt_list__, tmp_dsp_1)]
                    # option 2 increase pe
                    tmp_pe_2[index] = tmp_pe_2[index] + 1
                    tmp_dsp_2 = [a * b for a, b in zip(tmp_simd_2, tmp_pe_2)]
                    tmp_c_r_2 = [a / b for a, b in zip(self.__mul_opt_list__, tmp_dsp_2)]
                    if tmp_c_r_1[index] <= tmp_c_r_2[index]:
                        simd_index[index] = simd_index[index] + 1
                        self.__simd_list__[index] = simd_pre[index][tmp_simd_index_1[index]]
                    else:
                        self.__pe_list__[index] = tmp_pe_2[index]
            else:
                self.__pe_list__[index] = self.__pe_list__[index] + 1

            self.__dsp_allocation_list__ = [a * b for a, b in zip(self.__simd_list__, self.__pe_list__)]
            self.__c_r_list__ = [a / b for a, b in zip(self.__mul_opt_list__, self.__dsp_allocation_list__)]
            print(self.__c_r_list__)
            '''
            #################################################################################
            if simd_index[index] + 1 < len(simd_pre[index]):
                simd_index[index] = simd_index[index] + 1
                if sum(self.__dsp_allocation_list__) + simd_pre[index][simd_index[index]] - simd_pre[index][
                    simd_index[index - 1]] > HW_Resources.DSP_NUM:
                    if seq < len(self.__weight_list__):
                        seq = seq + 1
                        continue
                    else:
                        break
                else:
                    self.__simd_list__[index] = simd_pre[index][simd_index[index]]
            else:
                if sum(self.__dsp_allocation_list__) + self.__simd_list__[index] > HW_Resources.DSP_NUM:
                    if seq < len(self.__weight_list__):
                        seq = seq + 1
                        continue
                    else:
                        break
                else:
                    self.__pe_list__[index] = self.__pe_list__[index] + 1

            self.__dsp_allocation_list__ = [a * b for a, b in zip(self.__simd_list__, self.__pe_list__)]
            self.__c_r_list__ = [a / b for a, b in zip(self.__mul_opt_list__, self.__dsp_allocation_list__)]
        print("Last index: ", index)
        print("SIMD: ", self.__simd_list__)
        print("PE: ", self.__pe_list__)
        print("TOTAL: ", sum(self.__dsp_allocation_list__))

    def used_memory(self):
        i = 0
        while i < len(self.__weight_list__):
            self.__used_memory[i] = (self.__IFM_depth[i] + self.__IFM_after_gen_depth[i] + self.__weight_stream_depth[i] \
                                     + self.__conv_result_depth[i] + self.__pooling_result_depth[i]) * 512 / 8
            self.__used_memory[i] = self.__used_memory[i] \
                                    + (self.__ifm_with_padding_list__[i][0] * self.__pre_store_column[i] \
                                       * self.__ifm_with_padding_list__[i][2]) * self.__ifm_data_width__
            i = i + 1
        return sum(self.__used_memory)

    def used_bandwidth(self):
        i = 0
        while i < len(self.__weight_list__):
            if self.__weight_list__[i][0] == "fl_en":
                self.__used_bandwidth[i] = HW_Resources.CTC * self.__used_memory[i]
            else:
                self.__used_bandwidth[i] = HW_Resources.SINGLE_DSP_BAND * self.__dsp_allocation_list__[i] / \
                                           (self.__ifm_with_padding_list__[i][0] * self.__pre_store_column[i] \
                                            * self.__ifm_with_padding_list__[i][2] \
                                            * self.__ifm_data_width__)
            i = i + 1
        return sum(self.__used_bandwidth)

    def update_depth(self):
        i = 0
        while i < len(self.__weight_list__):
            self.__IFM_depth[i] = self.__ifm_with_padding_list__[i][0] * self.__pre_store_column[i] \
                                  * math.ceil(self.__weight_list__[i][5] / self.__simd_list__[i]) \
                                  * math.ceil(self.__simd_list__[i] / (512 / 8 / self.__filter_data_width__))

            if self.__weight_list__[i][1] == "pl_en":
                if self.__weight_list__[i + 1][0] == "fl_en":
                    self.__pooling_result_depth[i] = self.__ifm_with_padding_list__[i][0] * self.__pre_store_column[i] \
                                                     * math.ceil(self.__weight_list__[i][5] / self.__simd_list__[i]) \
                                                     * math.ceil(
                        self.__simd_list__[i] / (512 / 8 / self.__filter_data_width__))
                else:
                    self.__pooling_result_depth[i] = math.ceil(
                        self.__ofm_list__[i + 1][0] * self.__ofm_list__[i + 1][2] / (
                                    512 / 8 / self.__filter_data_width__))
            else:
                self.__pooling_result_depth[i] = 0
            i = i + 1

    def on_chip_mem_allocation(self):
        column_ratio_list = []
        for i in self.__weight_list__:
            self.__pre_store_column.append(0)
            self.__IFM_depth.append(0)
            self.__pooling_result_depth.append(0)
            self.__used_memory.append(0)
            self.__used_bandwidth.append(0)
            column_ratio_list.append(0)
        # initial basic allocation
        self.__pre_store_filter = [2 * b for b in self.__pe_list__]
        print(self.__pre_store_filter)
        i = 0
        while i < len(self.__pre_store_filter):
            self.__weight_stream_depth.append(math.ceil(self.__weight_list__[i][5] / self.__simd_list__[i]) \
                                              * math.ceil(
                self.__simd_list__[i] / (512 / 8 / self.__filter_data_width__)) \
                                              * self.__weight_list__[i][3] * self.__weight_list__[i][4] \
                                              * self.__pre_store_filter[i])

            self.__IFM_after_gen_depth.append(math.ceil(self.__weight_list__[i][5] / self.__simd_list__[i]) \
                                              * math.ceil(
                self.__simd_list__[i] / (512 / 8 / self.__filter_data_width__)) \
                                              * self.__pre_store_filter[i])

            if i < len(self.__pre_store_filter) - 1:
                if self.__weight_list__[i][1] == "pl_en":
                    self.__conv_result_depth.append(math.ceil(
                        2 * self.__ofm_list__[i][0] * self.__ofm_list__[i][2] / (512 / 8 / self.__filter_data_width__)))
                elif self.__weight_list__[i + 1][2] == "pd_en":
                    self.__conv_result_depth.append(
                        math.ceil(4 * self.__ofm_list__[i][2] / (512 / 8 / self.__filter_data_width__)))
                else:
                    self.__conv_result_depth.append(math.ceil(
                        self.__ofm_list__[i][0] * self.__ofm_list__[i][1] * self.__ofm_list__[i][2] / (
                                    512 / 8 / self.__filter_data_width__)))
            else:
                self.__conv_result_depth.append(math.ceil(
                    self.__ofm_list__[i][0] * self.__ofm_list__[i][1] * self.__ofm_list__[i][2] / (
                                512 / 8 / self.__filter_data_width__)))

            if self.__weight_list__[i][0] == "fl_en":
                self.__pre_store_column[i] = self.__ifm_with_padding_list__[i][1]
            else:
                self.__pre_store_column[i] = self.__weight_list__[i][4]
            self.update_depth()
            i = i + 1

        # increase pre-col
        self.used_bandwidth()
        tmp_band_list = []
        tmp_index_list = []
        i = 0
        while i < len(self.__weight_list__):
            if self.__weight_list__[i][0] == "fl_dis":
                tmp_band_list.append(self.__used_bandwidth[i])
                tmp_index_list.append(i)
            i = i + 1
        print("tmp_band_list:", tmp_band_list)
        print("tmp_index_list", tmp_index_list)
        i = 1
        while self.used_memory() < HW_Resources.ON_CHIP_MEM:
            index = find_seq_max_index(tmp_band_list, i)
            if self.__pre_store_column[tmp_index_list[index]] < self.__ifm_with_padding_list__[tmp_index_list[index]][
                1]:
                self.__pre_store_column[tmp_index_list[index]] = self.__pre_store_column[tmp_index_list[index]] + 1
                self.update_depth()
                self.used_bandwidth()
                i = 1
            else:
                i = i + 1

            if i == len(tmp_band_list):
                break

        print("weight_stream_depth:", self.__weight_stream_depth)
        print("DSP:", self.__dsp_allocation_list__)
        print("IFM_depth:", self.__IFM_depth)
        print("IFM_after_gen_depth:", self.__IFM_after_gen_depth)
        print("conv_result_depth:", self.__conv_result_depth)
        print("pooling_result_depth:", self.__pooling_result_depth)
        print("pre_store_column:", self.__pre_store_column)
        print("used mem:", self.used_memory())
        print("used bandwidth:", self.used_bandwidth())
        print("remain bandwidth:", HW_Resources.TOTAL_MEM_BAND - self.used_bandwidth())

    def print_define(self):
        f = open("define_.h", "w")
        f.write("#define WIDTH 512\n")
        f.write("//weight\n")
        f.write("#define WEIGHT_PRECISION %d\n" % (self.__filter_data_width__ * 8))
        i = 0
        while i < len(self.__weight_list__):
            f.write("#define KERNELDim_%d %d\n" % (i + 1, self.__weight_list__[i][3]))
            f.write("#define OFM_Channels_%d %d\n" % (i + 1, self.__ofm_list__[i][2]))
            i = i + 1
        f.write("\n")

        f.write("//input feature map\n")
        i = 0
        while i < len(self.__weight_list__):
            f.write("#define IFMDim_%d %d\n" % (i + 1, self.__ifm_list__[i][0]))
            f.write("#define IFM_Channels_%d %d\n" % (i + 1, self.__ifm_list__[i][2]))
            f.write("#define IFM_Channels_occupy_rows_%d %d\n" % (
            i + 1, math.ceil(self.__ifm_list__[i][2] / (512 / self.__filter_data_width__ / 8))))
            f.write("#define STRIDE_%d %d\n" % (i + 1, self.__weight_list__[i][7]))
            i = i + 1
        f.write("\n")

        f.write("//output after conv, ignore pooling\n//OFMDim=(IFMDim+2-KERNELDim)/STRIDE + 1, if padding\n")
        f.write("#define ACTIVATION_PRECISION %d\n" % (self.__ifm_data_width__ * 8))
        i = 0
        while i < len(self.__weight_list__):
            f.write("#define OFMDim_%d %d\n" % (i + 1, self.__ofm_list__[i][0]))
            i = i + 1
        f.write("\n")

        f.write("//PE array allocation\n")
        i = 0
        while i < len(self.__weight_list__):
            f.write("#define SIMD_%d %d\n" % (i + 1, self.__simd_list__[i]))
            f.write("#define PE_%d %d\n" % (i + 1, self.__pe_list__[i]))
            i = i + 1
        f.write("\n")

        f.write("//pre-store Input feature map for each layer(#column)\n")
        i = 0
        while i < len(self.__weight_list__):
            f.write("#define IFM_%d_precol %d\n" % (i + 1, self.__pre_store_column[i]))
            i = i + 1
        f.write("\n")

        f.write("//pre-store kernel number for each layer\n")
        i = 0
        while i < len(self.__weight_list__):
            f.write("#define Kernel_%d_pre %d\n" % (i + 1, self.__pre_store_filter[i]))
            i = i + 1
        f.write("\n")

        f.write("//Threshold\n")
        f.write("#define SIMD_W_TH %d\n" % (512 / self.__filter_data_width__ / 8))
        f.write("#define SIMD_I_TH %d\n" % (512 / self.__ifm_data_width__ / 8))
        f.write("\n")

        f.write("//SIMD Occupied Rows\n//L_I_SIMD_R=SIMD/SIMD_I_TH\n")
        i = 0
        while i < len(self.__weight_list__):
            f.write("#define L%d_W_SIMD_R %d\n" % (
            i + 1, (math.ceil(self.__simd_list__[i] / (512 / self.__filter_data_width__ / 8)))))
            f.write("#define L%d_I_SIMD_R %d\n" % (
            i + 1, (math.ceil(self.__simd_list__[i] / (512 / self.__ifm_data_width__ / 8)))))
            i = i + 1
        f.write("\n")

        f.write("//simd__loop_times=IFM_Channels_/SIMD_\n")
        i = 0
        while i < len(self.__weight_list__):
            f.write("#define simd_%d_loop_times %d\n" % (i + 1, self.__ifm_list__[i][2] / self.__simd_list__[i]))
            i = i + 1
        f.write("\n")

        f.write("//layer_i_filter_slide_number && layer_i_left_pe\n")
        i = 0
        while i < len(self.__weight_list__):
            f.write("#define layer%d_filter_slide_number %d\n" % (
            i + 1, math.ceil(self.__weight_list__[i][6] / self.__pe_list__[i])))
            if self.__weight_list__[i][6] % self.__pe_list__[i] == 0:
                f.write("#define layer%d_left_pe %d\n" % (i + 1, self.__pe_list__[i]))
            else:
                f.write("#define layer%d_left_pe %d\n" % (i + 1, (
                            self.__weight_list__[i][6] - math.ceil(self.__weight_list__[i][6] / self.__pe_list__[i]) *
                            self.__pe_list__[i]) % self.__pe_list__[i]))
            i = i + 1
        f.write("\n")

        f.write("//OFM_Cha_Occupy_Row=OFM_Channel/SIMD_I_TH\n")
        i = 0
        while i < len(self.__weight_list__):
            f.write("#define OFM_%d_Cha_Occupy_Row %d\n" % (
            i + 1, math.ceil(self.__weight_list__[i][6] * self.__ifm_data_width__ * 8 / 512)))
            i = i + 1
        f.write("\n")

        f.write(
            "//weight start and end index\n//start = previous_end\n//end = IFM_Channel * KERNELDim * KERNELDim * OFM_Channel * OFMDim * OFMDim( (/SIMD) or (*WEIGHT_PRECISION/512) ) + current_start\n")
        i = 0
        while i < len(self.__weight_list__):
            if i == 0:
                f.write("#define w%d_start %d\n" % (i + 1, 0))
                if self.__simd_list__[i] <= 512 / self.__filter_data_width__ / 8:
                    f.write("#define w%d_end %d\n" % (i + 1, self.__ifm_list__[i][2] * self.__weight_list__[i][3] \
                                                      * self.__weight_list__[i][4] * self.__ofm_list__[i][2] \
                                                      * self.__ofm_list__[i][0] * self.__ofm_list__[i][1] \
                                                      / self.__simd_list__[i]))
                else:
                    f.write("#define w%d_end %d\n" % (i + 1, self.__ifm_list__[i][2] * self.__weight_list__[i][3] \
                                                      * self.__weight_list__[i][4] * self.__ofm_list__[i][2] \
                                                      * self.__ofm_list__[i][0] * self.__ofm_list__[i][1] \
                                                      * self.__filter_data_width__ * 8 / 512))
            else:
                f.write("#define w%d_start w%d_end\n" % (i + 1, i))
                if self.__simd_list__[i] <= 512 / self.__filter_data_width__ / 8:
                    f.write("#define w%d_end (%d + w%d_end)\n" % (
                    i + 1, self.__ifm_list__[i][2] * self.__weight_list__[i][3] \
                    * self.__weight_list__[i][4] * self.__ofm_list__[i][2] \
                    * self.__ofm_list__[i][0] * self.__ofm_list__[i][1] \
                    / self.__simd_list__[i], i))
                else:
                    f.write("#define w%d_end (%d + w%d_end)\n" % (
                    i + 1, self.__ifm_list__[i][2] * self.__weight_list__[i][3] \
                    * self.__weight_list__[i][4] * self.__ofm_list__[i][2] \
                    * self.__ofm_list__[i][0] * self.__ofm_list__[i][1] \
                    * self.__filter_data_width__ * 8 / 512, i))
            i = i + 1
        f.write("\n")

        f.write("//weight_stream_depth\n")
        i = 0
        while i < len(self.__weight_list__):
            f.write("const int w%d_stream_depth=%d;\n" % (i + 1, math.ceil(self.__weight_stream_depth[i]/2)))
            i = i + 1
        f.write("\n")

        f.write("//IFM_stream_depth\n")
        i = 0
        while i < len(self.__weight_list__):
            f.write("const int ifm_%d_stream_depth=%d;\n" % (i + 1, math.ceil(self.__IFM_depth[i]/2)))
            i = i + 1
        f.write("\n")

        f.write("//IFM_gen_stream_depth\n")
        i = 0
        while i < len(self.__weight_list__):
            f.write("const int ifm_%d_gen_stream_depth=%d;\n" % (i + 1, math.ceil(self.__IFM_after_gen_depth[i]/2)))
            i = i + 1
        f.write("\n")

        f.write("//OFM_result_depth\n")
        i = 0
        while i < len(self.__weight_list__):
            f.write("const int mul_%d_result_stream_depth=%d;\n" % (i + 1, math.ceil(self.__conv_result_depth[i]/2)))
            i = i + 1
        f.write("\n")

        f.write("//Pooling_result_depth\n")
        i = 0
        while i < len(self.__weight_list__):
            f.write("const int pooling_%d_result_stream_depth=%d;\n" % (i + 1, math.ceil(self.__pooling_result_depth[i]/2)))
            i = i + 1
        f.write("\n")

        f.close()

    def print_design(self):
        f = open("one_layer.cpp", "w")
        f.write("""
///////////////////////////////////////////////////////////
/// Author Yun Feng(fengyun@usc.edu)
///        Arash Fayyazi(fayyazi@usc.edu)
///        Amirhossein Esmaili Dastjerdi(esmailid@usc.edu)
/// Date 04/12/2023
/// Org USC
////////////////////////////////////////////////////////////
#include \"define_.h\"
#include "bnn-library.h"
#include "activations.hpp"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "dma.h"
#include "mvau.hpp"
#include "conv.hpp"
using namespace hls;

//function -> for simd>32, read filter from buffer to stream
//buffer_weights -> where the actual data being stored
//out_stream -> where the data go
//start_point -> start index of buffer_weights for current layer
//end_point -> end index of buffer_weights for current layer
template<int start_point, int end_point1, int end_point2, int pe,int left_pe,int simd_r,int ofm >
void read_weight_long(ap_uint<512> *buffer_weights, stream<ap_uint<512> > *out_stream){
	cout<<"read weight long"<<endl;
	ap_uint<512> tmp[pe*simd_r];
	for(int i=0;i<ofm*ofm;i++){
		for(int j=start_point; j<end_point1;j=j+pe*simd_r){
			for(int k=0;k<pe;k++){
				for(int l=0;l<simd_r;l++){
					#pragma HLS PIPELINE=II
					tmp[k*simd_r+l]=buffer_weights[j+k*simd_r+l];
					out_stream[k*simd_r+l].write(tmp[k*simd_r+l]);
					//debug2<<tmp[k*simd_r+l]<<endl;
				}
			}
		}
		for(int j=end_point1;j<end_point2;j=j+left_pe*simd_r){
			for(int k=0;k<left_pe;k++){
				for(int l=0;l<simd_r;l++){
					#pragma HLS PIPELINE=II
					tmp[k*simd_r+l]=buffer_weights[j+k*simd_r+l];
					out_stream[k*simd_r+l].write(tmp[k*simd_r+l]);
				}
			}
		}

	}
}


//function -> for simd<=32, read filter from buffer to stream
//buffer_weights -> where the actual data being stored
//out_stream -> where the data go
//T -> simd*weight_precision, determine each element data width
//start_point -> start index of buffer_weights for current layer
//end_point -> end index of buffer_weights for current layer
template<int T,int start_point, int end_point1, int end_point2, int pe, int left_pe, int ofm>
void read_weight_short(ap_uint<512> *buffer_weights, stream<ap_uint<T> > *out_stream){
	cout<<"read weight short"<<endl;
	ap_uint<T> tmp[pe];
	loop1:
	for(int i=0;i<ofm*ofm;i++){
		loop2:
		for(int j=start_point; j<end_point1;j=j+pe){			
			loop3:
			for(int k=0;k<pe;k++){
				#pragma HLS PIPELINE=II
				tmp[k]=buffer_weights[j+k](T-1,0);
				out_stream[k].write(tmp[k]);
			}
		}
		loop4:
		for(int j=end_point1; j<end_point2;j=j+left_pe){
			loop5:
			for(int k=0;k<left_pe;k++){
				#pragma HLS PIPELINE=II
				tmp[k]=buffer_weights[j+k](T-1,0);
				out_stream[k].write(tmp[k]);
			}
		}
	}
}


//function -> for simd>32, read input data from buffer to stream
//buffer_IFM -> where the actual data being stored
//out_stream -> where the data go
//simd -> parallel in channel direction
//IFMDim -> input feature map width and height
//IFMCha -> input feature map channel
template<int simd, int IFMDim, int IFMCha>
void read_in_long(ap_uint<512> *buffer_IFM,stream<ap_uint<512> > &out_stream){
	for(int i=0;i<(((IFMDim)*(IFMDim)*(IFMCha))/simd);i++){
		out_stream.write(buffer_IFM[i]);

	}
}


//function -> for simd<=32, read input data from buffer to stream
//buffer_IFM -> where the actual data being stored
//out_stream -> where the data go
//T -> each element data width
//simd -> parallel in channel direction
//IFMDim -> input feature map width and height
//IFMCha -> input feature map channel
template<int T, int simd,int IFMDim, int IFMCha>
void read_in_short(ap_uint<512> *buffer_IFM,stream<ap_uint<T> > &out_stream){
	cout<<"read IFM"<<endl;
	for(int i=0;i<(((IFMDim)*(IFMDim)*(IFMCha))/simd);i++){
		out_stream.write(buffer_IFM[i](simd*ACTIVATION_PRECISION-1,0));
	}
}


template<int size>
void write_result(stream<ap_uint<512> > &input_stream, ap_uint<512> *buffer){
	for(int i=0;i<2048;i++){
			buffer[i]=input_stream.read();
			//debug3<<buffer[i]<<endl;
	}
}


//function -> for simd>32, based on input stream and each layer's info generate reused data and put it to a stream
//IFM -> where the actual data being stored
//out_stream -> where the data go
//simd_r -> for some cases, simd is larger than 32, we need to determine how many data should read out for the entire simd. eg. simd=64, then simd_r=2
//simd_loop->for some cases, channel=512, simd =128, there should have 4 loops to go through the entire channel.
//IFM_p -> pre-store input feature map columns
//KerDim -> kernel width and high
//pe -> pe number
//IFMDim -> IFM width and high
//OFM_s -> OFM width and high
//pe_loop_times -> determine how many times pe should loop to cover all kernels. eg. if we have 16 kernels and pe=3, then pe_loop_times=6
//left_pe -> determine the last pe loop's pe number. eg. if we have 16 kernels and pe=3, then left_pe=1. if 15 kernels and pe=3, then left_pe=3
//start_point and end_point are as same as the read_weight(), used to determine how many elements should be read out from weight stream
template<int simd_r,int simd_loop, int IFM_p, int KerDim, int pe, int IFMDim, int OFM_s, int pe_loop_times,int left_pe,int start_point,int end_point >
void Input_Generator_long(stream<ap_uint<512> > &IFM, stream<ap_uint<512> > *out_stream){
	cout<<"Input Generator long"<<endl;
	ap_uint<512> arr[IFMDim*IFM_p][simd_r*simd_loop];
	#pragma HLS array_partition variable=arr complete dim=2
	//#pragma HLS ARRAY_PARTITION variable=arr uram
	#pragma HLS bind_storage variable=arr type=RAM_1P impl=uram
	int head=0;
	int tail=0;
	int distance=0;

	int index_tail_move=0;
	int outer_loop=0;

	ap_uint<1> tail_end=0;
	ap_uint<1> full_enable=0;



	ap_uint<512> tmp_out_0[simd_r*simd_loop];
	ap_uint<512> tmp_out_1[simd_r*simd_loop];
		//deal with the distance value
loop1:
	while(outer_loop<(end_point-start_point)){//base on the number of weights that go through fifo determine the IFM loop-times

		int tail_head=tail-head;
		if(tail_head>0){
			distance= tail-head;
		}
		if(tail_head<0){
			distance=tail_head+IFM_p;
		}
		if(full_enable==0 && (tail_head==0)){
			distance=0;
		}
		if(full_enable==1 && (tail_head==0)){
			distance=IFM_p;
		}

		//based on distance determine whether read into arr or write out to stream
		if(distance>=KerDim){//write stream part(from arr to output)
			full_enable=1;
			//start to write out to output stream
			loop2:
			for(int i=0;i<OFM_s;i++){//IFM high dimension
				loop3:
				for(int j=0;j<pe_loop_times;j++){//move to following pe filters
					loop4:
					for(int g=0;g<simd_loop;g++){

						for(int m=0;m<KerDim;m++){//filter width dimension
							loop5:
							for(int n=0;n<KerDim;n++){ //filter high dimension
								int tmp;
								if((m+head)>=IFM_p)
									tmp=m+head-IFM_p;
								else
									tmp=m+head;


								if(j<(pe_loop_times-1)){//for some layers, filter number cannot be divided by pe, hence for the front filters use pe as loop times, for the remain filters use left_pe as loop times

									loop6:
									for(int L=0;L<simd_r;L++){
										#pragma HLS UNROLL
										tmp_out_0[L]=arr[tmp*IFMDim+n+i][g*simd_r+L];
										loop7:
										for(int k=0;k<pe;k++){//go over each pe filters
											#pragma HLS UNROLL
											out_stream[k*simd_r+L].write(tmp_out_0[L]);
											//debug2<<tmp_out_0[L]<<endl;
											outer_loop++;
										}
									}
								}
								else{
									loop8:
									for(int L=0;L<simd_r;L++){
										#pragma HLS UNROLL
										tmp_out_1[L]=arr[tmp*IFMDim+n+i][g*simd_r+L];
										loop9:
										for(int k=0;k<left_pe;k++){//go over each pe filters
											#pragma HLS UNROLL
											out_stream[k*simd_r+L].write(tmp_out_1[L]);
											//debug2<<tmp_out_1[L]<<endl;
											outer_loop++;
										}
									}
								}


							}
						}
					}
				}
			}
			head++;
			if(head==IFM_p){//store the head pointer address
				head=0;
			}

		}



		if(distance<IFM_p && index_tail_move<IFMDim){//read in IFM part(to arr)
			if(tail==IFM_p){//store the tail pointer address
				tail=0;
			}
			loop10:
			for(int i=0;i<IFMDim;i++){
				loop11:
				for(int j=0;j<simd_r*simd_loop;j++){
					#pragma HLS PIPELINE II=1
					arr[tail*IFMDim+i][j]=IFM.read();
				}
			}
			index_tail_move++; //store the tail index move operation times

			tail++;


		}
	}


}


//function -> for simd<=32, based on input stream and each layer's info generate reused data and put it to a stream
//IN_T-> input stream width
//IN_row-> the entire IFM channels occupy rows in the stream
//IFM -> where the actual data being stored
//out_stream -> where the data go
//T -> determine the read in and write out stream data width
//IFM_p -> pre-store input feature map columns
//KerDim -> kernel width and high
//pe -> pe number
//IFMDim -> IFM width and high
//OFM_s -> OFM width and high
//pe_loop_times -> determine how many times pe should loop to cover all kernels. eg. if we have 16 kernels and pe=3, then pe_loop_times=6
//left_pe -> determine the last pe loop's pe number. eg. if we have 16 kernels and pe=3, then left_pe=1. if 15 kernels and pe=3, then left_pe=3
//start_point and end_point are as same as the read_weight(), used to determine how many elements should be read out from weight stream
template<int IN_T, int IN_row, int T, int IFM_p, int KerDim,int pe, int IFMDim, int OFM_s, int pe_loop_times,int left_pe,int start_point,int end_point >
void Input_Generator_short_1(stream<ap_uint<IN_T> > &IFM, stream<ap_uint<T> > *out_stream){
	cout<<"Input_Generator_short_1"<<endl;
	ap_uint<IN_T> arr[IN_row*IFMDim*IFM_p];
	//#pragma HLS ARRAY_PARTITION variable=arr complete dim=0
	//#pragma HLS ARRAY_PARTITION variable=arr uram
	#pragma HLS bind_storage variable=arr type=RAM_1P impl=uram

	int head=0;
	int tail=0;
	int distance=0;

	int index_tail_move=0;
	int outer_loop=0;

	ap_uint<1> tail_end=0;
	ap_uint<1> full_enable=0;



		//deal with the distance value
loop1:
	while(outer_loop<(end_point-start_point)){//base on the number of weights that go through fifo determine the IFM loop-times
		//cout<<"here we are"<<endl;
		int tail_head=tail-head;
		if(tail_head>0){
			distance= tail_head;
		}
		if(tail_head<0){
			distance=tail_head+IFM_p;
		}
		if(full_enable==0 && (tail_head==0)){
			distance=0;
		}
		if(full_enable==1 && (tail_head==0)){
			distance=IFM_p;
		}

		//based on distance determine whether read into arr or write out to stream
		if(distance>=KerDim){//write stream part(from arr to output)
			full_enable=1;
			//start to write out to output stream
			loop2:
			for(int i=0;i<OFM_s;i++){//IFM high dimension
				loop3:
				for(int j=0;j<pe_loop_times;j++){//move to following pe filters
					loop4:
					for(int m=0;m<KerDim;m++){//filter width dimension
						loop5:
						for(int n=0;n<KerDim;n++){ //filter high dimension
							int tmp;
							if((m+head)>=IFM_p)
								tmp=m+head-IFM_p;
							else
								tmp=m+head;

							for (int g = 0; g < IN_row; g++) {
								ap_uint<IN_T> tmp_out = arr[tmp*IFMDim*IN_row + (n + i)* IN_row + g];
								if (j < (pe_loop_times - 1)) {//for some layers, filter number cannot be divided by pe, hence for the front filters use pe as loop times, for the remain filters use left_pe as loop times
									loop6:
									for (int k = 0; k < pe; k++) {//go over each pe filters
										#pragma HLS UNROLL
										for (int a = 0; a < IN_T / T; a++) {
											out_stream[k].write(tmp_out((a+1)*T-1,a*T));
											//debug3<<tmp_out<<endl;
											outer_loop++;
										}
									}
								}
								else {
									loop7:
									for (int k = 0; k < left_pe; k++) {//go over each pe filters
										#pragma HLS UNROLL
										for (int a = 0; a < IN_T / T; a++) {
											out_stream[k].write(tmp_out((a+1)*T - 1, a*T));
											//debug3<<tmp_out<<endl;
											outer_loop++;
										}
									}
								}
							}
						}
					}
				}
			}
			head++;
			if(head==IFM_p){//store the head pointer address
				head=0;
			}

		}



		if(distance < IFM_p && index_tail_move<IFMDim){//read in IFM part(to arr)
				if(tail==IFM_p){//store the tail pointer address
					tail=0;
				}
				loop8:
				for(int i=0;i<IFMDim;i++){
					for (int j = 0; j < IN_row; j++) {
						#pragma HLS PIPELINE II=1
						arr[tail*IFMDim*IN_row + i * IN_row + j] = IFM.read();
						//debug2<<arr[tail*IFMDim + i]<<endl;
					}

				}
				index_tail_move++; //store the tail index move operation times

				tail++;



		}
	}

}



//function -> for simd<=32, based on input stream and each layer's info generate reused data and put it to a stream
//IN_T-> input stream width
//IN_row-> the entire IFM channels occupy rows in the stream
//IFM -> where the actual data being stored
//out_stream -> where the data go
//T -> determine the read in and write out stream data width
//IFM_p -> pre-store input feature map columns
//KerDim -> kernel width and high
//pe -> pe number
//IFMDim -> IFM width and high
//OFM_s -> OFM width and high
//pe_loop_times -> determine how many times pe should loop to cover all kernels. eg. if we have 16 kernels and pe=3, then pe_loop_times=6
//left_pe -> determine the last pe loop's pe number. eg. if we have 16 kernels and pe=3, then left_pe=1. if 15 kernels and pe=3, then left_pe=3
//start_point and end_point are as same as the read_weight(), used to determine how many elements should be read out from weight stream
template<int IN_T, int IN_row, int T, int IFM_p, int KerDim,int pe, int IFMDim, int OFM_s, int pe_loop_times,int left_pe,int start_point,int end_point >
void Input_Generator_short_2(stream<ap_uint<IN_T> > &IFM, stream<ap_uint<T> > *out_stream){
	cout<<"Input_Generator_short_2"<<endl;
	ap_uint<IN_T> arr[IN_row*IFMDim*IFM_p];
	//#pragma HLS ARRAY_PARTITION variable=arr complete dim=0
	//#pragma HLS ARRAY_PARTITION variable=arr uram
	#pragma HLS bind_storage variable=arr type=RAM_1P impl=uram

	int head=0;
	int tail=0;
	int distance=0;

	int index_tail_move=0;
	int outer_loop=0;

	ap_uint<1> tail_end=0;
	ap_uint<1> full_enable=0;




		//deal with the distance value
loop1:
	while(outer_loop<(end_point-start_point)){//base on the number of weights that go through fifo determine the IFM loop-times
		//cout<<"here we are"<<endl;
		int tail_head=tail-head;
		if(tail_head>0){
			distance= tail_head;
		}
		if(tail_head<0){
			distance=tail_head+IFM_p;
		}
		if(full_enable==0 && (tail_head==0)){
			distance=0;
		}
		if(full_enable==1 && (tail_head==0)){
			distance=IFM_p;
		}

		//based on distance determine whether read into arr or write out to stream
		if(distance>=KerDim){//write stream part(from arr to output)
			full_enable=1;
			//start to write out to output stream
			loop2:
			for(int i=0;i<OFM_s;i++){//IFM high dimension
				//#pragma HLS loop_flatten off
				loop3:
				for(int j=0;j<pe_loop_times;j++){//move to following pe filters
					loop4:
					for(int m=0;m<KerDim;m++){//filter width dimension
						loop5:
						int tmp;
						if((m+head)>=IFM_p)
							tmp=m+head-IFM_p;
						else
							tmp=m+head;
						for(int n=0;n<KerDim;n++){ //filter high dimension


							for (int g = 0; g < IN_row; g++) {

								ap_uint<IN_T> tmp_out = arr[tmp*IFMDim*IN_row + (n + i)* IN_row + g];

								for (int a = 0; a < IN_T / T; a++) {
									#pragma HLS PIPELINE
									out_stream[0].write(tmp_out((a+1)*T-1,a*T));
									//debug3<<tmp_out<<endl;
									outer_loop++;
								}


							}
						}
					}
				}
			}
			head++;
			if(head==IFM_p){//store the head pointer address
				head=0;
			}

		}



		if(distance < IFM_p && index_tail_move<IFMDim){//read in IFM part(to arr)
			if(tail==IFM_p){//store the tail pointer address
				tail=0;
			}
			loop8:
			for(int i=0;i<IFMDim;i++){
				for (int j = 0; j < IN_row; j++) {
					#pragma HLS PIPELINE II=1
					arr[tail*IFMDim*IN_row + i * IN_row + j] = IFM.read();
					//debug2<<arr[tail*IFMDim + i]<<endl;
				}
			}
			index_tail_move++; //store the tail index move operation times

			tail++;
		}
	}

}



//function -> for simd>32, Mac filters with IFM by stream format
//IFM -> IFM data stream
//WEIGHT -> WEIGHT data stream
//out_stream -> where the data go
//IFM_r -> determines the number of elements that need to read out from stream. Eg, simd=64, then we need to read 2 times, then IFM_r=2
//WEIGHT_r -> determines the number of elements that need to read out from stream. Eg, simd=64, then we need to read 2 times, then WEIGHT_r=2
//pe -> pe number
//KerDim -> Kernel row and column number
//OFMDim -> OFM row and column number
//pe_loop_times -> pe window slide times, determine how many times pe should loop to cover all kernels. eg. if we have 16 kernels and pe=3, then pe_loop_times=6
//simd_loop_times -> determines the number of simd we need to use to calculate entire channel. Eg, channel=256, simd=64, then we need to move simd 4 times, then simd_loop_times=4
template<int IFM_r,int WEIGHT_r, int pe, int KerDim, int OFMDim, int pe_loop_times,int left_pe,int simd_loop_times>
void Mac_long(stream<ap_uint<512> > *IFM, stream<ap_uint<512> > *WEIGHT,stream<ap_uint<512> > &output){
		cout<<"Mac_long"<<endl;
		ap_uint<ACTIVATION_PRECISION> in[pe][IFM_r][SIMD_W_TH];
		ap_uint<ACTIVATION_PRECISION*2> tmp[pe];
		#pragma HLS ARRAY_PARTITION variable=in complete dim=0
		ap_uint<ACTIVATION_PRECISION> wt[pe][IFM_r][SIMD_W_TH];
		#pragma HLS ARRAY_PARTITION variable=wt complete dim=0
		ap_uint<ACTIVATION_PRECISION*2> tmp_[pe][IFM_r][SIMD_W_TH];
		#pragma HLS ARRAY_PARTITION variable=tmp_ complete dim=0

		ap_uint<512> tmp_i[pe][IFM_r];
		#pragma HLS ARRAY_PARTITION variable=tmp_i complete dim=0
		ap_uint<512> tmp_w[pe][IFM_r];
		#pragma HLS ARRAY_PARTITION variable=tmp_w complete dim=0

		for(int i=0;i<OFMDim;i++){	//IFM column
			loop1:
			for(int j=0;j<OFMDim;j++){	//IFM row
				ap_uint<512> result=0;	//in case output feature map channel larger than 512bits
				int index=0;	//record the number of channels than written to result
				loop2:
				for(int p=0;p<pe_loop_times;p++){	//pe window shift
					for(int f=0;f<pe;f++){
						#pragma HLS UNROLL
						tmp[f]=0;
					}
					loop3:
					for(int s=0;s<simd_loop_times;s++){	//for some cases channel number is way larger than simd, them need multiple simd to calculate all channel. Eg. channel num = 64, simd=32, then need 2 loops
						loop4:
						for(int l=0;l<KerDim;l++){	//traverse weight column
							//#pragma HLS PIPELINE OFF
							loop5:
							for(int m=0;m<KerDim;m++){	//traverse weight row
								#pragma HLS PIPELINE
								loop6:
								for(int k=0;k<pe;k++){	//pe pipeline
									#pragma HLS UNROLL
									if((k>(left_pe-1))&&(p==(pe_loop_times-1)))
										break;
									loop7:
									for(int r=0;r<IFM_r;r++){
										#pragma HLS UNROLL
										tmp_i[k][r]=IFM[k*IFM_r+r].read();
										tmp_w[k][r]=WEIGHT[k*IFM_r+r].read();
										loop8:
										for(int n=0;n<SIMD_W_TH;n++){	//simd parallel
											#pragma HLS UNROLL

											in[k][r][n]=tmp_i[k][r].range((n+1)*ACTIVATION_PRECISION-1,n*ACTIVATION_PRECISION) ;
											wt[k][r][n]=tmp_w[k][r].range((n+1)*WEIGHT_PRECISION-1,n*WEIGHT_PRECISION) ;

											tmp_[k][r][n]=in[k][r][n]*wt[k][r][n];
											tmp[k]+=tmp_[k][r][n];
											//debug3<<(tmp_i[r]((n+1)*ACTIVATION_PRECISION-1,n*ACTIVATION_PRECISION))<<"*"<<(tmp_w[r]((n+1)*WEIGHT_PRECISION-1,n*WEIGHT_PRECISION))<<endl;
										}
									}
								}
							}
						}
					}
					loop9:
					for(int r=0;r<pe;r++){	//traverse all tmp result
						#pragma HLS PIPELINE II=1
						result((index+1)*ACTIVATION_PRECISION-1,index*ACTIVATION_PRECISION)=tmp[r].range(ACTIVATION_PRECISION-1,0);	//write tmp to result one by one
						//debug2<<tmp[r].range(ACTIVATION_PRECISION-1,0)<<endl;
						if((p==(pe_loop_times-1))&&(r==(left_pe-1))){//if this is the last round and this round is end, write out the result to output directly
							output.write(result);
							//debug3<<result<<endl;
							break;
						}						
						if(index==SIMD_I_TH-1){	//if result is full, write to output and initialize result's value and index's value
							output.write(result);
							//debug2<<result<<endl;
							result=0;
							index=0;
						}
						else{
							index++;	//increase index every time it less than SIMD_I_TH-1
						}
					}
				}
			}
		}
}



//function -> for simd<=32, Mac filters with IFM by stream format
//IFM -> IFM data stream
//WEIGHT -> WEIGHT data stream
//out_stream -> where the data go
//IFM_size -> determine IFM data width, related to simd*ACTIVATION_PRECISION
//WEIGHT_size -> determine WEIGHT data width, related to simd*WEIGHT_PRECISION
//simd -> simd number
//pe -> pe number
//KerDim -> Kernel row and column number
//OFMDim -> OFM row and column number
//pe_loop_times -> pe window slide times, determine how many times pe should loop to cover all kernels. eg. if we have 16 kernels and pe=3, then pe_loop_times=6
//left_pe -> determine the last pe loop's pe number. eg. if we have 16 kernels and pe=3, then left_pe=1. if 15 kernels and pe=3, then left_pe=3
//simd_loop_times -> determines the number of simd we need to use to calculate entire channel. Eg, channel=256, simd=64, then we need to move simd 4 times, then simd_loop_times=4
template<int IFM_size,int WEIGHT_size, int simd, int pe, int KerDim, int OFMDim, int pe_loop_times,int left_pe, int simd_loop_times>
void Mac_short_1(stream<ap_uint<IFM_size> > *IFM, stream<ap_uint<WEIGHT_size>> *WEIGHT,stream<ap_uint<512>> &output){
	cout<<"Mac short_1"<<endl;
	ap_uint<ACTIVATION_PRECISION> in[simd];
	#pragma HLS ARRAY_PARTITION variable=in complete dim=0
	ap_uint<ACTIVATION_PRECISION> wt[simd];
	#pragma HLS ARRAY_PARTITION variable=wt complete dim=0
	ap_uint<ACTIVATION_PRECISION*2> tmp_[simd];
	#pragma HLS ARRAY_PARTITION variable=tmp_ complete dim=0

	ap_uint<IFM_size> tmp_i[pe];
	#pragma HLS ARRAY_PARTITION variable=tmp_i complete dim=0
	ap_uint<WEIGHT_size> tmp_w[pe];
	#pragma HLS ARRAY_PARTITION variable=tmp_w complete dim=0


	for(int i=0;i<OFMDim;i++){//IFM column

		for(int j=0;j<OFMDim;j++){//IFM row
			ap_uint<512> result=0;	//in case output feature map channel larger than 512bits
			int index=0;			//record the number of channels than written to result

			for(int p=0;p<pe_loop_times;p++){//pe window shift
				//ap_uint<pe*ACTIVATION_PRECISION> tmp=0; //store simd unit total multi-sum

				ap_uint<ACTIVATION_PRECISION*2> tmp[pe];
				for(int t=0;t<pe;t++){
					#pragma HLS UNROLL
					tmp[t]=0;
				}
				for(int s=0;s<simd_loop_times;s++){
					//#pragma HLS PIPELINE
					loop1:
					for(int l=0;l<KerDim;l++){//traverse weight column
						//#pragma HLS PIPELINE
						#pragma HLS loop_flatten off
						loop2:
						for(int m=0;m<KerDim;m++){	//traverse weight row
							#pragma HLS PIPELINE
							loop3:
							for(int k=0;k<pe;k++){
								#pragma HLS UNROLL
								if((k>(left_pe-1))&&(p==(pe_loop_times-1)))
									break;

								tmp_i[k]=IFM[k].read();

								tmp_w[k]=WEIGHT[k].read();
								loop4:
								for(int n=0;n<simd;n++){//simd parallel
									#pragma HLS UNROLL
									in[n].range()=tmp_i[k].range((n+1)*ACTIVATION_PRECISION-1,n*ACTIVATION_PRECISION);
									wt[n].range()=tmp_w[k].range((n+1)*WEIGHT_PRECISION-1,n*WEIGHT_PRECISION);
									//#pragma HLS BIND_OP varibale=tmp_ op=mul impl=dsp
									tmp_[n]=in[n]*wt[n];
									tmp[k]+=tmp_[n];
									//debug2<<"channel:"<<(k+p*pe)<<" "<<tmp[k]<<"="<<in<<"*"<<wt<<endl;
								}

							}
						}
					}
				}
				for(int r=0;r<pe;r++){//traverse all tmp result
					#pragma HLS PIPELINE II=1
					result((index+1)*ACTIVATION_PRECISION-1,index*ACTIVATION_PRECISION)=tmp[r].range(ACTIVATION_PRECISION-1,0); //write tmp to result one by one
					//debug2<<tmp[r].range(ACTIVATION_PRECISION-1,0)<<endl;
					//debug2<<tmp[r]<<endl;
					if((p==(pe_loop_times-1))&&(r==(left_pe-1))){//if this is the last round and this round is end, write out the result to output directly
						output.write(result);
						//debug3<<result<<endl;
						break;
					}
					if(index==SIMD_I_TH-1){  //if result is full, write to output and initialize result's value and index's value
						output.write(result);
						//debug3<<result<<endl;
						result=0;
						index=0;
					}
					else{
						index++;	//increase index every time it less than SIMD_I_TH-1

					}
					//debug2<<"a"<<endl;
				}
				//debug2<<"b"<<endl;
			}
			//debug2<<"c"<<endl;
		}

		//debug2<<"d"<<endl;
	}

	//debug2<<"e"<<endl;
}

template<int IFM_size,int WEIGHT_size, int simd, int pe, int KerDim, int OFMDim, int pe_loop_times,int left_pe, int simd_loop_times>
void Mac_short_2(stream<ap_uint<IFM_size> > *IFM, stream<ap_uint<WEIGHT_size>> *WEIGHT,stream<ap_uint<512>> &output){
	cout<<"Mac short_2"<<endl;
	ap_uint<ACTIVATION_PRECISION> in[simd];
	#pragma HLS ARRAY_PARTITION variable=in complete dim=0
	ap_uint<ACTIVATION_PRECISION> wt[simd];
	#pragma HLS ARRAY_PARTITION variable=wt complete dim=0
	ap_uint<ACTIVATION_PRECISION*2> tmp_[simd];
	#pragma HLS ARRAY_PARTITION variable=tmp_ complete dim=0

	ap_uint<IFM_size> tmp_i[pe];
	#pragma HLS ARRAY_PARTITION variable=tmp_i complete dim=0
	ap_uint<WEIGHT_size> tmp_w[pe];
	#pragma HLS ARRAY_PARTITION variable=tmp_w complete dim=0


	for(int i=0;i<OFMDim;i++){//IFM column

		for(int j=0;j<OFMDim;j++){//IFM row
			ap_uint<512> result=0;	//in case output feature map channel larger than 512bits
			int index=0;			//record the number of channels than written to result

			for(int p=0;p<pe_loop_times;p++){//pe window shift
				//ap_uint<pe*ACTIVATION_PRECISION> tmp=0; //store simd unit total multi-sum

				ap_uint<ACTIVATION_PRECISION*2> tmp[pe];
				for(int t=0;t<pe;t++){
					#pragma HLS UNROLL
					tmp[t]=0;
				}
				for(int s=0;s<simd_loop_times;s++){
					#pragma HLS PIPELINE
					loop1:
					for(int l=0;l<KerDim;l++){//traverse weight column
						#pragma HLS PIPELINE
						loop2:
						for(int m=0;m<KerDim;m++){	//traverse weight row
							#pragma HLS PIPELINE
							loop3:
							for(int k=0;k<pe;k++){
								#pragma HLS UNROLL
								if((k>(left_pe-1))&&(p==(pe_loop_times-1)))
									break;

								tmp_i[k]=IFM[k].read();

								tmp_w[k]=WEIGHT[k].read();
								loop4:
								for(int n=0;n<simd;n++){//simd parallel
									#pragma HLS UNROLL
									in[n].range()=tmp_i[k].range((n+1)*ACTIVATION_PRECISION-1,n*ACTIVATION_PRECISION);
									wt[n].range()=tmp_w[k].range((n+1)*WEIGHT_PRECISION-1,n*WEIGHT_PRECISION);
									//#pragma HLS BIND_OP varibale=tmp_ op=mul impl=dsp
									tmp_[n]=in[n]*wt[n];
									tmp[k]+=tmp_[n];
									//debug2<<"channel:"<<(k+p*pe)<<" "<<tmp[k]<<"="<<in<<"*"<<wt<<endl;
								}

							}
						}
					}
				}
				for(int r=0;r<pe;r++){//traverse all tmp result
					#pragma HLS PIPELINE II=1
					result((index+1)*ACTIVATION_PRECISION-1,index*ACTIVATION_PRECISION)=tmp[r].range(ACTIVATION_PRECISION-1,0); //write tmp to result one by one
					//debug2<<tmp[r].range(ACTIVATION_PRECISION-1,0)<<endl;
					//debug2<<tmp[r]<<endl;
					if((p==(pe_loop_times-1))&&(r==(left_pe-1))){//if this is the last round and this round is end, write out the result to output directly
						output.write(result);
						//debug3<<result<<endl;
						break;
					}
					if(index==SIMD_I_TH-1){  //if result is full, write to output and initialize result's value and index's value
						output.write(result);
						//debug3<<result<<endl;
						result=0;
						index=0;
					}
					else{
						index++;	//increase index every time it less than SIMD_I_TH-1

					}
					//debug2<<"a"<<endl;
				}
				//debug2<<"b"<<endl;
			}
			//debug2<<"c"<<endl;
		}

		//debug2<<"d"<<endl;
	}

	//debug2<<"e"<<endl;
}


//function -> padding matrix
//mul_result -> mul_result stream from Mac
//output -> output stream
//occupy_row_num -> each channel occupied row number. Eg. channel numer = 64, occupy_row_num = 2
//OFMDim -> OFMDim from previous layer. (After conv)
template<int occupy_row_num, int OFMDim>
void Padding(stream<ap_uint<512> > &mul_result, stream<ap_uint<512>> &output){
	cout<<"Padding"<<endl;
	//debug2<<"xxxxxxxxxxxxxxxxxxx"<<endl;
	loop1:
	for(int i=0;i<OFMDim+2;i++){
		loop2:
		for(int j=0;j<OFMDim+2;j++){
			loop3:
			for(int k=0;k<occupy_row_num;k++){
#pragma HLS PIPELINE II=1
				if(i==0 || j==0 || i==OFMDim+1 || j==OFMDim+1){
					output.write(0);
					//debug2<<0<<" ";
				}
				else{
					ap_uint<512> tmp=mul_result.read();
					output.write(tmp);
					//debug2<<tmp<<" ";
				}
			}
		}
		//debug2<<endl;
	}
	//debug2<<endl;
}


//function -> max pooling matrix
//mul_result -> upper stream
//output -> output stream
//occupy_row_num -> each channel occupied row number. Eg. channel numer = 64, occupy_row_num = 2
//OFMDim -> OFMDim from previous layer. (After after padding)
template<int occupy_row_num, int OFMDim>//Question:for max pooling we need space to store the multiplication results from mac layer first
void Pooling(stream<ap_uint<512> > &mul_result, stream<ap_uint<512> > &output){
	cout<<"Pooling"<<endl;
	ap_uint<512> tmp1[OFMDim/2*occupy_row_num];
	//#pragma HLS ARRAY_PARTITION variable=tmp1 complete dim=0
	ap_uint<512> tmp2;

	ap_uint<512> tmp4;



//int ggg=0;

	loop1:
	for(ap_uint<16> i=0;i<OFMDim/2;i++){
		loop2:
		for(ap_uint<16> j=0;j<OFMDim/2;j++){
			loop3:
			for(ap_uint<16> k=0;k<occupy_row_num;k++){
				#pragma HLS PIPELINE II=1
				tmp1[j*occupy_row_num+k]=mul_result.read();
			}
			loop4:
			for(ap_uint<16> l=0;l<occupy_row_num;l++){
				tmp2=mul_result.read();
				ap_uint<512> tmp3=tmp1[j*occupy_row_num+l];
				ap_uint<ACTIVATION_PRECISION> tmp5[SIMD_I_TH];
				//#pragma HLS bind_storage variable=tmp5 type=RAM_1P impl=uram
				ap_uint<ACTIVATION_PRECISION> tmp6[SIMD_I_TH];
				//#pragma HLS bind_storage variable=tmp6 type=RAM_1P impl=uram
				for(ap_uint<16> q=0;q<SIMD_I_TH;q++){
					#pragma HLS UNROLL
					tmp5[q]=tmp2((q+1)*ACTIVATION_PRECISION-1,q*ACTIVATION_PRECISION);
					tmp6[q]=tmp3((q+1)*ACTIVATION_PRECISION-1,q*ACTIVATION_PRECISION);
					if(tmp6[q]<tmp5[q]){
						tmp6[q]=tmp5[q];
					}
					tmp3((q+1)*ACTIVATION_PRECISION-1,q*ACTIVATION_PRECISION)=tmp6[q];
				}
				tmp1[j*occupy_row_num+l]=tmp3;
			}
		}

		loop6:
		for(ap_uint<16> j=0;j<OFMDim/2;j++){
			loop7:
			for(ap_uint<16> k=0;k<2;k++){
				loop8:
				for(ap_uint<16> l=0;l<occupy_row_num;l++){
					tmp4=mul_result.read();
					ap_uint<512> tmp3=tmp1[j*occupy_row_num+l];
					ap_uint<ACTIVATION_PRECISION> tmp5[SIMD_I_TH];
					ap_uint<ACTIVATION_PRECISION> tmp6[SIMD_I_TH];
					loop9:
					for(ap_uint<16> m=0;m<SIMD_I_TH;m++){
						#pragma HLS UNROLL
						tmp5[m]=tmp4((m+1)*ACTIVATION_PRECISION-1,m*ACTIVATION_PRECISION);
						tmp6[m]=tmp3((m+1)*ACTIVATION_PRECISION-1,m*ACTIVATION_PRECISION);
						if(tmp6[m]<tmp5[m]){
							tmp6[m]=tmp5[m];
						}
						tmp3((m+1)*ACTIVATION_PRECISION-1,m*ACTIVATION_PRECISION)=tmp6[m];
					}
					tmp1[j*occupy_row_num+l]=tmp3;
				}
			}
		}

		loop10:
		for(ap_uint<16> m=0;m<OFMDim/2;m++){
			loop11:
			for(ap_uint<16> n=0;n<occupy_row_num;n++){
				#pragma HLS PIPELINE II=1
				output.write(tmp1[m*occupy_row_num+n]);

				//debug3<<tmp1[m*occupy_row_num+n];
			}
			//debug3<<" ";
		}
		//debug3<<endl;
	}
}

extern "C" {
void one_layer(""")
        i = 0
        while i < len(self.__weight_list__) + 2:
            if i == 0:
                f.write("ap_uint<512> *buffer_IFM,\n")
            elif i == len(self.__weight_list__) + 1:
                f.write("\t\tap_uint<512> *buffer_out){\n")
            else:
                f.write("\t\tap_uint<512> *buffer_weight%d,\n" % (i))
            i = i + 1

        i = 0
        while i < len(self.__weight_list__) + 2:
            if i == 0:
                f.write(
                    "\t#pragma HLS INTERFACE m_axi port= buffer_IFM offset=slave bundle=gmem0 max_read_burst_length=64 max_write_burst_length=64\n")
            elif i == len(self.__weight_list__) + 1:
                f.write(
                    "\t#pragma HLS INTERFACE m_axi port=buffer_out offset=slave bundle=gmem%d max_read_burst_length=64 max_write_burst_length=64\n" % (
                        i))
            else:
                f.write(
                    "\t#pragma HLS INTERFACE m_axi port= buffer_weight%d offset=slave bundle=gmem%d max_read_burst_length=64 max_write_burst_length=64\n" % (
                    i, i))
            i = i + 1

        i = 0
        while i < len(self.__weight_list__) + 2:
            if i == 0:
                f.write("\t#pragma HLS INTERFACE s_axilite port=buffer_IFM bundle=control\n")
            elif i == len(self.__weight_list__) + 1:
                f.write("\t#pragma HLS INTERFACE s_axilite port=buffer_out bundle=control\n")
                f.write("\t#pragma HLS INTERFACE s_axilite port=return bundle=control\n")
            else:
                f.write("\t#pragma HLS INTERFACE s_axilite port=buffer_weight%d bundle=control\n" % (i))
            i = i + 1
        f.write("\n")

        f.write("""/////////////////////////////////////////////////////////////
//     start to declare weight streams for every layer     //
/////////////////////////////////////////////////////////////\n""")
        i = 0
        while i < len(self.__weight_list__):
            if self.__simd_list__[i] > 512 / self.__filter_data_width__ / 8:
                f.write("\tstatic stream<ap_uint<512> > w%d[PE_%d*L%d_I_SIMD_R];\n" % (i + 1, i + 1, i + 1))
            else:
                f.write("\tstatic stream<ap_uint<SIMD_%d*WEIGHT_PRECISION> > w%d[PE_%d];\n" % (i + 1, i + 1, i + 1))
            i = i + 1
        f.write("\n")

        f.write("""////////////////////////////////////////////////////////////
//     start to declare IFM streams for every layer    /////
////////////////////////////////////////////////////////////\n""")
        i = 0
        while i < len(self.__weight_list__):
            if self.__simd_list__[i] > 512 / self.__filter_data_width__ / 8:
                f.write("\tstatic stream<ap_uint<512> > IFM%d(\"IFM%d\");\n" % (i + 1, i + 1))
                f.write("\tstatic stream<ap_uint<512> > IFM%d_through_G[PE_%d*L%d_I_SIMD_R];\n" % (i + 1, i + 1, i + 1))
            else:
                f.write("\tstatic stream<ap_uint<SIMD_%d*ACTIVATION_PRECISION> > IFM%d(\"IFM%d\");\n" % (
                i + 1, i + 1, i + 1))
                f.write("\tstatic stream<ap_uint<SIMD_%d*ACTIVATION_PRECISION> > IFM%d_through_G[PE_%d];\n" % (
                i + 1, i + 1, i + 1))
            i = i + 1
        f.write("\n")

        f.write("""//////////////////////////////////////////////////////////
//start to declare Mul_result streams for every layer/////
//////////////////////////////////////////////////////////\n""")
        i = 0
        while i < len(self.__weight_list__):
            f.write("\tstatic stream<ap_uint<512> > Mul_result_%d(\"Mul_result_%d\");\n" % (i + 1, i + 1))
            i = i + 1
        f.write("\n")

        f.write("""/////////////////////////////////////
//start to declare pooling stream////
/////////////////////////////////////\n""")
        i = 0
        while i < len(self.__weight_list__):
            if self.__pooling_result_depth[i] != 0:
                f.write("\tstatic stream<ap_uint<512> > Mul_pooling_%d(\"Mul_pooling_%d\");\n" % (i + 1, i + 1))
            i = i + 1
        f.write("\n")

        f.write("""///////////////////////////////////////////////////
//start to declare every streams' depth////////////
///////////////////////////////////////////////////\n""")
        i = 0
        while i < len(self.__weight_list__):
            f.write("\t#pragma HLS STREAM variable = w%d  depth = w%d_stream_depth\n" % (i + 1, i + 1))
            f.write("\t#pragma HLS STREAM variable = IFM%d  depth = ifm_%d_stream_depth\n" % (i + 1, i + 1))
            f.write(
                "\t#pragma HLS STREAM variable = IFM%d_through_G  depth = ifm_%d_gen_stream_depth\n" % (i + 1, i + 1))
            f.write(
                "\t#pragma HLS STREAM variable = Mul_result_%d depth = mul_%d_result_stream_depth\n" % (i + 1, i + 1))
            if self.__pooling_result_depth[i] != 0:
                f.write("\t#pragma HLS STREAM variable = Mul_pooling_%d depth = pooling_%d_result_stream_depth\n" % (
                i + 1, i + 1))
            f.write("\n")
            i = i + 1
        f.write("\n")

        f.write("""\t#pragma HLS dataflow   //begin parallel
////////////////////////read data in, IFM-> generator->PE array, W->PE array
""")
        i = 0
        while i < len(self.__weight_list__):
            if i > 0:
                if self.__weight_list__[i][2] == "pd_en":
                    if self.__weight_list__[i - 1][1] == "pl_en":
                        f.write(
                            "\tPadding<OFM_%d_Cha_Occupy_Row,OFMDim_%d/2>(Mul_pooling_%d,IFM%d);\n" % (i, i, i, i + 1))
                    else:
                        f.write("\tPadding<OFM_%d_Cha_Occupy_Row,OFMDim_%d>(Mul_result_%d,IFM%d);\n" % (i, i, i, i + 1))

            if self.__simd_list__[i] > 512 / self.__filter_data_width__ / 8:
                if i == 0:
                    f.write("\tread_in_long<SIMD_%d,IFMDim_%d+2,IFM_Channels_%d>(buffer_IFM,IFM%d);\n" % (
                    i + 1, i + 1, i + 1, i + 1))
                f.write(
                    "\tread_weight_long<0,(((w%d_end-w%d_start)/(OFMDim_%d*OFMDim_%d))/(KERNELDim_%d*KERNELDim_%d*PE_%d*L%d_I_SIMD_R))*(KERNELDim_%d*KERNELDim_%d*PE_%d*L%d_I_SIMD_R),(w%d_end-w%d_start)/(OFMDim_%d*OFMDim_%d),PE_%d,layer%d_left_pe,L%d_I_SIMD_R,OFMDim_%d>(buffer_weight%d,w%d);\n"
                    % (i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1,
                       i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1))
                f.write(
                    "\tInput_Generator_long<L%d_I_SIMD_R,simd_%d_loop_times,IFM_%d_precol,KERNELDim_%d,PE_%d,IFMDim_%d+2,OFMDim_%d,layer%d_filter_slide_number,layer%d_left_pe,w%d_start,w%d_end>(IFM%d,IFM%d_through_G);\n"
                    % (i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1))
                f.write(
                    "\tMac_long<L%d_I_SIMD_R,L%d_W_SIMD_R,PE_%d,KERNELDim_%d,OFMDim_%d,layer%d_filter_slide_number,layer%d_left_pe,simd_%d_loop_times>(IFM%d_through_G,w%d,Mul_result_%d);\n"
                    % (i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1))
            else:
                if i == 0:
                    f.write(
                        "\tread_in_short<SIMD_%d*ACTIVATION_PRECISION,SIMD_%d,IFMDim_%d+2,IFM_Channels_%d>(buffer_IFM,IFM%d);\n" % (
                        i + 1, i + 1, i + 1, i + 1, i + 1))
                f.write(
                    "\tread_weight_short<SIMD_%d*WEIGHT_PRECISION,0,(((w%d_end-w%d_start)/(OFMDim_%d*OFMDim_%d))/(KERNELDim_%d*KERNELDim_%d*PE_%d))*(KERNELDim_%d*KERNELDim_%d*PE_%d),(w%d_end-w%d_start)/(OFMDim_%d*OFMDim_%d),PE_%d,layer%d_left_pe,OFMDim_%d>(buffer_weight%d,w%d);\n"
                    % (i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1,
                       i + 1, i + 1, i + 1, i + 1, i + 1, i + 1))

                if self.__ifm_list__[i][2] <= self.__simd_list__[i]:
                    if self.__weight_list__[i][0] == "fl_dis":
                        f.write(
                            "\tInput_Generator_short_1<SIMD_%d*ACTIVATION_PRECISION, IFM_Channels_occupy_rows_%d,SIMD_%d*ACTIVATION_PRECISION,IFM_%d_precol,KERNELDim_%d,PE_%d,IFMDim_%d+2,OFMDim_%d,layer%d_filter_slide_number,layer%d_left_pe,w%d_start,w%d_end>(IFM%d,IFM%d_through_G);\n"
                            % (
                            i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1,
                            i + 1))
                    else:
                        f.write(
                            "\tInput_Generator_short_1<SIMD_%d*ACTIVATION_PRECISION, IFM_Channels_occupy_rows_%d,SIMD_%d*ACTIVATION_PRECISION,IFM_%d_precol,KERNELDim_%d,PE_%d,IFMDim_%d+2,OFMDim_%d,layer%d_filter_slide_number,layer%d_left_pe,w%d_start,w%d_end>(Mul_result_%d,IFM%d_through_G);\n"
                            % (i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i,
                               i + 1))
                    f.write(
                        "\tMac_short_1<SIMD_%d*ACTIVATION_PRECISION,SIMD_%d*WEIGHT_PRECISION,SIMD_%d,PE_%d,KERNELDim_%d,OFMDim_%d,layer%d_filter_slide_number,layer%d_left_pe,simd_%d_loop_times>(IFM%d_through_G,w%d,Mul_result_%d);\n"
                        % (i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1))
                else:
                    if self.__weight_list__[i][0] == "fl_dis":
                        f.write(
                            "\tInput_Generator_short_2<512,ACTIVATION_PRECISION,SIMD_%d*ACTIVATION_PRECISION,IFM_%d_precol,KERNELDim_%d,PE_%d,IFMDim_%d,OFMDim_%d,layer%d_filter_slide_number,layer%d_left_pe,w%d_start,w%d_end>(IFM%d,IFM%d_through_G);\n"
                            % (i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1))
                    else:
                        if self.__weight_list__[i - 1][1] == "pl_en":
                            f.write(
                                "\tInput_Generator_short_2<512,IFM_Channels_occupy_rows_%d,SIMD_%d*ACTIVATION_PRECISION,IFM_%d_precol,KERNELDim_%d,PE_%d,IFMDim_%d,OFMDim_%d,layer%d_filter_slide_number,layer%d_left_pe,w%d_start,w%d_end>(Mul_pooling_%d,IFM%d_through_G);\n"
                                % (
                                i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i, i + 1))
                        else:
                            f.write(
                                "\tInput_Generator_short_2<512,IFM_Channels_occupy_rows_%d,SIMD_%d*ACTIVATION_PRECISION,IFM_%d_precol,KERNELDim_%d,PE_%d,IFMDim_%d,OFMDim_%d,layer%d_filter_slide_number,layer%d_left_pe,w%d_start,w%d_end>(Mul_result_%d,IFM%d_through_G);\n"
                                % (
                                i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i, i + 1))
                    f.write(
                        "\tMac_short_2<SIMD_%d*ACTIVATION_PRECISION,SIMD_%d*WEIGHT_PRECISION,SIMD_%d,PE_%d,KERNELDim_%d,OFMDim_%d,layer%d_filter_slide_number,layer%d_left_pe,simd_%d_loop_times>(IFM%d_through_G,w%d,Mul_result_%d);\n"
                        % (i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1, i + 1))

            if self.__weight_list__[i][1] == "pl_en":
                f.write("\tPooling<OFM_%d_Cha_Occupy_Row,OFMDim_%d>(Mul_result_%d, Mul_pooling_%d);\n" % (
                i + 1, i + 1, i + 1, i + 1))
            f.write("\n")
            i = i + 1

        f.write("""\tap_uint<512> tmp = Mul_result_16.read();
\tbuffer_out[0]=tmp;
\tcout<<tmp<<endl;
\t
\t}
}""")
        f.close()

    def print_host(self):
        f = open("host.cpp", "w")
        f.write("""//***************from finn*************************//
#include <iostream>
#include <time.h>
#include <cmath>
#include <ctime>
#include <cstring>
#include <hls_stream.h>
#include <cstdlib>
#include <ap_int.h>
//#include "bnn-library.h"
////#include "weights.hpp"
//#include "activations.hpp"
//#include "interpret.hpp"
//#include "mvau.hpp"
//#include "conv.hpp"

//***************from hybrid_PU*************************//
#include <vector>
#include <ap_fixed.h>
#include <CL/cl.h>
//#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h>
#include "xcl2.hpp"

//***************other libs*************************//
#include <fstream>
#include <math.h>
#include "define_.h"

using namespace hls;
using namespace std;

//*****************************following part are class or function definition********************************//
ofstream debug("./debug_file1.txt");
#define DATA_RANGE 9
enum data_type{FM,WEIGHT};

template<class T>
class input_data{
public:
	input_data(){};
	input_data(int k,int c,int d, data_type t, int s=3, int p=10);//construct function
	void matrix_gen();//based on input_data parameters allocate corresponding memory space for it, and then randomly assign value to it
	void matrix_padding();//based on matrix that we create, allocate a new memory space and implement padding
	void organize_data();//organize IFM and weight into desired format
	void print_matrix();//print out the matrix
	void duplicate(int width);//for filters we duplicate them in the software side
	void print_vector();//print out the vector
	void soft_matrix_gen(int a, int b, int c);
	void pooling();
	input_data<T> &operator = (const input_data<T> &rhs);	//overload "=" operator to copy construct
	input_data operator * (input_data<T> & data2); //overload "*" operator to realize matrix conv
	vector<ap_uint<512>, aligned_allocator<ap_uint<512>> > return_vector();//used to avoid errors when passing data to hw side by opencl
private:
	int kernel,channel,dim;
	data_type dt;
	int simd;
	int pe;
	T ****matrix;
	vector<ap_uint<512>, aligned_allocator<ap_uint<512> > > data_vector;
	///////////////////////////////
	T ****soft_result;
	int result_kernel=1;
	int result_channel;
	int result_dim;
	data_type result_dt=FM;
};


template<class T>
input_data<T>::input_data(int k, int c, int d, data_type t,int s, int p){
	kernel=k;
	channel=c;
	dim=d;
	dt=t;
	simd=s;
	pe=p;
}

template<class T>
void input_data<T>::matrix_gen(){
	matrix=new T ***[kernel];
	for(int i=0;i<kernel;i++){
		matrix[i]=new T **[channel];
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			matrix[i][j]=new T *[dim];
		}
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			for(int k=0;k<dim;k++){
				matrix[i][j][k]=new T [dim];
			}
		}
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			for(int k=0;k<dim;k++){
				for(int l=0;l<dim;l++){
					matrix[i][j][k][l]= rand()%(DATA_RANGE-1)+1;
				}
			}
		}
	}
}

template<class T>
void input_data<T>::matrix_padding(){
	T ****tmp = new T ***[kernel];
	for(int i=0; i< kernel; i++){
		tmp[i] = new T **[channel];
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			tmp[i][j]=new T *[dim+2];
		}
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			for(int k=0;k<dim+2;k++){
				tmp[i][j][k]=new T [dim+2];
			}
		}
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			for(int k=0;k<dim+2;k++){
				for(int l=0;l<dim+2;l++){
					tmp[i][j][k][l]= 0;
				}
			}
		}
	}


	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			for(int k=1;k<dim+1;k++){
				for(int l=1;l<dim+1;l++){
					tmp[i][j][k][l]= matrix[i][j][k-1][l-1];
				}
			}
		}
	}

	//delete[] matrix;
	matrix = new T ***[kernel];
	for(int i=0; i< kernel; i++){
		matrix[i] = new T **[channel];
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			matrix[i][j]=new T *[dim+2];
		}
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			for(int k=0;k<dim+2;k++){
				matrix[i][j][k]=new T [dim+2];
			}
		}
	}
	for(int i=0;i<kernel;i++){
		for(int j=0;j<channel;j++){
			for(int k=0;k<dim+2;k++){
				for(int l=0;l<dim+2;l++){
					matrix[i][j][k][l]= tmp[i][j][k][l];
				}
			}
		}
	}
	dim = dim + 2;
	//delete[] tmp;

}

template<class T>
void input_data<T>::organize_data(){
	if(dt==FM){
		for(int m=0;m<kernel;m++){
			for(int i=0;i<dim;i++){
				for(int j=0;j<dim;j++){
					ap_uint<512> tmp=0;
					for(int k=0;k<channel;k++){
						tmp((k+1)*ACTIVATION_PRECISION-1,k*ACTIVATION_PRECISION)=matrix[m][k][j][i];
					}
					data_vector.push_back(tmp);
				}
			}
		}
	}
	else{
		for(int i=0;i<kernel;i=i+pe){
			for(int l=0;l<channel;l=l+simd){
				for(int j=0;j<dim;j++){
					for(int k=0;k<dim;k++){
						for(int pp=0;pp<pe;pp++){
							if(simd>(512/WEIGHT_PRECISION)){
								ap_uint<512> tmp=0;
								for(int ss=0;ss<simd;ss=ss+(512/WEIGHT_PRECISION)){
									for(int n=0;n<512/WEIGHT_PRECISION;n++){
										tmp((n+1)*WEIGHT_PRECISION-1,n*WEIGHT_PRECISION)=matrix[i+pp][l+ss+n][k][j];

										if(n== (512/WEIGHT_PRECISION-1)){
											data_vector.push_back(tmp);
											tmp=0;
										}
									}
								}
							}
							else{
								ap_uint<512> tmp=0;
								for(int ss=0;ss<simd;ss++){
									tmp((ss+1)*WEIGHT_PRECISION-1,ss*WEIGHT_PRECISION)=matrix[i+pp][l+ss][k][j];
									//cout<<"/////"<<matrix[i+pp][l+ss][k][j]<<endl;
								}
								//cout<<tmp<<endl;
								//cout<<endl;

								data_vector.push_back(tmp);

							}
							if(kernel==(pp+i+1))
								break;
						}
					}
				}
			}
		}
	}
}

template<class T>
void input_data<T>::print_matrix(){
	for(int i=0;i<kernel;i++){
		if(dt==FM)
			debug<<"IFM:"<<endl;
		else if(dt==WEIGHT)
			debug<<"kernel #"<<i<<":"<<endl;
		for(int j=0;j<channel;j++){
			debug<<"channel #"<<j<<":"<<endl;
			for(int k=0;k<dim;k++){
				for(int l=0;l<dim;l++){
					debug<<matrix[i][j][k][l]<<" ";
				}
				debug<<endl;
			}
			debug<<endl;
		}
		debug<<endl;
	}
}

template<class T>
void input_data<T>::duplicate(int width){
	int index=data_vector.size();
	for(int i=0;i<width*width-1;i++){
		for(int j=0;j<index;j++){
			data_vector.push_back(data_vector[j]);
		}
	}
}

template<class T>
void input_data<T>::print_vector(){
	for(unsigned int i=0;i<data_vector.size();i++)
		debug<<data_vector[i]<<endl;
}

template<class T>
void input_data<T>::pooling(){
	T ****tmp=new T ***[kernel];
		for(int i=0;i<kernel;i++){
			tmp[i]=new T **[channel];
		}
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				tmp[i][j]=new T *[dim/2];
			}
		}
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				for(int k=0;k<dim/2;k++){
					tmp[i][j][k]=new T [dim/2];
				}
			}
		}
		//processing part
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				for(int k=0;k<dim/2;k++){
					for(int l=0;l<dim/2;l++){
						int tmp_max=-99999;
						for(int m=0;m<2;m++){
							for(int n=0;n<2;n++){
								if(matrix[i][j][2*k+m][2*l+n]>tmp_max)
									tmp_max=matrix[i][j][2*k+m][2*l+n];
							}
						}
						tmp[i][j][k][l]= tmp_max;
					}
				}
			}
		}

		delete[] matrix;
		T ****matrix = new T ***[kernel];
		for(int i=0; i< kernel; i++){
			matrix[i] = new T **[channel];
		}
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				matrix[i][j]=new T *[dim/2];
			}
		}
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				for(int k=0;k<dim/2;k++){
					matrix[i][j][k]=new T [dim/2];
				}
			}
		}
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				for(int k=0;k<dim/2;k++){
					for(int l=0;l<dim/2;l++){
						matrix[i][j][k][l]= tmp[i][j][k][l];
					}
				}
			}
		}
		dim=dim/2;

}

template<class T>
input_data<T> &input_data<T>::operator = (const input_data<T> &rhs){

		kernel=rhs.result_kernel;
		channel=rhs.result_channel;
		dim=rhs.result_dim;
		dt=rhs.result_dt;

		matrix=new T ***[kernel];
		for(int i=0;i<kernel;i++){
			matrix[i]=new T **[channel];
		}
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				matrix[i][j]=new T *[dim];
			}
		}
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				for(int k=0;k<dim;k++){
					matrix[i][j][k]=new T [dim];
				}
			}
		}
		for(int i=0;i<kernel;i++){
			for(int j=0;j<channel;j++){
				for(int k=0;k<dim;k++){
					for(int l=0;l<dim;l++){
						matrix[i][j][k][l]= rhs.soft_result[i][j][k][l];
					}
				}
			}
		}

	return *this;
}

template<class T>
input_data<T> input_data<T>::operator * (input_data<T> &data2){

	result_channel=data2.kernel;
	result_dim=dim-data2.dim+1;

	soft_result=new T ***[result_kernel];
	for(int i=0;i<result_kernel;i++){
		soft_result[i]=new T **[result_channel];
	}
	for(int i=0;i<result_kernel;i++){
		for(int j=0;j<result_channel;j++){
			soft_result[i][j]=new T *[result_dim];
		}
	}
	for(int i=0;i<result_kernel;i++){
		for(int j=0;j<result_channel;j++){
			for(int k=0;k<result_dim;k++){
				soft_result[i][j][k]=new T [result_dim];
			}
		}
	}

	for(int i=0;i<result_kernel;i++){
		for(int j=0;j<result_channel;j++){
			for(int k=0;k<result_dim;k++){
				for(int l=0;l<result_dim;l++){
					soft_result[i][j][k][l]=0;
				}
			}
		}
	}

	for(int i=0;i<result_kernel;i++){

			for(int k=0;k<result_dim;k++){
				for(int l=0;l<result_dim;l++){

					for(int m=0;m<data2.kernel;m++){
						for(int n=0;n<data2.channel;n++){
							for(int p=0;p<data2.dim;p++){
								for(int q=0;q<data2.dim;q++){
									soft_result[i][m][k][l]+=matrix[i][n][k+p][l+q]*data2.matrix[m][n][p][q];
								}
							}
						}
					}
				}

		}
	}

	return *this;
}




template<class T>
vector<ap_uint<512>, aligned_allocator<ap_uint<512>> > input_data<T>::return_vector(){
	return data_vector;
}

int main(int argc, char** argv){
	if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }
	//generate all needed data
	srand((int) time(0));
""")
        f.write("\tinput_data<ap_uint<ACTIVATION_PRECISION> > IFM(1,IFM_Channels_1,IFMDim_1,FM);\n")
        i = 0
        while i < len(self.__weight_list__):
            f.write(
                "\tinput_data<ap_uint<WEIGHT_PRECISION> > W%d(OFM_Channels_%d,IFM_Channels_%d,KERNELDim_%d,WEIGHT,SIMD_%d,PE_%d);\n"
                % (i + 1, i + 1, i + 1, i + 1, i + 1, i + 1))
            i = i + 1
        f.write("\n")

        i = 0
        while i < len(self.__weight_list__):
            f.write("\tinput_data<ap_uint<ACTIVATION_PRECISION> > OFM%d;\n"
                    % (i + 1))
            i = i + 1
        f.write("\n")

        f.write("\tIFM.matrix_gen();\n")
        i = 0
        while i < len(self.__weight_list__):
            f.write("\tW%d.matrix_gen();\n"
                    % (i + 1))
            i = i + 1
        f.write("\n")

        i = 0
        while i < len(self.__weight_list__):
            f.write("\tW%d.organize_data();\n"
                    % (i + 1))
            i = i + 1
        f.write("\n")

        i = 0
        while i < len(self.__weight_list__):
            if self.__weight_list__[i][2] == "pd_en":
                if i == 0:
                    f.write("\tIFM.matrix_padding();\n")
                    f.write("\tIFM.organize_data();\n")
                else:
                    f.write("\tOFM%d.matrix_padding();\n" % (i))
            if i == 0:
                f.write("\tOFM1 = IFM * W1;\n")
            else:
                f.write("\tOFM%d = OFM%d * W%d;\n" % (i + 1, i, i + 1))

            if self.__weight_list__[i][1] == "pl_en":
                f.write("\tOFM%d.pooling();\n" % (i + 1))

            i = i + 1
        f.write("\n")

        f.write('\tcout<<"Here0"<<endl;\n')
        f.write("""\tint out_size = 1*1*1*WIDTH/8;
    vector< ap_uint<WIDTH>, aligned_allocator<ap_uint<WIDTH>>> output_vector(out_size);
    vector< ap_uint<WIDTH>, aligned_allocator<ap_uint<WIDTH>> > IFM_(IFM.return_vector());\n""")
        i = 0
        while i < len(self.__weight_list__):
            f.write("\tvector< ap_uint<WIDTH>, aligned_allocator<ap_uint<WIDTH>> > W%d_(W%d.return_vector());\n" % (
            i + 1, i + 1))
            i = i + 1
        f.write("\n")

        f.write("\tint IFM_size = IFM.return_vector().size()*WIDTH/8;\n")
        i = 0
        while i < len(self.__weight_list__):
            f.write("\tint weight%d_size=W%d_.size()*WIDTH/8;//byte\n" % (i + 1, i + 1))
            i = i + 1
        f.write("\n")

        f.write("""\t//Open_cl host code area start
    cl_int err;
	std::string binaryFile = argv[1];
	auto devices = xcl::get_xil_devices();
	auto device = devices[0];

	OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
	OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
	OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

	auto fileBuf = xcl::read_binary_file(binaryFile);
	cl::Program::Binaries bins{{fileBuf.data(),fileBuf.size()}};
	devices.resize(1);
	OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
	OCL_CHECK(err, cl::Kernel one_layer(program, "one_layer", &err));
	// calculate buffer size

	//allocate buffer in global memory
	cout<<"***host allocate buffer in global memory***"<<endl;
	
	OCL_CHECK(err, cl::Buffer buffer_IFM(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, IFM_size, IFM_.data(), &err));\n""")
        i = 0
        while i < len(self.__weight_list__):
            f.write(
                "\tOCL_CHECK(err, cl::Buffer buffer_weight%d(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, weight%d_size, W%d_.data(), &err));\n"
                % (i + 1, i + 1, i + 1))
            i = i + 1
        f.write("""\tOCL_CHECK(err, cl::Buffer buffer_out(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, out_size, output_vector.data(), &err));

	//set the Kernel Arguments
	cout<<"***host set the Kernel Arguments***"<<endl;
	int narg=0;
	OCL_CHECK(err, err = one_layer.setArg(narg++, buffer_IFM));\n""")

        i = 0
        while i < len(self.__weight_list__):
            f.write("\tOCL_CHECK(err, err = one_layer.setArg(narg++, buffer_weight%d));\n"
                    % (i + 1))
            i = i + 1
        f.write("""\tOCL_CHECK(err, err = one_layer.setArg(narg++, buffer_out));\n""")

        f.write("""//copy data from host to device
	cout<<"***host copy data from host to device***"<<endl;
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_IFM}, 0 /* 0 means from host*/));
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({""")
        i = 0
        while i < len(self.__weight_list__):
            if i == 0:
                f.write("buffer_weight1,\n")
            elif i == len(self.__weight_list__) - 1:
                f.write("\t\t\t\t\t\t\t\t\t\t\t\t\t buffer_weight%d}, 0 /* 0 means from host*/));\n" % (i + 1))
            else:
                f.write("\t\t\t\t\t\t\t\t\t\t\t\t\t buffer_weight%d,\n" % (i + 1))
            i = i + 1

        f.write("""\t// Launch the Kernel
	cout<<"***host Launch the Kernel***"<<endl;
    OCL_CHECK(err, err = q.enqueueTask(one_layer));

	// Copy Result from Device Global Memory to Host Local Memory
	cout<<"***host Copy Result from Device Global Memory to Host Local Memory***"<<endl;
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out}, CL_MIGRATE_MEM_OBJECT_HOST));
    cout<<"***here***"<<endl;
    q.finish();
    cout<<"***end***"<<endl;
	return 0;
}""")


test = Compiler(2, 2)
test.read_network()
test.layer_info_calculation()
test.pe_simd_dsp_allocation()
test.on_chip_mem_allocation()
test.print_define()
test.print_design()
test.print_host()
# arr = [0,1,2,3,4,5,6,7,8,9,10]
# print(find_seq_max_index(arr,5))
# print(arr)
