# included lib
import math

TOTAL_DSP = 6840

MEM_BAND = 2666 * 1024 * 1024 / 8  # Byte/s
ON_CHIP_MEM = 4320 * 18 * 1024  # Bit
TOTAL_DSP_BAND = 21.3 * 1000000000  # 8 bit Operation per second per
Single_DSP_BAND = TOTAL_DSP_BAND / TOTAL_DSP  # Operation per second
DATA_WIDTH = 16  # input date width bits
WEIGHT_WIDTH = 16  # weight width bits
Frequency = 200 * 1000000  # Hz
CTC = TOTAL_DSP_BAND / MEM_BAND  # operation per Byte
filepath = "./output_.txt"
outfile = open(filepath, 'w')


# this class contains each layer's weight size and weight number
class layer:
    weight_number = 0
    weight_h = 0
    weight_w = 0
    weight_c = 0
    stride = 0
    method = ''

    def __init__(self, weight_num, weight_width, weight_high, weight_channel, each_stride, move_method):
        self.weight_number = weight_num
        self.weight_h = weight_high
        self.weight_w = weight_width
        self.weight_c = weight_channel
        self.stride = each_stride
        self.method = move_method  # in our design we only have same and valid


# this class only be used to store the input feature map
class input:
    channel = 0
    width = 0
    high = 0

    def __init__(self, input_width, input_high, input_channel):
        self.channel = input_channel
        self.width = input_width
        self.high = input_high


# this class store the size of each layer's output(after convolution with kernels)
class output:
    layer_number = 0
    channel = 0  # layer channel
    width = 0  # layer width
    high = 0  # layer high
    mul_opt = 0  # layer multiplication operation number
    add_opt = 0  # layer addition operation number
    C_R = 0  # layer complicate resource ratio
    dsp_num = 0  # layer allocated DSP number
    dsp_bandwidth = 0
    col = 1
    on_chip_mem = 0
    ctc = 0  # GOP/Byte
    pre_ker = 0  # pre-stored kernel number

    def __init__(self, input_channel, input_width, input_high):
        self.channel = input_channel
        self.width = input_width
        self.high = input_high


# calculate the output size and multiplication and addition operation number
def calculation_start(layer_in, layer_info, layer_out):
    if layer_info.method == 'same':
        layer_out.channel = layer_info.weight_number
        layer_out.high = layer_in.high
        layer_out.width = layer_in.width
        layer_out.mul_opt = layer_in.width * layer_in.high * layer_info.weight_w * layer_info.weight_h * layer_info.weight_c * layer_info.weight_number
        layer_out.add_opt = (
                                    layer_info.weight_w * layer_info.weight_h * layer_info.weight_c - 1) * layer_in.width * layer_in.high * layer_info.weight_number
    elif layer_info.method == 'valid':
        layer_out.channel = layer_info.weight_number
        layer_out.high = layer_in.high - layer_info.weight_h + 1
        layer_out.width = layer_in.width - layer_info.weight_w + 1
        layer_out.mul_opt = layer_out.high * layer_out.width * layer_info.weight_w * layer_info.weight_c * layer_info.weight_h * layer_info.weight_number
        layer_out.add_opt = (
                                    layer_info.weight_w * layer_info.weight_c * layer_info.weight_h - 1) * layer_out.channel * layer_out.high * layer_out.width


# calculate the output size and multiplication and addition operation number
def calculation_follow(layer_in, layer_info, layer_out):
    if layer_info.method == 'same':
        layer_out.channel = layer_info.weight_number
        layer_out.high = layer_in.high
        layer_out.width = layer_in.width
        layer_out.mul_opt = layer_in.width * layer_in.high * layer_info.weight_w * layer_info.weight_h * layer_info.weight_c * layer_info.weight_number
        layer_out.add_opt = (
                                    layer_info.weight_w * layer_info.weight_h * layer_info.weight_c - 1) * layer_in.width * layer_in.high * layer_info.weight_number
    elif layer_info.method == 'valid':
        layer_out.channel = layer_info.weight_number
        layer_out.high = layer_in.high - layer_info.weight_h + 1
        layer_out.width = layer_in.width - layer_info.weight_w + 1
        layer_out.mul_opt = layer_out.high * layer_out.width * layer_info.weight_w * layer_info.weight_c * layer_info.weight_h * layer_info.weight_number
        layer_out.add_opt = (
                                    layer_info.weight_w * layer_info.weight_c * layer_info.weight_h - 1) * layer_out.channel * layer_out.high * layer_out.width


# calculate current total used DSP number
def total_resource(list_):
    total = 0
    for i in list_:
        total = total + i.dsp_num
    return total


# find the layer index that has the maximum C_R rate
def find_max(list_):
    tmpCR = 0
    index_i = 0
    remains = TOTAL_DSP - total_resource(list_)
    for i in list_:
        if (i.C_R >= tmpCR) & (remains >= i.dsp_num):
            tmpCR = i.C_R

    for i in list_:
        if tmpCR == i.C_R:
            return index_i
        index_i = index_i + 1


# initial each input feature map size and each layer's kernel size
# --kernel number --width --high --channel --stride --conv method
input1 = input(32, 32, 3)
layer1 = layer(64, 3, 3, 3, 1, 'same')
layer2 = layer(64, 3, 3, 64, 1, 'same')
layer3 = layer(128, 3, 3, 64, 1, 'same')
layer4 = layer(128, 3, 3, 128, 1, 'same')
layer5 = layer(256, 3, 3, 128, 1, 'same')
layer6 = layer(256, 3, 3, 256, 1, 'same')
layer7 = layer(256, 3, 3, 256, 1, 'same')
layer8 = layer(512, 3, 3, 256, 1, 'same')
layer9 = layer(512, 3, 3, 512, 1, 'same')
layer10 = layer(512, 3, 3, 512, 1, 'same')
layer11 = layer(512, 3, 3, 512, 1, 'same')
layer12 = layer(512, 3, 3, 512, 1, 'same')
layer13 = layer(512, 3, 3, 512, 1, 'same')
layer14 = layer(512, 1, 1, 512, 1, 'valid')
layer15 = layer(512, 1, 1, 512, 1, 'valid')
layer16 = layer(10, 1, 1, 512, 1, 'valid')
conv_out1 = output(0, 0, 0)
conv_out2 = output(0, 0, 0)
conv_out3 = output(0, 0, 0)
conv_out4 = output(0, 0, 0)
conv_out5 = output(0, 0, 0)
conv_out6 = output(0, 0, 0)
conv_out7 = output(0, 0, 0)
conv_out8 = output(0, 0, 0)
conv_out9 = output(0, 0, 0)
conv_out10 = output(0, 0, 0)
conv_out11 = output(0, 0, 0)
conv_out12 = output(0, 0, 0)
conv_out13 = output(0, 0, 0)
conv_out14 = output(0, 0, 0)
conv_out15 = output(0, 0, 0)
conv_out16 = output(0, 0, 0)
out1 = output(64, 32, 32)
out2 = output(64, 16, 16)
out3 = output(128, 16, 16)
out4 = output(128, 8, 8)
out5 = output(256, 8, 8)
out6 = output(256, 8, 8)
out7 = output(256, 4, 4)
out8 = output(512, 4, 4)
out9 = output(512, 4, 4)
out10 = output(512, 2, 2)
out11 = output(512, 2, 2)
out12 = output(512, 2, 2)
out13 = output(512, 1, 1)
out14 = output(512, 1, 1)
out15 = output(512, 1, 1)
out16 = output(10, 1, 1)
layer_list = [layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9, layer10, layer11, layer12,
              layer13, layer14, layer15, layer16]
conv_out_list = [conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6, conv_out7, conv_out8, conv_out9,
                 conv_out10, conv_out11, conv_out12, conv_out13, conv_out14, conv_out15, conv_out16]
output_list = [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16]

# call the calculation function layer by layer
i = 0
for each_layer in layer_list:
    if i == 0:
        calculation_start(input1, each_layer, conv_out_list[i])
    else:
        calculation_follow(output_list[i - 1], each_layer, conv_out_list[i])
    i = i + 1

# print out each layer's output info and write it into file
total_operation = 0  # use to record the total operations
i = 1
for out in conv_out_list:
    out.layer_number = i
    outfile.write(
        "After conv with layer %d output width is %d, output high is %d, output channel is %d, mul operation is %d, add operation is %d\n" % (
            i, out.width, out.high, out.channel, out.mul_opt, out.add_opt))
    total_operation = total_operation + out.mul_opt
    print(
        "After conv with layer %d output width is %d, output high is %d, output channel is %d, mul operation is %d, add operation is %d" % (
            i, out.width, out.high, out.channel, out.mul_opt, out.add_opt))
    i = i + 1

########################################################################################################################
# Realize first algorithm ----- Computation resource allocation
# calculate the initial DSP number for each layer
for out in conv_out_list:  # calculate the initial dsp number for each layer
    out.dsp_num = math.ceil(math.pow(2, math.floor(math.log2(out.mul_opt / total_operation * TOTAL_DSP))))
    out.C_R = out.mul_opt / out.dsp_num

# start fine-tuning
while total_resource(conv_out_list) <= TOTAL_DSP:
    index = find_max(conv_out_list)
    if index is None:
        break
    if total_resource(conv_out_list) + conv_out_list[index].dsp_num <= TOTAL_DSP:
        conv_out_list[index].dsp_num = 2 * conv_out_list[index].dsp_num
        conv_out_list[index].C_R = conv_out_list[index].mul_opt / conv_out_list[index].dsp_num
    else:
        break

# additional adjust for first layer

tmp = math.ceil(conv_out_list[0].dsp_num / 3) * 3
if total_resource(conv_out_list) - conv_out_list[0].dsp_num + tmp <= TOTAL_DSP:
    conv_out_list[0].dsp_num = tmp
else:
    conv_out_list[0].dsp_num = math.floor(conv_out_list[0].dsp_num / 3) * 3

for out in conv_out_list:
    print("layer %d, DSP number %d, C_R %f" % (out.layer_number, out.dsp_num, out.C_R))
    outfile.write("layer %d, DSP number %d, C_R %f\n" % (out.layer_number, out.dsp_num, out.C_R))
print("total used DSP number %d, remain DSP number %d" % (
    total_resource(conv_out_list), TOTAL_DSP - total_resource(conv_out_list)))
outfile.write("total used DSP number %d, remain DSP number %d\n" % (
    total_resource(conv_out_list), TOTAL_DSP - total_resource(conv_out_list)))


########################################################################################################################
# Realize the second algorithm ---- Memory bandwidth resource allocation
def BW_total(list_):
    tmp = 0
    for out in list_:
        tmp += out.dsp_bandwidth
    return tmp


def BW_conv_total(list_):
    tmp = 0
    i = 0
    while i <= 12:
        tmp += list_[i].dsp_bandwidth
        i += 1
    return tmp


def MEM_total(list_):
    tmp = 0
    for out in list_:
        tmp += out.on_chip_mem
    return tmp


def find_max_band(list_1, list_2):
    tmpBD = 0
    index_j = 0

    for i in list_2:
        if list_1[i].dsp_bandwidth >= tmpBD:
            tmpBD = list_1[i].dsp_bandwidth

    for j in list_2:
        if tmpBD == list_1[j].dsp_bandwidth:
            return j


'''
512*1*1*512 13
512*1*1*512 14
10*1*1*512  15'''
conv_out_list[13].on_chip_mem = (layer_list[13].weight_number * layer_list[13].weight_w * layer_list[13].weight_h *
                                 layer_list[13].weight_c + conv_out_list[13].high * conv_out_list[13].width *
                                 conv_out_list[13].channel) * 16
conv_out_list[14].on_chip_mem = (layer_list[14].weight_number * layer_list[14].weight_w * layer_list[14].weight_h *
                                 layer_list[14].weight_c + conv_out_list[14].high * conv_out_list[14].width *
                                 conv_out_list[14].channel) * 16
conv_out_list[15].on_chip_mem = (layer_list[15].weight_number * layer_list[15].weight_w * layer_list[15].weight_h *
                                 layer_list[15].weight_c) * 16
conv_out_list[13].dsp_bandwidth = CTC * conv_out_list[13].on_chip_mem / 8
conv_out_list[14].dsp_bandwidth = CTC * conv_out_list[14].on_chip_mem / 8
conv_out_list[15].dsp_bandwidth = CTC * conv_out_list[15].on_chip_mem / 8
conv_out_list[13].col = conv_out_list[13].width - layer_list[13].weight_w + 1
conv_out_list[14].col = conv_out_list[14].width - layer_list[13].weight_w + 1
conv_out_list[15].col = conv_out_list[15].width - layer_list[13].weight_w + 1
FC_total_mem = conv_out_list[13].on_chip_mem + conv_out_list[14].on_chip_mem + conv_out_list[15].on_chip_mem
i = 0
CONV_total_mem = 0
layer_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
while i <= 12:
    if i == 0:
        conv_out_list[i].pre_ker = conv_out_list[i].dsp_num / input1.channel
        CONV_total_mem += input1.high * input1.channel * 4
        CONV_total_mem += layer_list[i].weight_h * layer_list[i].weight_w * layer_list[i].weight_c * conv_out_list[
            i].pre_ker
        conv_out_list[i].on_chip_mem = (input1.high * input1.channel * 4 + layer_list[i].weight_h * layer_list[
            i].weight_w * layer_list[i].weight_c * conv_out_list[i].pre_ker) * 16
        conv_out_list[i].dsp_bandwidth = conv_out_list[i].dsp_num * Single_DSP_BAND / (
                    conv_out_list[i].high * conv_out_list[i].col)
    else:
        conv_out_list[i].pre_ker = conv_out_list[i].dsp_num / layer_list[i].weight_c
        if conv_out_list[i].pre_ker <= 1:
            conv_out_list[i].pre_ker = 1
        CONV_total_mem += conv_out_list[i - 1].high * conv_out_list[i - 1].channel * 4
        CONV_total_mem += layer_list[i].weight_h * layer_list[i].weight_w * layer_list[i].weight_c * conv_out_list[
            i].pre_ker
        conv_out_list[i].on_chip_mem = (conv_out_list[i - 1].high * conv_out_list[i - 1].channel * 4 + layer_list[
            i].weight_h * layer_list[i].weight_w * layer_list[i].weight_c * conv_out_list[i].pre_ker) * 16
        conv_out_list[i].dsp_bandwidth = conv_out_list[i].dsp_num * Single_DSP_BAND / (
                    conv_out_list[i].high * conv_out_list[i].col)
    i += 1
CONV_total_mem = CONV_total_mem * 16
remain_band = TOTAL_DSP_BAND - conv_out_list[13].dsp_bandwidth - conv_out_list[14].dsp_bandwidth - conv_out_list[
    15].dsp_bandwidth

if FC_total_mem + CONV_total_mem < ON_CHIP_MEM:
    print("FC layers input feature map and kernels can be totally stored in on chip memory")
    print("remain memory %d" % (ON_CHIP_MEM - MEM_total(conv_out_list)))
    print("current band width %d" % (BW_conv_total(conv_out_list)))
    outfile.write("FC layers input feature map and kernels can be totally stored in on chip memory\n")
    outfile.write("remain memory %d\n" % (ON_CHIP_MEM - MEM_total(conv_out_list)))
    outfile.write("current band width %d\n" % (BW_conv_total(conv_out_list)))
    # start to allocate memory bandwidth
    while 1:
        if BW_conv_total(conv_out_list) <= remain_band:
            index = find_max_band(conv_out_list, layer_index_list)
            if conv_out_list[index].col + 2 < conv_out_list[index].width:
                col_tmp = conv_out_list[index].col + 1
                bw_tmp = conv_out_list[index].dsp_bandwidth * conv_out_list[index].col / col_tmp
                mem_tmp = (conv_out_list[index].high * conv_out_list[index].channel * (4 + col_tmp - 2) + layer_list[
                    index].weight_h * layer_list[index].weight_w * layer_list[index].weight_c) * 16
                if MEM_total(conv_out_list) - conv_out_list[index].on_chip_mem + mem_tmp <= ON_CHIP_MEM:
                    conv_out_list[index].on_chip_mem = mem_tmp
                    conv_out_list[index].dsp_bandwidth = bw_tmp
                    conv_out_list[index].col = col_tmp
                else:
                    break
            else:
                layer_index_list.remove(index)
                if len(layer_index_list) == 0:
                    break
        else:
            break
else:
    print("not enough space")
    outfile.write("not enough space\n")
    exit(0)

for out in conv_out_list:
    if out.layer_number >= 14:
        out.pre_ker = layer_list[conv_out_list.index(out)].weight_number
    if out.col == 1:
        out.col = min(out.col + 2, conv_out_list[conv_out_list.index(out) - 1].width)
    else:
        out.col = out.col + 2

    print("layer %d, on chip mem size %d, bandwidth %f, %d cols, %d kernels" % (
    out.layer_number, out.on_chip_mem, out.dsp_bandwidth, out.col, out.pre_ker))
    outfile.write("layer %d, on chip mem size %d, bandwidth %f, %d cols, %d kernels\n" % (
    out.layer_number, out.on_chip_mem, out.dsp_bandwidth, out.col, out.pre_ker))
print("remain on chip memory size(bit) %d" % (ON_CHIP_MEM - MEM_total(conv_out_list)))
print("remain DSP bandwidth %d" % (TOTAL_DSP_BAND - BW_total(conv_out_list)))
outfile.write("remain on chip memory size(bit) %d\n" % (ON_CHIP_MEM - MEM_total(conv_out_list)))
outfile.write("remain DSP bandwidth %d\n" % (TOTAL_DSP_BAND - BW_total(conv_out_list)))
outfile.close()

################################## Algorithm End ################################## Script Start ##################################

filepath = "./define_.h"
define_ = open(filepath, 'w')

### script for weight part ###
KERNELDim_list = []
OFM_channel_list=[]
print("//weight")
define_.write("//weight\n")
print("#define WEIGHT_PRECISION 16")
define_.write("#define WEIGHT_PRECISION 16\n")
i = 1
for weight in layer_list:
    print("#define KERNELDim_%d %d" % (i, weight.weight_w))
    define_.write("#define KERNELDim_%d %d\n" % (i, weight.weight_w))
    KERNELDim_list.append(weight.weight_w)
    print("#define OFM_Channels_%d %d" % (i, conv_out_list[i - 1].channel))
    define_.write("#define OFM_Channels_%d %d\n" % (i, conv_out_list[i - 1].channel))
    OFM_channel_list.append(conv_out_list[i - 1].channel)
    i = i + 1

### script for input feature map part ###
IFM_channel_list=[]
IFM_Dim_list = []
print("\n//input feature map")
define_.write("\n//input feature map\n")
print("#define INPUT_PRECISION 16")
i = 1
for feature_map in output_list:
    if i == 17:
        break
    if i == 1:
        print("#define IFMDim_%d %d" % (i, input1.high))
        define_.write("#define IFMDim_%d %d\n" % (i, input1.high))
        IFM_Dim_list.append(input1.high)
        print("#define IFM_Channels_%d %d" % (i, input1.channel))
        define_.write("#define IFM_Channels_%d %d\n" % (i, input1.channel))
        IFM_channel_list.append(input1.channel)
        print("#define STRIDE_%d %d" % (i, 1))
        define_.write("#define STRIDE_%d %d\n" % (i, 1))
        i = i + 1
    print("#define IFMDim_%d %d" % (i, feature_map.width))
    define_.write("#define IFMDim_%d %d\n" % (i, feature_map.width))
    IFM_Dim_list.append(feature_map.width)
    print("#define IFM_Channels_%d %d" % (i, feature_map.channel))
    define_.write("#define IFM_Channels_%d %d\n" % (i, feature_map.channel))
    IFM_channel_list.append(feature_map.channel)
    print("#define STRIDE_%d %d" % (i, 1))
    define_.write("#define STRIDE_%d %d\n" % (i, 1))
    i = i + 1

### script for output feature map part ###
OFM_Dim_list = []
print("\n//output after conv, ignore pooling")
define_.write("\n//output after conv, ignore pooling\n")
print("//OFMDim=(IFMDim+2-KERNELDim)/STRIDE + 1, if padding")
define_.write("//OFMDim=(IFMDim+2-KERNELDim)/STRIDE + 1, if padding\n")
print("#define ACTIVATION_PRECISION 16")  ###############
define_.write("#define ACTIVATION_PRECISION 16\n")

i = 1
while i <= 16:
    if i<=13:
        print("#define OFMDim_%d %d" % (i, (IFM_Dim_list[i-1] + 2 - KERNELDim_list[i-1])/layer_list[i-1].stride + 1))
        define_.write("#define OFMDim_%d %d\n" % (i, (IFM_Dim_list[i-1] + 2 - KERNELDim_list[i-1])/layer_list[i-1].stride + 1))
        OFM_Dim_list.append((IFM_Dim_list[i-1] + 2 - KERNELDim_list[i-1])/layer_list[i-1].stride + 1)
    else:
        print("#define OFMDim_%d %d" % (i, (IFM_Dim_list[i - 1] - KERNELDim_list[i - 1]) / layer_list[i - 1].stride + 1))
        define_.write("#define OFMDim_%d %d\n" % (i, (IFM_Dim_list[i - 1] - KERNELDim_list[i - 1]) / layer_list[i - 1].stride + 1))
        OFM_Dim_list.append((IFM_Dim_list[i - 1] - KERNELDim_list[i - 1]) / layer_list[i - 1].stride + 1)
    i = i + 1


### script for PE array allocation part ###
pe_list=[]
simd_list=[]

def pe_all(layer_num, ch_num, ker_num, dsp_num):
    if ch_num * ker_num == dsp_num:
        print("#define SIMD_%d %d" % (layer_num, ch_num))
        define_.write("#define SIMD_%d %d\n" % (layer_num, ch_num))
        simd_list.append(ch_num)
        print("#define PE_%d %d" % (layer_num, ker_num))
        define_.write("#define PE_%d %d\n" % (layer_num, ker_num))
        pe_list.append(ker_num)
    elif dsp_num < ch_num:
        print("#define SIMD_%d %d" % (layer_num, dsp_num))
        define_.write("#define SIMD_%d %d\n" % (layer_num, dsp_num))
        simd_list.append(dsp_num)
        print("#define PE_%d %d" % (layer_num, 1))
        define_.write("#define PE_%d %d\n" % (layer_num, 1))
        pe_list.append(1)


def pe_array(layer_num, layer_list_, conv_out_list_):
    channel_num = layer_list_[layer_num - 1].weight_c
    kernel_num = conv_out_list_[layer_num - 1].pre_ker
    dsp_num = conv_out_list_[layer_num - 1].dsp_num
    pe_all(layer_num, channel_num, kernel_num, dsp_num)


print("\n//PE array allocation")
define_.write("\n//PE array allocation\n")
i = 1
while i <= 16:
    pe_array(i, layer_list, conv_out_list)
    i = i + 1

print("\n//pre-store Input feature map for each layer(#column)")
define_.write("\n//pre-store Input feature map for each layer(#column)\n")
i = 1
while i <= 16:
    print("#define IFM_%d_precol %d" % (i, conv_out_list[i-1].col))
    define_.write("#define IFM_%d_precol %d\n" % (i, conv_out_list[i-1].col))
    i = i + 1

pre_kernel_list = []
print("\n//pre-store kernel number for each layer")
define_.write("\n//pre-store kernel number for each layer\n")
i = 1
while i <= 16:
    print("#define Kernel_%d_pre %d" % (i, conv_out_list[i-1].pre_ker))
    define_.write("#define Kernel_%d_pre %d\n" % (i, conv_out_list[i-1].pre_ker))
    pre_kernel_list.append(conv_out_list[i-1].pre_ker)
    i = i + 1

#SIMD threshold
print("\n//SIMD Threshold")
define_.write("\n//SIMD Threshold")
print("#define SIMD_W_TH 512/(WEIGHT_PRECISION)\t//simd threshold for weight(maximum simd each 512 bits)")
define_.write("\n#define SIMD_W_TH 512/(WEIGHT_PRECISION)\t//simd threshold for weight(maximum simd each 512 bits)")
print("#define SIMD_I_TH 512/(ACTIVATION_PRECISION)\t//simd threshold for IFM(maximum simd each 512 bits)")
define_.write("\n#define SIMD_I_TH 512/(ACTIVATION_PRECISION)\t//simd threshold for IFM(maximum simd each 512 bits)\n")


#SIMD Occupied rows
print("\n//SIMD Occupied Rows")
define_.write("\n////SIMD Occupied Rows")
print("//L_I_SIMD_R=SIMD/SIMD_I_TH\n//L_W_SIMD_R=SIMD/SIMD_W_TH")
define_.write("\n//L_I_SIMD_R=SIMD/SIMD_I_TH\n//L_W_SIMD_R=SIMD/SIMD_W_TH")
i = 1
while i <= 16:
    if (simd_list[i-1]/(512/WEIGHT_WIDTH))<1:
        a=1
    else:
        a=simd_list[i-1]/(512/WEIGHT_WIDTH)

    if (simd_list[i-1] / (512 / DATA_WIDTH))<1:
        b=1
    else:
        b=simd_list[i-1]/(512/WEIGHT_WIDTH)
    print("#define L%d_W_SIMD_R %d" % (i,a))
    define_.write("#define L%d_W_SIMD_R %d\n" % (i,a))
    print("#define L%d_I_SIMD_R %d" % (i, b))
    define_.write("#define L%d_I_SIMD_R %d\n" % (i, b))
    i = i + 1


# simd loop times
print("\n//simd__loop_times=IFM_Channels_/SIMD_")
define_.write("\n//simd__loop_times=IFM_Channels_/SIMD_")
i = 1
while i <= 16:
    if(IFM_channel_list[i-1]/simd_list[i-1]<1):
        a=1
    else:
        a=IFM_channel_list[i-1]/simd_list[i-1]
    print("#define simd_%d_loop_times %d"%(i,a))
    define_.write("\n#define simd_%d_loop_times %d"%(i,a))
    i = i + 1

# layer i filter slide number & layer i left pe number
print("\n//layer_i_filter_slide_number && layer_i_left_pe")
define_.write("\n\n//layer_i_filter_slide_number && layer_i_left_pe\n")
i = 1
while i<=16:
    tmp = math.ceil(OFM_channel_list[i-1]/pe_list[i-1])
    print("#define layer%d_filter_slide_number %d" %(i,tmp))
    define_.write("#define layer%d_filter_slide_number %d\n" %(i,tmp))
    if(tmp>OFM_channel_list[i-1]/pe_list[i-1]):
        print("#define layer%d_left_pe %d" %(i,OFM_channel_list[i-1]-(tmp-1)*pe_list[i-1]))
        define_.write("#define layer%d_left_pe %d\n" %(i,OFM_channel_list[i-1]-(tmp-1)*pe_list[i-1]))
    elif(tmp==OFM_channel_list[i-1]/pe_list[i-1]):
        print("#define layer%d_left_pe %d" %(i,pe_list[i-1]))
        define_.write("#define layer%d_left_pe %d\n" %(i,pe_list[i-1]))
    else:
        print("#define layer%d_left_pe %d" % (i, OFM_channel_list[i-1]-tmp*pe_list[i-1]))
        define_.write("#define layer%d_left_pe %d\n" % (i, OFM_channel_list[i-1]-tmp*pe_list[i-1]))
    i = i + 1

# layer i OFM_Cha_Occupy_Row
print("\n//OFM_Cha_Occupy_Row=OFM_Channel/SIMD_I_TH")
define_.write("\n//OFM_Cha_Occupy_Row=OFM_Channel/SIMD_I_TH\n")
i = 1
while i<=16 :
    print("#define OFM_%d_Cha_Occupy_Row %d" % (i, math.ceil(OFM_channel_list[i-1]/(512/WEIGHT_WIDTH))))
    define_.write("#define OFM_%d_Cha_Occupy_Row %d\n" % (i, math.ceil(OFM_channel_list[i-1]/(512/WEIGHT_WIDTH))))
    i = i + 1


# layer i weight start and end index
print("\n//weight start and end index")
print("//start = previous_end")
print("//end = IFM_Channel * KERNELDim * KERNELDim * OFM_Channel * OFMDim * OFMDim( (/SIMD) or (*WEIGHT_PRECISION/512) ) + current_start")
define_.write("\n//weight start and end index\n")
define_.write("//start = previous_end\n")
define_.write("//end = IFM_Channel * KERNELDim * KERNELDim * OFM_Channel * OFMDim * OFMDim( (/SIMD) or (*WEIGHT_PRECISION/512) ) + current_start\n")
i = 1
while i <= 16:
    if i == 1:
        print("#define w1_start 0")
        define_.write("#define w1_start 0\n")
        if(simd_list[i-1]*WEIGHT_WIDTH > 512):
            print("#define w1_end %d" % (IFM_channel_list[i-1]*KERNELDim_list[i-1]*KERNELDim_list[i-1]*OFM_channel_list[i-1]*OFM_Dim_list[i-1]*OFM_Dim_list[i-1]*WEIGHT_WIDTH/512))
            define_.write("#define w1_end %d\n" % (IFM_channel_list[i-1]*KERNELDim_list[i-1]*KERNELDim_list[i-1]*OFM_channel_list[i-1]*OFM_Dim_list[i-1]*OFM_Dim_list[i-1]*WEIGHT_WIDTH/512))
        else:
            print("#define w1_end %d" % (IFM_channel_list[i-1]*KERNELDim_list[i-1]*KERNELDim_list[i-1]*OFM_channel_list[i-1]*OFM_Dim_list[i-1]*OFM_Dim_list[i-1]/simd_list[i-1]))
            define_.write("#define w1_end %d\n" % (IFM_channel_list[i-1]*KERNELDim_list[i-1]*KERNELDim_list[i-1]*OFM_channel_list[i-1]*OFM_Dim_list[i-1]*OFM_Dim_list[i-1]/simd_list[i-1]))
    else:
        print("#define w%d_start w%d_end" % (i, i-1))
        define_.write("#define w%d_start w%d_end\n" % (i, i-1))
        if (simd_list[i - 1] * WEIGHT_WIDTH > 512):
            print("#define w%d_end (%d + w%d_start)" % (i, IFM_channel_list[i - 1] * KERNELDim_list[i - 1] * KERNELDim_list[i - 1] * OFM_channel_list[i - 1] * OFM_Dim_list[i - 1] * OFM_Dim_list[i - 1] * WEIGHT_WIDTH / 512,i))
            define_.write("#define w%d_end (%d + w%d_start)\n" % (i, IFM_channel_list[i - 1] * KERNELDim_list[i - 1] * KERNELDim_list[i - 1] * OFM_channel_list[i - 1] * OFM_Dim_list[i - 1] * OFM_Dim_list[i - 1] * WEIGHT_WIDTH / 512,i))
        else:
            print("#define w%d_end (%d + w%d_start)" % (i, IFM_channel_list[i - 1] * KERNELDim_list[i - 1] * KERNELDim_list[i - 1] * OFM_channel_list[i - 1] * OFM_Dim_list[i - 1] * OFM_Dim_list[i - 1] / simd_list[i - 1], i))
            define_.write("#define w%d_end (%d + w%d_start)\n" % (i, IFM_channel_list[i - 1] * KERNELDim_list[i - 1] * KERNELDim_list[i - 1] * OFM_channel_list[i - 1] * OFM_Dim_list[i - 1] * OFM_Dim_list[i - 1] / simd_list[i - 1], i))
    i = i + 1


#layer i w_stream_depth
print("\n//weight_stream_depth=kernel_pre*IFM_Channel*KERNELDim*KERNELDim/SIMD")
define_.write("\n//weight_stream_depth=kernel_pre*IFM_Channel*KERNELDim*KERNELDim/SIMD\n")
i = 1
while i<=16:
    if(simd_list[i-1] > 512/WEIGHT_WIDTH):
        print("int w%d_stream_depth=%d;" % (i,pre_kernel_list[i-1]*IFM_channel_list[i-1]*KERNELDim_list[i-1]*KERNELDim_list[i-1]*WEIGHT_WIDTH/512))
        define_.write("int w%d_stream_depth=%d;\n" % (i,pre_kernel_list[i-1]*IFM_channel_list[i-1]*KERNELDim_list[i-1]*KERNELDim_list[i-1]*WEIGHT_WIDTH/512))
    else:
        print("int w%d_stream_depth=%d;" % (i,pre_kernel_list[i-1]*IFM_channel_list[i-1]*KERNELDim_list[i-1]*KERNELDim_list[i-1]/simd_list[i-1]))
        define_.write("int w%d_stream_depth=%d;\n" % (i,pre_kernel_list[i-1]*IFM_channel_list[i-1]*KERNELDim_list[i-1]*KERNELDim_list[i-1]/simd_list[i-1]))
    i = i + 1