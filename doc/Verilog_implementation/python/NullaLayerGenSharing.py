# ============================================================================#
#                Adding layers as synthesized NullaNet layer                 #
#                                                                            #
#                      Arash Fayyazi and Massoud Pedram                      #
#     SPORT Lab, University of Southern California, Los Angeles, CA 90089    #
#                          http://sportlab.usc.edu/                          #
#                                                                            #
# For licensing, please refer to the LICENSE file in the main directory.     #
#                                                                            #
#                                                                            #
#                                                                            #
# ============================================================================#
# !/usr/bin/python
import os, argparse, timeit


############## class definition ############
class IOPORT(object):

    def __init__(self, NAME, TYPE):
        self.SIZE = 1
        self.NAME = NAME
        self.TYPE = TYPE


def parseCommandLines():
    global CircuitName, SpecFile, header
    header = True
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--header", action="store_true", help="Don't Show the header")
    parser.add_argument("Layer_Specification", help="the input Layer Specification file")
    args = parser.parse_args()
    if args.header: header = args.header
    SpecFile = args.Layer_Specification
    CircuitName = os.path.splitext(os.path.basename(args.Layer_Specification))[0]
    current_path = os.getcwd()


##########################################################
### run it
##########################################################
def Layer_generation():
    VerilogFile = CircuitName + "_Nulla.v"

    infile = open(SpecFile, 'r')
    # Parsing the input Spec file
    kernelNames = infile.readline().rstrip('\n')
    kernelNamesList = kernelNames.split()
    specList = [list(map(int, line.strip().split(' '))) for line in infile]
    # spec : each line, 0 -->i_w, 1 --> i_h, 2 --> i_c, 3 --> o_w, 4 --> o_h, 5 --> o_c, 6 --> w_w, 7 --> w_h,
    # 8 --> p_w , 9 --> p_h, 10 --> Conv or FC, 11 --> Pooling or Not, 
    inName = 'in'
    outName = 'out'
    sharing = True
    numRows = 2

    infile.close()

    # writing the output file
    outfile = open(VerilogFile, 'w')


    # add top module
    '''
    outfile.write("`timescale 1ns/1ps\n")
    # writing Module def
    outfile.write("module Top(%s, %s, clk);\n" % (inName, outName))
    if specList[0][10] == 1:
        inSize = (specList[0][0] * specList[0][1] * specList[0][2]) / 2 - 1
    else:
        inSize = specList[0][0] / 2 - 1

    if specList[-1][10] == 1:
        outSize = specList[-1][3] * specList[-1][4] * specList[-1][5] - 1
    else:
        outSize = specList[-1][3] - 1
    outfile.write("\tinput [%d : 0] %s;\n" % (inSize, inName))
    outfile.write("\tinput clk;\n")
    outfile.write("\toutput [%d : 0] %s;\n" % (outSize, outName))
    outfile.write("\n")

    # Wires & Regs
    if specList[0][10] == 1:
        memSize = (specList[0][0] * specList[0][1] * specList[0][2]) / 2 - 1
    else:
        memSize = specList[0][0] / 2 - 1
    nullaInSize = (memSize + 1) * 2 - 1
    outfile.write("\twire [%d : 0] reg_input_d;\n" % memSize)
    outfile.write("\treg [%d : 0] reg_input_q;\n" % memSize)
    outfile.write("\twire [%d : 0] nullaIn;\n" % nullaInSize)
    outfile.write("\n")

    # Instantiate NullaLayer Module
    outfile.write("\tnullaLayers nulla1 (.%s(nullaIn), .%s(%s), .clk(clk));\n" % (inName, outName, outName))
    # Registers
    outfile.write("\talways @(posedge clk)\n")
    outfile.write("\tbegin\n")
    outfile.write("\t\treg_input_q <= reg_input_d;\n")
    outfile.write("\tend\n")
    outfile.write("\n")

    # Write the connections and assignments
    outfile.write("\tassign nullaIn = {reg_input_q, reg_input_d}; \n")
    outfile.write("\tassign reg_input_d = %s; \n" % inName)
    outfile.write("\n")

    outfile.write("endmodule\n")
    outfile.write("\n")
    outfile.write("\n")
    '''
    # writing Module def
    outfile.write("module nullaLayers(%s, %s, clk);\n" % (inName, outName))
    '''
    # define input/output one by one (not in array style)
    for idx in range(specList[0][0] * specList[0][1] * specList[0][2]):
        outfile.write(" %s_%d," % (inName, idx))
    for idx in range(specList[-1][0] * specList[-1][1] * specList[-1][2] - 1):
        outfile.write(" %s_%d," % (outName, idx))
    outfile.write(" %s_%d); \n" % (outName, specList[-1][0] * specList[-1][1] * specList[-1][2] - 1))
    '''

    # writing the input/output
    if specList[0][10] == 1:
        inSize = specList[0][0] * (specList[0][7]+1) * specList[0][2] - 1
    else:
        inSize = specList[0][0] - 1

    if specList[-1][10] == 1:
        outSize = specList[-1][3] * specList[-1][4] * specList[-1][5] - 1
    else:
        outSize = specList[-1][3] - 1
    outfile.write("\tinput [%d : 0] %s;\n" % (inSize, inName))
    outfile.write("\tinput clk;\n")
    outfile.write("\toutput [%d : 0] %s;\n" % (outSize, outName))
    '''
    # define input/output one by one (not in array style)
    outfile.write("input")
    for idx in range(specList[0][0] * specList[0][1] * specList[0][2] - 1):
        outfile.write(" %s_%d," % (inName, idx))
    outfile.write(" %s_%d; \n" % (inName, specList[0][0] * specList[0][1] * specList[0][2] - 1))
    outfile.write("output")
    for idx in range(specList[-1][0] * specList[-1][1] * specList[-1][2] - 1):
       outfile.write(" %s_%d," % (outName, idx))
    outfile.write(" %s_%d; \n" % (outName, specList[-1][0] * specList[-1][1] * specList[-1][2] - 1))
    '''
    outfile.write("\n")

    # Wires & Regs
    for k in range(len(specList) + 1):
        # input Regs
        if k == 0:
            if specList[k][10] == 1:
                memSize = specList[k][0] * (specList[k][7]+1) * specList[k][2] - 1
            else:
                memSize = specList[k][0] - 1
            actSize = memSize
        # Other Regs
        else:
            if specList[k-1][10] == 1:
                if sharing:
                    memSize = specList[k - 1][3] * 1 * specList[k - 1][5] - 1
                else:
                    memSize = specList[k - 1][3] * specList[k - 1][4] * specList[k - 1][5] - 1
            else:
                memSize = specList[k - 1][3] - 1
            # if there is a pooling Layer
            if specList[k-1][11] == 1:
                if sharing:
                    actSize = specList[k - 1][9] * specList[k - 1][3] * specList[k - 1][8] * \
                              specList[k - 1][5] - 1
                else:
                    actSize = specList[k - 1][4] * specList[k - 1][9] * specList[k - 1][3] * specList[k - 1][8] * \
                              specList[k - 1][5] - 1
            else:
                actSize = memSize
        if specList[k - 1][10] == 1 and sharing:
            for timeIdx in range(int(specList[k - 1][4] * specList[k - 1][9] / numRows)):
                if timeIdx == 0:
                    outfile.write("\treg [%d : 0] reg%d_%d_mem_read_data;\n" % (memSize, k, timeIdx))
                    outfile.write("\twire [%d : 0] reg%d_%d_mem_write_data;\n" % (memSize, k, timeIdx))
                    outfile.write("\twire [%d : 0] reg%d_%d_write_data;\n" % (actSize, k, timeIdx))
                    if specList[k - 1][10] == 1 and specList[k][10] == 0:
                        outfile.write("\twire [%d : 0] reg%d_%d_write_flatten_data;\n" % (memSize, k, timeIdx))
                else:
                    outfile.write("\treg [%d : 0] reg%d_%d_mem_read_data;\n" % (memSize, k, timeIdx))
                    outfile.write("\twire [%d : 0] reg%d_%d_mem_write_data;\n" % (memSize, k, timeIdx))
        else:
            outfile.write("\treg [%d : 0] reg%d_mem_read_data;\n" % (memSize, k))
            outfile.write("\twire [%d : 0] reg%d_mem_write_data;\n" % (memSize, k))
            outfile.write("\twire [%d : 0] reg%d_write_data;\n" % (actSize, k))
            if specList[k-1][10] == 1 and specList[k][10] == 0:
                outfile.write("\twire [%d : 0] reg%d_write_flatten_data;\n" % (memSize, k))
    outfile.write("\n")

    # Instantiate the kernels
    memIdx = 0
    for k in range(len(specList)):
        # if it is ConvLayer
        if specList[k][10] == 1:
            if sharing:
                # if it has pooling layer
                if specList[k][11] == 1:
                    hActSize = specList[k][9]
                    wActSize = specList[k][3] * specList[k][8]
                else:
                    hActSize = specList[k][4]
                    wActSize = specList[k][3]
                for hFirst in range(hActSize):
                    for wFirst in range(wActSize):
                        outfile.write(
                            "\t%s %s_%d(" % (kernelNamesList[k], kernelNamesList[k], wFirst + hFirst * wActSize))
                        for d in range(specList[k][2]):
                            for h in range(hFirst, hFirst + specList[k][7]):
                                for w in range(wFirst, wFirst + specList[k][6]):
                                    memIdx = (h * specList[k][0] + w) * specList[k][2] + d
                                    outfile.write(" reg%d_mem_read_data[%d]," % (k, memIdx))
                        for c in range(specList[k][5] - 1):
                            memIdx = (wFirst + hFirst * wActSize) * specList[k][5] + c
                            outfile.write(" reg%d_0_write_data[%d]," % (k + 1, memIdx))
                        outfile.write(" reg%d_0_write_data[%d]);\n" % (k + 1, memIdx + 1))
            else:
                # if it has pooling layer
                if specList[k][11] == 1:
                    hActSize = specList[k][4] * specList[k][9]
                    wActSize = specList[k][3] * specList[k][8]
                else:
                    hActSize = specList[k][4]
                    wActSize = specList[k][3]
                for hFirst in range(hActSize):
                    for wFirst in range(wActSize):
                        outfile.write("\t%s %s_%d(" % (kernelNamesList[k], kernelNamesList[k], wFirst + hFirst * wActSize))
                        for d in range(specList[k][2]):
                            for h in range(hFirst, hFirst + specList[k][7]):
                                for w in range(wFirst, wFirst + specList[k][6]):
                                    memIdx = (h * specList[k][0] + w) * specList[k][2] + d
                                    outfile.write(" reg%d_mem_read_data[%d]," % (k, memIdx))
                        for c in range(specList[k][5] - 1):
                            memIdx = (wFirst + hFirst * wActSize) * specList[k][5] + c
                            outfile.write(" reg%d_write_data[%d]," % (k + 1, memIdx))
                        outfile.write(" reg%d_write_data[%d]);\n" % (k + 1, memIdx + 1))
        # if it is FCLayer
        else:
            if sharing and specList[k-1][10] == 1:
                outfile.write("\t%s %s_0(" % (kernelNamesList[k], kernelNamesList[k]))
                for timeIdx in range(int(specList[k - 1][4] * 2 / numRows)):
                    for fcIn in range(int(specList[k][0] / (specList[k - 1][4] * 2 / numRows))):
                        memIdx = fcIn
                        outfile.write(" reg%d_%d_mem_read_data[%d]," % (k, timeIdx, memIdx))
                for fcOut in range(specList[k][3] - 1):
                    memIdx = fcOut
                    outfile.write(" reg%d_write_data[%d]," % (k + 1, memIdx))
                outfile.write(" reg%d_write_data[%d]);\n" % (k + 1, memIdx + 1))
            else:
                outfile.write("\t%s %s_0(" % (kernelNamesList[k], kernelNamesList[k]))
                for fcIn in range(specList[k][0]):
                    memIdx = fcIn
                    outfile.write(" reg%d_mem_read_data[%d]," % (k, memIdx))
                for fcOut in range(specList[k][3]-1):
                    memIdx = fcOut
                    outfile.write(" reg%d_write_data[%d]," % (k + 1, memIdx))
                outfile.write(" reg%d_write_data[%d]);\n" % (k + 1, memIdx + 1))
    outfile.write("\n")

    # Registers
    outfile.write("\talways @(posedge clk)\n")
    outfile.write("\tbegin\n")
    for k in range(len(specList)+1):
        if specList[k - 1][10] == 1 and sharing:
            for timeIdx in range(int(specList[k - 1][4] * 2 / numRows)):
                outfile.write("\t\treg%d_%d_mem_read_data <= reg%d_%d_mem_write_data;\n" % (k, timeIdx, k, timeIdx))
        else:
            outfile.write("\t\treg%d_mem_read_data <= reg%d_mem_write_data;\n" % (k, k))

    outfile.write("\tend\n")
    outfile.write("\n")

    # Write the connections and assignments
    outfile.write("\tassign reg0_mem_write_data = %s; \n" % inName)
    outfile.write("\tassign %s = reg%d_mem_read_data; \n" % (outName, len(specList)))
    for k in range(len(specList)):
        if specList[k][11] == 1:
            if sharing:
                for hFirst in range(0, numRows, specList[k][9]):
                    for wFirst in range(0, specList[k][3] * specList[k][8], specList[k][8]):
                        for c in range(specList[k][5]):
                            if specList[k][10] == 1 and specList[k + 1][10] == 0:
                                outfile.write("\tassign reg%d_0_write_flatten_data[%d] ="
                                              % (k + 1, c + (
                                        wFirst / specList[k][8] + (hFirst / specList[k][9]) * specList[k][3]) *
                                                 specList[k][5]))
                            else:
                                outfile.write("\tassign reg%d_0_mem_write_data[%d] ="
                                              % (k + 1, c + (
                                        wFirst / specList[k][8] + (hFirst / specList[k][9]) * specList[k][3]) *
                                                 specList[k][5]))
                            for h in range(hFirst, hFirst + specList[k][9]):
                                for w in range(wFirst, wFirst + specList[k][8]):
                                    memIdx = (h * specList[k][3] * specList[k][8] + w) * specList[k][5] + c
                                    if (h == hFirst + specList[k][9] - 1) and (w == wFirst + specList[k][8] - 1):
                                        outfile.write(" reg%d_0_write_data[%d];\n" % (k + 1, memIdx))
                                    else:
                                        outfile.write(" reg%d_0_write_data[%d] |" % (k + 1, memIdx))
            else:
                for hFirst in range(0, specList[k][4] * specList[k][9], specList[k][9]):
                    for wFirst in range(0, specList[k][3] * specList[k][8], specList[k][8]):
                        for c in range(specList[k][5]):
                            if specList[k][10] == 1 and specList[k + 1][10] == 0:
                                outfile.write("\tassign reg%d_write_flatten_data[%d] ="
                                              % (k + 1, c + (
                                        wFirst / specList[k][8] + (hFirst / specList[k][9]) * specList[k][3]) *
                                                 specList[k][5]))
                            else:
                                outfile.write("\tassign reg%d_mem_write_data[%d] ="
                                              % (k + 1, c + (
                                        wFirst / specList[k][8] + (hFirst / specList[k][9]) * specList[k][3]) *
                                                 specList[k][5]))
                            for h in range(hFirst, hFirst + specList[k][9]):
                                for w in range(wFirst, wFirst + specList[k][8]):
                                    memIdx = (h * specList[k][3] * specList[k][8] + w) * specList[k][5] + c
                                    if (h == hFirst + specList[k][9] - 1) and (w == wFirst + specList[k][8] - 1):
                                        outfile.write(" reg%d_write_data[%d];\n" % (k + 1, memIdx))
                                    else:
                                        outfile.write(" reg%d_write_data[%d] |" % (k + 1, memIdx))
            if specList[k][10] == 1 and specList[k+1][10] == 0:
                idx = 0
                if sharing:
                    for timeIdx in range(0, int(specList[k][4] * 2 / numRows)):
                        if timeIdx == 0:
                            for d in range(specList[k][5]):
                                for h in range(0, 1):
                                    for w in range(0, specList[k][3]):
                                        memIdx = (h * specList[k][3] + w) * specList[k][5] + d
                                        outfile.write(
                                            "\tassign reg%d_%d_mem_write_data[%d] = reg%d_%d_write_flatten_data[%d];\n"
                                            % (k + 1,timeIdx, idx, k + 1, timeIdx, memIdx))
                                        idx += 1
                        else:
                            outfile.write("\tassign reg%d_%d_mem_write_data = reg%d_%d_mem_read_data;\n"
                                          % (k + 1, timeIdx, k + 1, timeIdx - 1))

                else:
                    for d in range(specList[k][5]):
                        for h in range(0, specList[k][4]):
                            for w in range(0, specList[k][3]):
                                memIdx = (h * specList[k][3] + w) * specList[k][5] + d
                                outfile.write("\tassign reg%d_mem_write_data[%d] = reg%d_write_flatten_data[%d];\n"
                                              % (k+1, idx, k+1, memIdx))
                                idx += 1

        else:
            outfile.write("\tassign reg%d_mem_write_data = reg%d_write_data; \n" % (k + 1, k + 1))

    outfile.write("endmodule\n")

    outfile.close()


##########################################################
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


################################################################################
start_time = timeit.default_timer()
parseCommandLines()
if not header:
    print("      +------------------------------------------------------------------+")
    print("      |                          Nulla Layer Gen 2.0                     |")
    print("      |                                                                  |")
    print("      | Copyright (C) 2019, SPORT Lab, University of Southern California |")
    print("      +------------------------------------------------------------------+\n")

Layer_generation()
stop_time = timeit.default_timer()
if not header:
    print("--------------------------------------------------------------------------------")
    print("Generating '" + CircuitName + "' finished (Runtime: %s)" % (hms_string(stop_time - start_time)))
    print("Generated Nulla Layer is " + CircuitName + "_Nulla.v")
