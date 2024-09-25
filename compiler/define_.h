//weight
#define WEIGHT_PRECISION 16
#define KERNELDim_1 3
#define OFM_Channels_1 64
#define KERNELDim_2 3
#define OFM_Channels_2 64
#define KERNELDim_3 3
#define OFM_Channels_3 128
#define KERNELDim_4 3
#define OFM_Channels_4 128
#define KERNELDim_5 3
#define OFM_Channels_5 256
#define KERNELDim_6 3
#define OFM_Channels_6 256
#define KERNELDim_7 3
#define OFM_Channels_7 256
#define KERNELDim_8 3
#define OFM_Channels_8 512
#define KERNELDim_9 3
#define OFM_Channels_9 512
#define KERNELDim_10 3
#define OFM_Channels_10 512
#define KERNELDim_11 3
#define OFM_Channels_11 512
#define KERNELDim_12 3
#define OFM_Channels_12 512
#define KERNELDim_13 3
#define OFM_Channels_13 512
#define KERNELDim_14 1
#define OFM_Channels_14 512
#define KERNELDim_15 1
#define OFM_Channels_15 512
#define KERNELDim_16 1
#define OFM_Channels_16 10

//input feature map
#define IFMDim_1 32
#define IFM_Channels_1 3
#define STRIDE_1 1
#define IFMDim_2 32
#define IFM_Channels_2 64
#define STRIDE_2 1
#define IFMDim_3 16
#define IFM_Channels_3 64
#define STRIDE_3 1
#define IFMDim_4 16
#define IFM_Channels_4 128
#define STRIDE_4 1
#define IFMDim_5 8
#define IFM_Channels_5 128
#define STRIDE_5 1
#define IFMDim_6 8
#define IFM_Channels_6 256
#define STRIDE_6 1
#define IFMDim_7 8
#define IFM_Channels_7 256
#define STRIDE_7 1
#define IFMDim_8 4
#define IFM_Channels_8 256
#define STRIDE_8 1
#define IFMDim_9 4
#define IFM_Channels_9 512
#define STRIDE_9 1
#define IFMDim_10 4
#define IFM_Channels_10 512
#define STRIDE_10 1
#define IFMDim_11 2
#define IFM_Channels_11 512
#define STRIDE_11 1
#define IFMDim_12 2
#define IFM_Channels_12 512
#define STRIDE_12 1
#define IFMDim_13 2
#define IFM_Channels_13 512
#define STRIDE_13 1
#define IFMDim_14 1
#define IFM_Channels_14 512
#define STRIDE_14 1
#define IFMDim_15 1
#define IFM_Channels_15 512
#define STRIDE_15 1
#define IFMDim_16 1
#define IFM_Channels_16 512
#define STRIDE_16 1

//output after conv, ignore pooling
//OFMDim=(IFMDim+2-KERNELDim)/STRIDE + 1, if padding
#define ACTIVATION_PRECISION 16
#define OFMDim_1 32
#define OFMDim_2 32
#define OFMDim_3 16
#define OFMDim_4 16
#define OFMDim_5 8
#define OFMDim_6 8
#define OFMDim_7 8
#define OFMDim_8 4
#define OFMDim_9 4
#define OFMDim_10 4
#define OFMDim_11 2
#define OFMDim_12 2
#define OFMDim_13 2
#define OFMDim_14 1
#define OFMDim_15 1
#define OFMDim_16 1

//PE array allocation
#define SIMD_1 3
#define PE_1 10
#define SIMD_2 64
#define PE_2 16
#define SIMD_3 64
#define PE_3 8
#define SIMD_4 128
#define PE_4 8
#define SIMD_5 128
#define PE_5 4
#define SIMD_6 256
#define PE_6 4
#define SIMD_7 256
#define PE_7 4
#define SIMD_8 256
#define PE_8 1
#define SIMD_9 512
#define PE_9 1
#define SIMD_10 512
#define PE_10 1
#define SIMD_11 128
#define PE_11 1
#define SIMD_12 128
#define PE_12 1
#define SIMD_13 128
#define PE_13 1
#define SIMD_14 8
#define PE_14 1
#define SIMD_15 8
#define PE_15 1
#define SIMD_16 8
#define PE_16 1

//pre-store Input feature map for each layer(#column)
#define IFM_1_precol 32
#define IFM_2_precol 32
#define IFM_3_precol 16
#define IFM_4_precol 16
#define IFM_5_precol 8
#define IFM_6_precol 8
#define IFM_7_precol 8
#define IFM_8_precol 4
#define IFM_9_precol 4
#define IFM_10_precol 4
#define IFM_11_precol 3
#define IFM_12_precol 2
#define IFM_13_precol 2
#define IFM_14_precol 2
#define IFM_15_precol 1
#define IFM_16_precol 1

//pre-store kernel number for each layer
#define Kernel_1_pre 10
#define Kernel_2_pre 16
#define Kernel_3_pre 8
#define Kernel_4_pre 8
#define Kernel_5_pre 4
#define Kernel_6_pre 4
#define Kernel_7_pre 4
#define Kernel_8_pre 1
#define Kernel_9_pre 1
#define Kernel_10_pre 1
#define Kernel_11_pre 1
#define Kernel_12_pre 1
#define Kernel_13_pre 1
#define Kernel_14_pre 512
#define Kernel_15_pre 512
#define Kernel_16_pre 10

//SIMD Threshold
#define SIMD_W_TH 512/(WEIGHT_PRECISION)	//simd threshold for weight(maximum simd each 512 bits)
#define SIMD_I_TH 512/(ACTIVATION_PRECISION)	//simd threshold for IFM(maximum simd each 512 bits)

////SIMD Occupied Rows
//L_I_SIMD_R=SIMD/SIMD_I_TH
//L_W_SIMD_R=SIMD/SIMD_W_TH#define L1_W_SIMD_R 1
#define L1_I_SIMD_R 1
#define L2_W_SIMD_R 2
#define L2_I_SIMD_R 2
#define L3_W_SIMD_R 2
#define L3_I_SIMD_R 2
#define L4_W_SIMD_R 4
#define L4_I_SIMD_R 4
#define L5_W_SIMD_R 4
#define L5_I_SIMD_R 4
#define L6_W_SIMD_R 8
#define L6_I_SIMD_R 8
#define L7_W_SIMD_R 8
#define L7_I_SIMD_R 8
#define L8_W_SIMD_R 8
#define L8_I_SIMD_R 8
#define L9_W_SIMD_R 16
#define L9_I_SIMD_R 16
#define L10_W_SIMD_R 16
#define L10_I_SIMD_R 16
#define L11_W_SIMD_R 4
#define L11_I_SIMD_R 4
#define L12_W_SIMD_R 4
#define L12_I_SIMD_R 4
#define L13_W_SIMD_R 4
#define L13_I_SIMD_R 4
#define L14_W_SIMD_R 1
#define L14_I_SIMD_R 1
#define L15_W_SIMD_R 1
#define L15_I_SIMD_R 1
#define L16_W_SIMD_R 1
#define L16_I_SIMD_R 1

//simd__loop_times=IFM_Channels_/SIMD_
#define simd_1_loop_times 1
#define simd_2_loop_times 1
#define simd_3_loop_times 1
#define simd_4_loop_times 1
#define simd_5_loop_times 1
#define simd_6_loop_times 1
#define simd_7_loop_times 1
#define simd_8_loop_times 1
#define simd_9_loop_times 1
#define simd_10_loop_times 1
#define simd_11_loop_times 4
#define simd_12_loop_times 4
#define simd_13_loop_times 4
#define simd_14_loop_times 64
#define simd_15_loop_times 64
#define simd_16_loop_times 64

//layer_i_filter_slide_number && layer_i_left_pe
#define layer1_filter_slide_number 7
#define layer1_left_pe 4
#define layer2_filter_slide_number 4
#define layer2_left_pe 16
#define layer3_filter_slide_number 16
#define layer3_left_pe 8
#define layer4_filter_slide_number 16
#define layer4_left_pe 8
#define layer5_filter_slide_number 64
#define layer5_left_pe 4
#define layer6_filter_slide_number 64
#define layer6_left_pe 4
#define layer7_filter_slide_number 64
#define layer7_left_pe 4
#define layer8_filter_slide_number 512
#define layer8_left_pe 1
#define layer9_filter_slide_number 512
#define layer9_left_pe 1
#define layer10_filter_slide_number 512
#define layer10_left_pe 1
#define layer11_filter_slide_number 512
#define layer11_left_pe 1
#define layer12_filter_slide_number 512
#define layer12_left_pe 1
#define layer13_filter_slide_number 512
#define layer13_left_pe 1
#define layer14_filter_slide_number 512
#define layer14_left_pe 1
#define layer15_filter_slide_number 512
#define layer15_left_pe 1
#define layer16_filter_slide_number 10
#define layer16_left_pe 1

//OFM_Cha_Occupy_Row=OFM_Channel/SIMD_I_TH
#define OFM_1_Cha_Occupy_Row 2
#define OFM_2_Cha_Occupy_Row 2
#define OFM_3_Cha_Occupy_Row 4
#define OFM_4_Cha_Occupy_Row 4
#define OFM_5_Cha_Occupy_Row 8
#define OFM_6_Cha_Occupy_Row 8
#define OFM_7_Cha_Occupy_Row 8
#define OFM_8_Cha_Occupy_Row 16
#define OFM_9_Cha_Occupy_Row 16
#define OFM_10_Cha_Occupy_Row 16
#define OFM_11_Cha_Occupy_Row 16
#define OFM_12_Cha_Occupy_Row 16
#define OFM_13_Cha_Occupy_Row 16
#define OFM_14_Cha_Occupy_Row 16
#define OFM_15_Cha_Occupy_Row 16
#define OFM_16_Cha_Occupy_Row 1

//weight start and end index
//start = previous_end
//end = IFM_Channel * KERNELDim * KERNELDim * OFM_Channel * OFMDim * OFMDim( (/SIMD) or (*WEIGHT_PRECISION/512) ) + current_start
#define w1_start 0
#define w1_end 589824
#define w2_start w1_end
#define w2_end (1179648 + w2_start)
#define w3_start w2_end
#define w3_end (589824 + w3_start)
#define w4_start w3_end
#define w4_end (1179648 + w4_start)
#define w5_start w4_end
#define w5_end (589824 + w5_start)
#define w6_start w5_end
#define w6_end (1179648 + w6_start)
#define w7_start w6_end
#define w7_end (1179648 + w7_start)
#define w8_start w7_end
#define w8_end (589824 + w8_start)
#define w9_start w8_end
#define w9_end (1179648 + w9_start)
#define w10_start w9_end
#define w10_end (1179648 + w10_start)
#define w11_start w10_end
#define w11_end (294912 + w11_start)
#define w12_start w11_end
#define w12_end (294912 + w12_start)
#define w13_start w12_end
#define w13_end (294912 + w13_start)
#define w14_start w13_end
#define w14_end (32768 + w14_start)
#define w15_start w14_end
#define w15_end (32768 + w15_start)
#define w16_start w15_end
#define w16_end (640 + w16_start)

//weight_stream_depth=kernel_pre*IFM_Channel*KERNELDim*KERNELDim/SIMD
int w1_stream_depth=90;
int w2_stream_depth=288;
