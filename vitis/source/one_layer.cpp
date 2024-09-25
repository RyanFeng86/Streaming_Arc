
///////////////////////////////////////////////////////////
/// Author Yun Feng(fengyun@usc.edu)
///        Arash Fayyazi(fayyazi@usc.edu)
///        Amirhossein Esmaili Dastjerdi(esmailid@usc.edu)
/// Date 04/12/2023
/// Org USC
////////////////////////////////////////////////////////////
#include "one_layer.hpp"


void one_layer(ap_uint<512> *buffer_IFM,
		ap_uint<512> *buffer_weight1,
		ap_uint<512> *buffer_weight2,
		ap_uint<512> *buffer_weight3,
		ap_uint<512> *buffer_weight4,
		ap_uint<512> *buffer_weight5,
		ap_uint<512> *buffer_weight6,
		ap_uint<512> *buffer_weight7,
		ap_uint<512> *buffer_weight8,
		ap_uint<512> *buffer_weight9,
		ap_uint<512> *buffer_weight10,
		ap_uint<512> *buffer_weight11,
		ap_uint<512> *buffer_weight12,
		ap_uint<512> *buffer_weight13,
		ap_uint<512> *buffer_weight14,
		ap_uint<512> *buffer_weight15,
		ap_uint<512> *buffer_weight16,
		ap_uint<512> *buffer_out){
	#pragma HLS INTERFACE m_axi port= buffer_IFM offset=slave bundle=gmem0 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port= buffer_weight1 offset=slave bundle=gmem1 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port= buffer_weight2 offset=slave bundle=gmem2 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port= buffer_weight3 offset=slave bundle=gmem3 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port= buffer_weight4 offset=slave bundle=gmem4 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port= buffer_weight5 offset=slave bundle=gmem5 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port= buffer_weight6 offset=slave bundle=gmem6 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port= buffer_weight7 offset=slave bundle=gmem7 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port= buffer_weight8 offset=slave bundle=gmem8 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port= buffer_weight9 offset=slave bundle=gmem9 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port= buffer_weight10 offset=slave bundle=gmem10 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port= buffer_weight11 offset=slave bundle=gmem11 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port= buffer_weight12 offset=slave bundle=gmem12 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port= buffer_weight13 offset=slave bundle=gmem13 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port= buffer_weight14 offset=slave bundle=gmem14 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port= buffer_weight15 offset=slave bundle=gmem15 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port= buffer_weight16 offset=slave bundle=gmem16 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE m_axi port=buffer_out offset=slave bundle=gmem17 max_read_burst_length=64 max_write_burst_length=64
	#pragma HLS INTERFACE s_axilite port=buffer_IFM bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_weight1 bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_weight2 bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_weight3 bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_weight4 bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_weight5 bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_weight6 bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_weight7 bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_weight8 bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_weight9 bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_weight10 bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_weight11 bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_weight12 bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_weight13 bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_weight14 bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_weight15 bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_weight16 bundle=control
	#pragma HLS INTERFACE s_axilite port=buffer_out bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

/////////////////////////////////////////////////////////////
//     start to declare weight streams for every layer     //
/////////////////////////////////////////////////////////////
	static stream<ap_uint<SIMD_1*WEIGHT_PRECISION> > w1[PE_1];
	static stream<ap_uint<512> > w2[PE_2*L2_I_SIMD_R];
	static stream<ap_uint<512> > w3[PE_3*L3_I_SIMD_R];
	static stream<ap_uint<512> > w4[PE_4*L4_I_SIMD_R];
	static stream<ap_uint<512> > w5[PE_5*L5_I_SIMD_R];
	static stream<ap_uint<512> > w6[PE_6*L6_I_SIMD_R];
	static stream<ap_uint<512> > w7[PE_7*L7_I_SIMD_R];
	static stream<ap_uint<512> > w8[PE_8*L8_I_SIMD_R];
	static stream<ap_uint<512> > w9[PE_9*L9_I_SIMD_R];
	static stream<ap_uint<512> > w10[PE_10*L10_I_SIMD_R];
	static stream<ap_uint<512> > w11[PE_11*L11_I_SIMD_R];
	static stream<ap_uint<512> > w12[PE_12*L12_I_SIMD_R];
	static stream<ap_uint<512> > w13[PE_13*L13_I_SIMD_R];
	static stream<ap_uint<SIMD_14*WEIGHT_PRECISION> > w14[PE_14];
	static stream<ap_uint<SIMD_15*WEIGHT_PRECISION> > w15[PE_15];
	static stream<ap_uint<SIMD_16*WEIGHT_PRECISION> > w16[PE_16];

////////////////////////////////////////////////////////////
//     start to declare IFM streams for every layer    /////
////////////////////////////////////////////////////////////
	static stream<ap_uint<SIMD_1*ACTIVATION_PRECISION> > IFM1("IFM1");
	static stream<ap_uint<SIMD_1*ACTIVATION_PRECISION> > IFM1_through_G[PE_1];
	static stream<ap_uint<512> > IFM2("IFM2");
	static stream<ap_uint<512> > IFM2_through_G[PE_2*L2_I_SIMD_R];
	static stream<ap_uint<512> > IFM3("IFM3");
	static stream<ap_uint<512> > IFM3_through_G[PE_3*L3_I_SIMD_R];
	static stream<ap_uint<512> > IFM4("IFM4");
	static stream<ap_uint<512> > IFM4_through_G[PE_4*L4_I_SIMD_R];
	static stream<ap_uint<512> > IFM5("IFM5");
	static stream<ap_uint<512> > IFM5_through_G[PE_5*L5_I_SIMD_R];
	static stream<ap_uint<512> > IFM6("IFM6");
	static stream<ap_uint<512> > IFM6_through_G[PE_6*L6_I_SIMD_R];
	static stream<ap_uint<512> > IFM7("IFM7");
	static stream<ap_uint<512> > IFM7_through_G[PE_7*L7_I_SIMD_R];
	static stream<ap_uint<512> > IFM8("IFM8");
	static stream<ap_uint<512> > IFM8_through_G[PE_8*L8_I_SIMD_R];
	static stream<ap_uint<512> > IFM9("IFM9");
	static stream<ap_uint<512> > IFM9_through_G[PE_9*L9_I_SIMD_R];
	static stream<ap_uint<512> > IFM10("IFM10");
	static stream<ap_uint<512> > IFM10_through_G[PE_10*L10_I_SIMD_R];
	static stream<ap_uint<512> > IFM11("IFM11");
	static stream<ap_uint<512> > IFM11_through_G[PE_11*L11_I_SIMD_R];
	static stream<ap_uint<512> > IFM12("IFM12");
	static stream<ap_uint<512> > IFM12_through_G[PE_12*L12_I_SIMD_R];
	static stream<ap_uint<512> > IFM13("IFM13");
	static stream<ap_uint<512> > IFM13_through_G[PE_13*L13_I_SIMD_R];
	static stream<ap_uint<SIMD_14*ACTIVATION_PRECISION> > IFM14("IFM14");
	static stream<ap_uint<SIMD_14*ACTIVATION_PRECISION> > IFM14_through_G[PE_14];
	static stream<ap_uint<SIMD_15*ACTIVATION_PRECISION> > IFM15("IFM15");
	static stream<ap_uint<SIMD_15*ACTIVATION_PRECISION> > IFM15_through_G[PE_15];
	static stream<ap_uint<SIMD_16*ACTIVATION_PRECISION> > IFM16("IFM16");
	static stream<ap_uint<SIMD_16*ACTIVATION_PRECISION> > IFM16_through_G[PE_16];

//////////////////////////////////////////////////////////
//start to declare Mul_result streams for every layer/////
//////////////////////////////////////////////////////////
	static stream<ap_uint<512> > Mul_result_1("Mul_result_1");
	static stream<ap_uint<512> > Mul_result_2("Mul_result_2");
	static stream<ap_uint<512> > Mul_result_3("Mul_result_3");
	static stream<ap_uint<512> > Mul_result_4("Mul_result_4");
	static stream<ap_uint<512> > Mul_result_5("Mul_result_5");
	static stream<ap_uint<512> > Mul_result_6("Mul_result_6");
	static stream<ap_uint<512> > Mul_result_7("Mul_result_7");
	static stream<ap_uint<512> > Mul_result_8("Mul_result_8");
	static stream<ap_uint<512> > Mul_result_9("Mul_result_9");
	static stream<ap_uint<512> > Mul_result_10("Mul_result_10");
	static stream<ap_uint<512> > Mul_result_11("Mul_result_11");
	static stream<ap_uint<512> > Mul_result_12("Mul_result_12");
	static stream<ap_uint<512> > Mul_result_13("Mul_result_13");
	static stream<ap_uint<512> > Mul_result_14("Mul_result_14");
	static stream<ap_uint<512> > Mul_result_15("Mul_result_15");
	static stream<ap_uint<512> > Mul_result_16("Mul_result_16");

/////////////////////////////////////
//start to declare pooling stream////
/////////////////////////////////////
	static stream<ap_uint<512> > Mul_pooling_2("Mul_pooling_2");
	static stream<ap_uint<512> > Mul_pooling_4("Mul_pooling_4");
	static stream<ap_uint<512> > Mul_pooling_7("Mul_pooling_7");
	static stream<ap_uint<512> > Mul_pooling_10("Mul_pooling_10");
	static stream<ap_uint<512> > Mul_pooling_13("Mul_pooling_13");

///////////////////////////////////////////////////
//start to declare every streams' depth////////////
///////////////////////////////////////////////////
	#pragma HLS STREAM variable = w1  depth = w1_stream_depth
	#pragma HLS STREAM variable = IFM1  depth = ifm_1_stream_depth
	#pragma HLS STREAM variable = IFM1_through_G  depth = ifm_1_gen_stream_depth
	#pragma HLS STREAM variable = Mul_result_1 depth = mul_1_result_stream_depth

	#pragma HLS STREAM variable = w2  depth = w2_stream_depth
	#pragma HLS STREAM variable = IFM2  depth = ifm_2_stream_depth
	#pragma HLS STREAM variable = IFM2_through_G  depth = ifm_2_gen_stream_depth
	#pragma HLS STREAM variable = Mul_result_2 depth = mul_2_result_stream_depth
	#pragma HLS STREAM variable = Mul_pooling_2 depth = pooling_2_result_stream_depth

	#pragma HLS STREAM variable = w3  depth = w3_stream_depth
	#pragma HLS STREAM variable = IFM3  depth = ifm_3_stream_depth
	#pragma HLS STREAM variable = IFM3_through_G  depth = ifm_3_gen_stream_depth
	#pragma HLS STREAM variable = Mul_result_3 depth = mul_3_result_stream_depth

	#pragma HLS STREAM variable = w4  depth = w4_stream_depth
	#pragma HLS STREAM variable = IFM4  depth = ifm_4_stream_depth
	#pragma HLS STREAM variable = IFM4_through_G  depth = ifm_4_gen_stream_depth
	#pragma HLS STREAM variable = Mul_result_4 depth = mul_4_result_stream_depth
	#pragma HLS STREAM variable = Mul_pooling_4 depth = pooling_4_result_stream_depth

	#pragma HLS STREAM variable = w5  depth = w5_stream_depth
	#pragma HLS STREAM variable = IFM5  depth = ifm_5_stream_depth
	#pragma HLS STREAM variable = IFM5_through_G  depth = ifm_5_gen_stream_depth
	#pragma HLS STREAM variable = Mul_result_5 depth = mul_5_result_stream_depth

	#pragma HLS STREAM variable = w6  depth = w6_stream_depth
	#pragma HLS STREAM variable = IFM6  depth = ifm_6_stream_depth
	#pragma HLS STREAM variable = IFM6_through_G  depth = ifm_6_gen_stream_depth
	#pragma HLS STREAM variable = Mul_result_6 depth = mul_6_result_stream_depth

	#pragma HLS STREAM variable = w7  depth = w7_stream_depth
	#pragma HLS STREAM variable = IFM7  depth = ifm_7_stream_depth
	#pragma HLS STREAM variable = IFM7_through_G  depth = ifm_7_gen_stream_depth
	#pragma HLS STREAM variable = Mul_result_7 depth = mul_7_result_stream_depth
	#pragma HLS STREAM variable = Mul_pooling_7 depth = pooling_7_result_stream_depth

	#pragma HLS STREAM variable = w8  depth = w8_stream_depth
	#pragma HLS STREAM variable = IFM8  depth = ifm_8_stream_depth
	#pragma HLS STREAM variable = IFM8_through_G  depth = ifm_8_gen_stream_depth
	#pragma HLS STREAM variable = Mul_result_8 depth = mul_8_result_stream_depth

	#pragma HLS STREAM variable = w9  depth = w9_stream_depth
	#pragma HLS STREAM variable = IFM9  depth = ifm_9_stream_depth
	#pragma HLS STREAM variable = IFM9_through_G  depth = ifm_9_gen_stream_depth
	#pragma HLS STREAM variable = Mul_result_9 depth = mul_9_result_stream_depth

	#pragma HLS STREAM variable = w10  depth = w10_stream_depth
	#pragma HLS STREAM variable = IFM10  depth = ifm_10_stream_depth
	#pragma HLS STREAM variable = IFM10_through_G  depth = ifm_10_gen_stream_depth
	#pragma HLS STREAM variable = Mul_result_10 depth = mul_10_result_stream_depth
	#pragma HLS STREAM variable = Mul_pooling_10 depth = pooling_10_result_stream_depth

	#pragma HLS STREAM variable = w11  depth = w11_stream_depth
	#pragma HLS STREAM variable = IFM11  depth = ifm_11_stream_depth
	#pragma HLS STREAM variable = IFM11_through_G  depth = ifm_11_gen_stream_depth
	#pragma HLS STREAM variable = Mul_result_11 depth = mul_11_result_stream_depth

	#pragma HLS STREAM variable = w12  depth = w12_stream_depth
	#pragma HLS STREAM variable = IFM12  depth = ifm_12_stream_depth
	#pragma HLS STREAM variable = IFM12_through_G  depth = ifm_12_gen_stream_depth
	#pragma HLS STREAM variable = Mul_result_12 depth = mul_12_result_stream_depth

	#pragma HLS STREAM variable = w13  depth = w13_stream_depth
	#pragma HLS STREAM variable = IFM13  depth = ifm_13_stream_depth
	#pragma HLS STREAM variable = IFM13_through_G  depth = ifm_13_gen_stream_depth
	#pragma HLS STREAM variable = Mul_result_13 depth = mul_13_result_stream_depth
	#pragma HLS STREAM variable = Mul_pooling_13 depth = pooling_13_result_stream_depth

	#pragma HLS STREAM variable = w14  depth = w14_stream_depth
	#pragma HLS STREAM variable = IFM14  depth = ifm_14_stream_depth
	#pragma HLS STREAM variable = IFM14_through_G  depth = ifm_14_gen_stream_depth
	#pragma HLS STREAM variable = Mul_result_14 depth = mul_14_result_stream_depth

	#pragma HLS STREAM variable = w15  depth = w15_stream_depth
	#pragma HLS STREAM variable = IFM15  depth = ifm_15_stream_depth
	#pragma HLS STREAM variable = IFM15_through_G  depth = ifm_15_gen_stream_depth
	#pragma HLS STREAM variable = Mul_result_15 depth = mul_15_result_stream_depth

	#pragma HLS STREAM variable = w16  depth = w16_stream_depth
	#pragma HLS STREAM variable = IFM16  depth = ifm_16_stream_depth
	#pragma HLS STREAM variable = IFM16_through_G  depth = ifm_16_gen_stream_depth
	#pragma HLS STREAM variable = Mul_result_16 depth = mul_16_result_stream_depth


	#pragma HLS dataflow   //begin parallel
////////////////////////read data in, IFM-> generator->PE array, W->PE array
	read_in_short<SIMD_1*ACTIVATION_PRECISION,SIMD_1,IFMDim_1+2,IFM_Channels_1>(buffer_IFM,IFM1);
	read_weight_short<SIMD_1*WEIGHT_PRECISION,0,(((w1_end-w1_start)/(OFMDim_1*OFMDim_1))/(KERNELDim_1*KERNELDim_1*PE_1))*(KERNELDim_1*KERNELDim_1*PE_1),(w1_end-w1_start)/(OFMDim_1*OFMDim_1),PE_1,layer1_left_pe,OFMDim_1>(buffer_weight1,w1);
	Input_Generator_short_1<SIMD_1*ACTIVATION_PRECISION, IFM_Channels_occupy_rows_1,SIMD_1*ACTIVATION_PRECISION,IFM_1_precol,KERNELDim_1,PE_1,IFMDim_1+2,OFMDim_1,layer1_filter_slide_number,layer1_left_pe,w1_start,w1_end>(IFM1,IFM1_through_G);
	Mac_short_1<SIMD_1*ACTIVATION_PRECISION,SIMD_1*WEIGHT_PRECISION,SIMD_1,PE_1,KERNELDim_1,OFMDim_1,layer1_filter_slide_number,layer1_left_pe,simd_1_loop_times>(IFM1_through_G,w1,Mul_result_1);

	Padding<OFM_1_Cha_Occupy_Row,OFMDim_1>(Mul_result_1,IFM2);
	read_weight_long<0,(((w2_end-w2_start)/(OFMDim_2*OFMDim_2))/(KERNELDim_2*KERNELDim_2*PE_2*L2_I_SIMD_R))*(KERNELDim_2*KERNELDim_2*PE_2*L2_I_SIMD_R),(w2_end-w2_start)/(OFMDim_2*OFMDim_2),PE_2,layer2_left_pe,L2_I_SIMD_R,OFMDim_2>(buffer_weight2,w2);
	Input_Generator_long<L2_I_SIMD_R,simd_2_loop_times,IFM_2_precol,KERNELDim_2,PE_2,IFMDim_2+2,OFMDim_2,layer2_filter_slide_number,layer2_left_pe,w2_start,w2_end>(IFM2,IFM2_through_G);
	Mac_long<L2_I_SIMD_R,L2_W_SIMD_R,PE_2,KERNELDim_2,OFMDim_2,layer2_filter_slide_number,layer2_left_pe,simd_2_loop_times>(IFM2_through_G,w2,Mul_result_2);
	Pooling<OFM_2_Cha_Occupy_Row,OFMDim_2>(Mul_result_2, Mul_pooling_2);

	Padding<OFM_2_Cha_Occupy_Row,OFMDim_2/2>(Mul_pooling_2,IFM3);
	read_weight_long<0,(((w3_end-w3_start)/(OFMDim_3*OFMDim_3))/(KERNELDim_3*KERNELDim_3*PE_3*L3_I_SIMD_R))*(KERNELDim_3*KERNELDim_3*PE_3*L3_I_SIMD_R),(w3_end-w3_start)/(OFMDim_3*OFMDim_3),PE_3,layer3_left_pe,L3_I_SIMD_R,OFMDim_3>(buffer_weight3,w3);
	Input_Generator_long<L3_I_SIMD_R,simd_3_loop_times,IFM_3_precol,KERNELDim_3,PE_3,IFMDim_3+2,OFMDim_3,layer3_filter_slide_number,layer3_left_pe,w3_start,w3_end>(IFM3,IFM3_through_G);
	Mac_long<L3_I_SIMD_R,L3_W_SIMD_R,PE_3,KERNELDim_3,OFMDim_3,layer3_filter_slide_number,layer3_left_pe,simd_3_loop_times>(IFM3_through_G,w3,Mul_result_3);

	Padding<OFM_3_Cha_Occupy_Row,OFMDim_3>(Mul_result_3,IFM4);
	read_weight_long<0,(((w4_end-w4_start)/(OFMDim_4*OFMDim_4))/(KERNELDim_4*KERNELDim_4*PE_4*L4_I_SIMD_R))*(KERNELDim_4*KERNELDim_4*PE_4*L4_I_SIMD_R),(w4_end-w4_start)/(OFMDim_4*OFMDim_4),PE_4,layer4_left_pe,L4_I_SIMD_R,OFMDim_4>(buffer_weight4,w4);
	Input_Generator_long<L4_I_SIMD_R,simd_4_loop_times,IFM_4_precol,KERNELDim_4,PE_4,IFMDim_4+2,OFMDim_4,layer4_filter_slide_number,layer4_left_pe,w4_start,w4_end>(IFM4,IFM4_through_G);
	Mac_long<L4_I_SIMD_R,L4_W_SIMD_R,PE_4,KERNELDim_4,OFMDim_4,layer4_filter_slide_number,layer4_left_pe,simd_4_loop_times>(IFM4_through_G,w4,Mul_result_4);
	Pooling<OFM_4_Cha_Occupy_Row,OFMDim_4>(Mul_result_4, Mul_pooling_4);

	Padding<OFM_4_Cha_Occupy_Row,OFMDim_4/2>(Mul_pooling_4,IFM5);
	read_weight_long<0,(((w5_end-w5_start)/(OFMDim_5*OFMDim_5))/(KERNELDim_5*KERNELDim_5*PE_5*L5_I_SIMD_R))*(KERNELDim_5*KERNELDim_5*PE_5*L5_I_SIMD_R),(w5_end-w5_start)/(OFMDim_5*OFMDim_5),PE_5,layer5_left_pe,L5_I_SIMD_R,OFMDim_5>(buffer_weight5,w5);
	Input_Generator_long<L5_I_SIMD_R,simd_5_loop_times,IFM_5_precol,KERNELDim_5,PE_5,IFMDim_5+2,OFMDim_5,layer5_filter_slide_number,layer5_left_pe,w5_start,w5_end>(IFM5,IFM5_through_G);
	Mac_long<L5_I_SIMD_R,L5_W_SIMD_R,PE_5,KERNELDim_5,OFMDim_5,layer5_filter_slide_number,layer5_left_pe,simd_5_loop_times>(IFM5_through_G,w5,Mul_result_5);

	Padding<OFM_5_Cha_Occupy_Row,OFMDim_5>(Mul_result_5,IFM6);
	read_weight_long<0,(((w6_end-w6_start)/(OFMDim_6*OFMDim_6))/(KERNELDim_6*KERNELDim_6*PE_6*L6_I_SIMD_R))*(KERNELDim_6*KERNELDim_6*PE_6*L6_I_SIMD_R),(w6_end-w6_start)/(OFMDim_6*OFMDim_6),PE_6,layer6_left_pe,L6_I_SIMD_R,OFMDim_6>(buffer_weight6,w6);
	Input_Generator_long<L6_I_SIMD_R,simd_6_loop_times,IFM_6_precol,KERNELDim_6,PE_6,IFMDim_6+2,OFMDim_6,layer6_filter_slide_number,layer6_left_pe,w6_start,w6_end>(IFM6,IFM6_through_G);
	Mac_long<L6_I_SIMD_R,L6_W_SIMD_R,PE_6,KERNELDim_6,OFMDim_6,layer6_filter_slide_number,layer6_left_pe,simd_6_loop_times>(IFM6_through_G,w6,Mul_result_6);

	Padding<OFM_6_Cha_Occupy_Row,OFMDim_6>(Mul_result_6,IFM7);
	read_weight_long<0,(((w7_end-w7_start)/(OFMDim_7*OFMDim_7))/(KERNELDim_7*KERNELDim_7*PE_7*L7_I_SIMD_R))*(KERNELDim_7*KERNELDim_7*PE_7*L7_I_SIMD_R),(w7_end-w7_start)/(OFMDim_7*OFMDim_7),PE_7,layer7_left_pe,L7_I_SIMD_R,OFMDim_7>(buffer_weight7,w7);
	Input_Generator_long<L7_I_SIMD_R,simd_7_loop_times,IFM_7_precol,KERNELDim_7,PE_7,IFMDim_7+2,OFMDim_7,layer7_filter_slide_number,layer7_left_pe,w7_start,w7_end>(IFM7,IFM7_through_G);
	Mac_long<L7_I_SIMD_R,L7_W_SIMD_R,PE_7,KERNELDim_7,OFMDim_7,layer7_filter_slide_number,layer7_left_pe,simd_7_loop_times>(IFM7_through_G,w7,Mul_result_7);
	Pooling<OFM_7_Cha_Occupy_Row,OFMDim_7>(Mul_result_7, Mul_pooling_7);

	Padding<OFM_7_Cha_Occupy_Row,OFMDim_7/2>(Mul_pooling_7,IFM8);
	read_weight_long<0,(((w8_end-w8_start)/(OFMDim_8*OFMDim_8))/(KERNELDim_8*KERNELDim_8*PE_8*L8_I_SIMD_R))*(KERNELDim_8*KERNELDim_8*PE_8*L8_I_SIMD_R),(w8_end-w8_start)/(OFMDim_8*OFMDim_8),PE_8,layer8_left_pe,L8_I_SIMD_R,OFMDim_8>(buffer_weight8,w8);
	Input_Generator_long<L8_I_SIMD_R,simd_8_loop_times,IFM_8_precol,KERNELDim_8,PE_8,IFMDim_8+2,OFMDim_8,layer8_filter_slide_number,layer8_left_pe,w8_start,w8_end>(IFM8,IFM8_through_G);
	Mac_long<L8_I_SIMD_R,L8_W_SIMD_R,PE_8,KERNELDim_8,OFMDim_8,layer8_filter_slide_number,layer8_left_pe,simd_8_loop_times>(IFM8_through_G,w8,Mul_result_8);

	Padding<OFM_8_Cha_Occupy_Row,OFMDim_8>(Mul_result_8,IFM9);
	read_weight_long<0,(((w9_end-w9_start)/(OFMDim_9*OFMDim_9))/(KERNELDim_9*KERNELDim_9*PE_9*L9_I_SIMD_R))*(KERNELDim_9*KERNELDim_9*PE_9*L9_I_SIMD_R),(w9_end-w9_start)/(OFMDim_9*OFMDim_9),PE_9,layer9_left_pe,L9_I_SIMD_R,OFMDim_9>(buffer_weight9,w9);
	Input_Generator_long<L9_I_SIMD_R,simd_9_loop_times,IFM_9_precol,KERNELDim_9,PE_9,IFMDim_9+2,OFMDim_9,layer9_filter_slide_number,layer9_left_pe,w9_start,w9_end>(IFM9,IFM9_through_G);
	Mac_long<L9_I_SIMD_R,L9_W_SIMD_R,PE_9,KERNELDim_9,OFMDim_9,layer9_filter_slide_number,layer9_left_pe,simd_9_loop_times>(IFM9_through_G,w9,Mul_result_9);

	Padding<OFM_9_Cha_Occupy_Row,OFMDim_9>(Mul_result_9,IFM10);
	read_weight_long<0,(((w10_end-w10_start)/(OFMDim_10*OFMDim_10))/(KERNELDim_10*KERNELDim_10*PE_10*L10_I_SIMD_R))*(KERNELDim_10*KERNELDim_10*PE_10*L10_I_SIMD_R),(w10_end-w10_start)/(OFMDim_10*OFMDim_10),PE_10,layer10_left_pe,L10_I_SIMD_R,OFMDim_10>(buffer_weight10,w10);
	Input_Generator_long<L10_I_SIMD_R,simd_10_loop_times,IFM_10_precol,KERNELDim_10,PE_10,IFMDim_10+2,OFMDim_10,layer10_filter_slide_number,layer10_left_pe,w10_start,w10_end>(IFM10,IFM10_through_G);
	Mac_long<L10_I_SIMD_R,L10_W_SIMD_R,PE_10,KERNELDim_10,OFMDim_10,layer10_filter_slide_number,layer10_left_pe,simd_10_loop_times>(IFM10_through_G,w10,Mul_result_10);
	Pooling<OFM_10_Cha_Occupy_Row,OFMDim_10>(Mul_result_10, Mul_pooling_10);

	Padding<OFM_10_Cha_Occupy_Row,OFMDim_10/2>(Mul_pooling_10,IFM11);
	read_weight_long<0,(((w11_end-w11_start)/(OFMDim_11*OFMDim_11))/(KERNELDim_11*KERNELDim_11*PE_11*L11_I_SIMD_R))*(KERNELDim_11*KERNELDim_11*PE_11*L11_I_SIMD_R),(w11_end-w11_start)/(OFMDim_11*OFMDim_11),PE_11,layer11_left_pe,L11_I_SIMD_R,OFMDim_11>(buffer_weight11,w11);
	Input_Generator_long<L11_I_SIMD_R,simd_11_loop_times,IFM_11_precol,KERNELDim_11,PE_11,IFMDim_11+2,OFMDim_11,layer11_filter_slide_number,layer11_left_pe,w11_start,w11_end>(IFM11,IFM11_through_G);
	Mac_long<L11_I_SIMD_R,L11_W_SIMD_R,PE_11,KERNELDim_11,OFMDim_11,layer11_filter_slide_number,layer11_left_pe,simd_11_loop_times>(IFM11_through_G,w11,Mul_result_11);

	Padding<OFM_11_Cha_Occupy_Row,OFMDim_11>(Mul_result_11,IFM12);
	read_weight_long<0,(((w12_end-w12_start)/(OFMDim_12*OFMDim_12))/(KERNELDim_12*KERNELDim_12*PE_12*L12_I_SIMD_R))*(KERNELDim_12*KERNELDim_12*PE_12*L12_I_SIMD_R),(w12_end-w12_start)/(OFMDim_12*OFMDim_12),PE_12,layer12_left_pe,L12_I_SIMD_R,OFMDim_12>(buffer_weight12,w12);
	Input_Generator_long<L12_I_SIMD_R,simd_12_loop_times,IFM_12_precol,KERNELDim_12,PE_12,IFMDim_12+2,OFMDim_12,layer12_filter_slide_number,layer12_left_pe,w12_start,w12_end>(IFM12,IFM12_through_G);
	Mac_long<L12_I_SIMD_R,L12_W_SIMD_R,PE_12,KERNELDim_12,OFMDim_12,layer12_filter_slide_number,layer12_left_pe,simd_12_loop_times>(IFM12_through_G,w12,Mul_result_12);

	Padding<OFM_12_Cha_Occupy_Row,OFMDim_12>(Mul_result_12,IFM13);
	read_weight_long<0,(((w13_end-w13_start)/(OFMDim_13*OFMDim_13))/(KERNELDim_13*KERNELDim_13*PE_13*L13_I_SIMD_R))*(KERNELDim_13*KERNELDim_13*PE_13*L13_I_SIMD_R),(w13_end-w13_start)/(OFMDim_13*OFMDim_13),PE_13,layer13_left_pe,L13_I_SIMD_R,OFMDim_13>(buffer_weight13,w13);
	Input_Generator_long<L13_I_SIMD_R,simd_13_loop_times,IFM_13_precol,KERNELDim_13,PE_13,IFMDim_13+2,OFMDim_13,layer13_filter_slide_number,layer13_left_pe,w13_start,w13_end>(IFM13,IFM13_through_G);
	Mac_long<L13_I_SIMD_R,L13_W_SIMD_R,PE_13,KERNELDim_13,OFMDim_13,layer13_filter_slide_number,layer13_left_pe,simd_13_loop_times>(IFM13_through_G,w13,Mul_result_13);
	Pooling<OFM_13_Cha_Occupy_Row,OFMDim_13>(Mul_result_13, Mul_pooling_13);

	read_weight_short<SIMD_14*WEIGHT_PRECISION,0,(((w14_end-w14_start)/(OFMDim_14*OFMDim_14))/(KERNELDim_14*KERNELDim_14*PE_14))*(KERNELDim_14*KERNELDim_14*PE_14),(w14_end-w14_start)/(OFMDim_14*OFMDim_14),PE_14,layer14_left_pe,OFMDim_14>(buffer_weight14,w14);
	Input_Generator_short_2<512,IFM_Channels_occupy_rows_14,SIMD_14*ACTIVATION_PRECISION,IFM_14_precol,KERNELDim_14,PE_14,IFMDim_14,OFMDim_14,layer14_filter_slide_number,layer14_left_pe,w14_start,w14_end>(Mul_pooling_13,IFM14_through_G);
	Mac_short_2<SIMD_14*ACTIVATION_PRECISION,SIMD_14*WEIGHT_PRECISION,SIMD_14,PE_14,KERNELDim_14,OFMDim_14,layer14_filter_slide_number,layer14_left_pe,simd_14_loop_times>(IFM14_through_G,w14,Mul_result_14);

	read_weight_short<SIMD_15*WEIGHT_PRECISION,0,(((w15_end-w15_start)/(OFMDim_15*OFMDim_15))/(KERNELDim_15*KERNELDim_15*PE_15))*(KERNELDim_15*KERNELDim_15*PE_15),(w15_end-w15_start)/(OFMDim_15*OFMDim_15),PE_15,layer15_left_pe,OFMDim_15>(buffer_weight15,w15);
	Input_Generator_short_2<512,IFM_Channels_occupy_rows_15,SIMD_15*ACTIVATION_PRECISION,IFM_15_precol,KERNELDim_15,PE_15,IFMDim_15,OFMDim_15,layer15_filter_slide_number,layer15_left_pe,w15_start,w15_end>(Mul_result_14,IFM15_through_G);
	Mac_short_2<SIMD_15*ACTIVATION_PRECISION,SIMD_15*WEIGHT_PRECISION,SIMD_15,PE_15,KERNELDim_15,OFMDim_15,layer15_filter_slide_number,layer15_left_pe,simd_15_loop_times>(IFM15_through_G,w15,Mul_result_15);

	read_weight_short<SIMD_16*WEIGHT_PRECISION,0,(((w16_end-w16_start)/(OFMDim_16*OFMDim_16))/(KERNELDim_16*KERNELDim_16*PE_16))*(KERNELDim_16*KERNELDim_16*PE_16),(w16_end-w16_start)/(OFMDim_16*OFMDim_16),PE_16,layer16_left_pe,OFMDim_16>(buffer_weight16,w16);
	Input_Generator_short_2<512,IFM_Channels_occupy_rows_16,SIMD_16*ACTIVATION_PRECISION,IFM_16_precol,KERNELDim_16,PE_16,IFMDim_16,OFMDim_16,layer16_filter_slide_number,layer16_left_pe,w16_start,w16_end>(Mul_result_15,IFM16_through_G);
	Mac_short_2<SIMD_16*ACTIVATION_PRECISION,SIMD_16*WEIGHT_PRECISION,SIMD_16,PE_16,KERNELDim_16,OFMDim_16,layer16_filter_slide_number,layer16_left_pe,simd_16_loop_times>(IFM16_through_G,w16,Mul_result_16);

	ap_uint<512> tmp = Mul_result_16.read();
	buffer_out[0]=tmp;
	cout<<tmp<<endl;
	
	}
