
///////////////////////////////////////////////////////////
/// Author Yun Feng(fengyun@usc.edu)
///        Arash Fayyazi(fayyazi@usc.edu)
///        Amirhossein Esmaili Dastjerdi(esmailid@usc.edu)
/// Date 04/12/2023
/// Org USC
////////////////////////////////////////////////////////////
#include "define_.h"
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
}