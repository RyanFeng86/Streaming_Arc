
///////////////////////////////////////////////////////////
/// Author Yun Feng(fengyun@usc.edu)
///        Arash Fayyazi(fayyazi@usc.edu)
///        Amirhossein Esmaili Dastjerdi(esmailid@usc.edu)
/// Date 04/12/2023
/// Org USC
////////////////////////////////////////////////////////////
#include <hls_stream.h>
#include "ap_int.h"
#include <iostream>
#include <string>
#include "define_.h"


using namespace hls;
using namespace std;

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
		ap_uint<512> *buffer_out);