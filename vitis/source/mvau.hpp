/******************************************************************************
 *  Copyright (c) 2019, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *******************************************************************************/

/*******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  \file mvau.hpp
 *
 *  This file lists a templated funtion used to implement  
 *  Matrix-Vector-Activation Unit
 *
 *  This project has received funding from the European Union's Framework
 *  Programme for Research and Innovation Horizon 2020 (2014-2020) under
 *  the Marie Sk��odowska-Curie Grant Agreement No. 751339.
 *
 *******************************************************************************/

#ifndef MVAU_HPP
#define MVAU_HPP

#include "hls_stream.h"

#include "mac.hpp"
#include "interpret.hpp"

/**
 * \brief Matrix vector activate function
 *
 * The function performs the multiplication between a weigth matrix and the input activation vector,
 * accumulating the results and then applying an activation function on the accumulated result.
 *
 * 
 * \tparam MatrixW    Width of the input matrix
 * \tparam MatrixH    Heigth of the input matrix
 * \tparam SIMD       Number of input columns computed in parallel
 * \tparam PE         Number of output rows computed in parallel
 * \tparam MMV        Number of output pixels computed in parallel
 * \tparam TSrcI      DataType of the input activation (as used in the MAC)
 * \tparam TDstI      DataType of the output activation (as generated by the activation)
 * \tparam TWeightI   DataType of the weights and how to access them in the array
 * \tparam TI         DataType of the input stream - safely deducible from the paramaters
 * \tparam TO         DataType of the output stream - safely deducible from the paramaters
 * \tparam TW         DataType of the weights matrix - safely deducible from the paramaters
 * \tparam TA         DataType of the activation class (e.g. thresholds) - safely deducible from the paramaters
 * \tparam R          Datatype for the resource used for FPGA implementation of the MAC  - safely deducible from the paramaters
 *
 * \param in          Input stream
 * \param out         Output stream
 * \param weights     Weights matrix (currently supports BinaryWeights or FixedPointWeights)
 * \param activation  Activation class
 * \param reps        Number of time the function has to be repeatedly executed (e.g. number of images)
 * \param r           Resource type for the hardware implementation of the MAC block
 */
template<
  unsigned MatrixW, unsigned MatrixH, unsigned SIMD, unsigned PE, unsigned MMV, 
  typename TSrcI = Identity, typename TDstI = Identity, typename TWeightI = Identity,
  typename TI, typename TO, typename TW, typename TA, typename R
>
void Matrix_Vector_Activate_Batch(hls::stream<TI> &in,
				  hls::stream<TO> &out,
				  TW  const &weights,
				  TA  const &activation,
				  int const  reps,
				  R const &r) {

  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const  NF = MatrixH / PE;

  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks
  unsigned const  SF = MatrixW / SIMD;

  // input vector buffers
  TI  inputBuf[SF];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=0


  decltype(activation.init(0,0))  accu[MMV][PE];
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

  unsigned  nf   = 0;
  unsigned  sf   = 0;
  unsigned  tile = 0; // invariant: tile = nf*SF + sf

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  unsigned const TOTAL_FOLD = NF * SF;
  for(unsigned  i = 0; i < reps * TOTAL_FOLD; i++) {
#pragma HLS PIPELINE II=1
    TI  inElem;
    if(nf == 0) {
      // read input from stream
      inElem = in.read();
      // store in appropriate buffer for reuse
      inputBuf[sf] = inElem;
    }
    else {
      // reuse buffered input
      inElem = inputBuf[sf];
    }

    // Threshold Initialisation
    if(sf == 0) {
      for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
        for(unsigned mmv = 0; mmv < MMV; mmv++) {
#pragma HLS UNROLL
          accu[mmv][pe] = activation.init(nf, pe);
        }
      }
    }

    // compute matrix-vector product for each processing element
    auto const &w = weights.weights(tile);
    for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
      auto const  wgt = TWeightI()(w[pe]);
      for (unsigned mmv = 0; mmv < MMV; mmv++){
        auto const  act = TSrcI()(inElem, mmv);
        accu[mmv][pe] = mac<SIMD>(accu[mmv][pe], wgt, act, r, mmv);
      }
    }

    // keep track of which folded synapse/neuron we are processing
    ++tile;
    if(++sf == SF) {
      // produce output and clear accumulators
      auto  outElem = TDstI().template operator()<TO>();
      for (unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
        for (unsigned mmv = 0; mmv < MMV; mmv++){
#pragma HLS UNROLL
          outElem(pe,mmv,1) = activation.activate(nf, pe, accu[mmv][pe]);
        }
      }
      out.write(outElem);
      // next folded neuron or image
      sf = 0;
      if(++nf == NF) {
	    nf   = 0;
	    tile = 0;
      }
    }
  }
}


/**
 * \brief Matrix vector activate function with streaming weights
 *
 * The function performs the multiplication between a weigth matrix, presnted as an input stream, and the input activation vector,
 * accumulating the results and then applying an activation function on the accumulated result. Does not support MMV.
 *
 * 
 * \tparam MatrixW    Width of the input matrix
 * \tparam MatrixH    Heigth of the input matrix
 * \tparam SIMD       Number of input columns computed in parallel
 * \tparam PE         Number of output rows computed in parallel
 * \tparam TSrcI      DataType of the input activation (as used in the MAC)
 * \tparam TDstI      DataType of the output activation (as generated by the activation)
 * \tparam TWeightI   DataType of the weights and how to access them in the array
 * \tparam TW         DataType of the weights (as used in the MAC) - not deducible from the paramaters
 * \tparam TI         DataType of the input stream - safely deducible from the paramaters
 * \tparam TO         DataType of the output stream - safely deducible from the paramaters
 * \tparam TA         DataType of the activation class (e.g. thresholds) - safely deducible from the paramaters
 * \tparam R          Datatype for the resource used for FPGA implementation of the MAC  - safely deducible from the paramaters
 *
 * \param in          Input stream
 * \param out         Output stream
 * \param weight      Weight stream (currently supports BinaryWeights or FixedPointWeights)
 * \param activation  Activation class
 * \param reps        Number of time the function has to be repeatedly executed (e.g. number of images)
 * \param r           Resource type for the hardware implementation of the MAC block
 */
template<
  unsigned MatrixW, unsigned MatrixH, unsigned SIMD, unsigned PE, 
  typename TSrcI = Identity, typename TDstI = Identity, typename TWeightI = Identity, typename TW,
  typename TI, typename TO, typename TA, typename R
>
void Matrix_Vector_Activate_Stream_Batch(hls::stream<TI> &in,
          hls::stream<TO> &out,
          hls::stream<ap_uint<PE*SIMD*TW::width>> &weight,
          TA  const &activation,
          int const  reps,
          R const &r) {

  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const  NF = MatrixH / PE;
//cout<<"NF:"<<NF<<endl;
  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks
  unsigned const  SF = MatrixW / SIMD;
  //cout<<"SF:"<<NF<<endl;
  // input vector buffers
  TI  inputBuf[SF];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1
  // accumulators
  decltype(activation.init(0,0))  accu[1][PE];
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0
  // unpacked and packed buffers for weight stream
  Weights_Tile<SIMD, TW, PE > w;
#pragma HLS ARRAY_PARTITION variable=w.m_weights complete dim=0
  ap_uint<PE * SIMD * TW::width> W_packed;


  unsigned  nf   = 0;
  unsigned  sf   = 0;
  unsigned  tile = 0; // invariant: tile = nf*SF + sf
  
  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  unsigned const TOTAL_FOLD = NF * SF;

  cout<<"reps:"<<reps<<endl;
  cout<<"TOTAL_FOLD:"<<TOTAL_FOLD<<endl;
  //cout<<"reps * TOTAL_FOLD:"<<reps * TOTAL_FOLD<<endl;
  for(unsigned  i = 0; i < reps * TOTAL_FOLD; i++) {
#pragma HLS PIPELINE II=1
    TI  inElem;

    if(nf == 0) {
      // read input from stream
      inElem = in.read();
      // store in appropriate buffer for reuse
      inputBuf[sf] = inElem;
    }
    else {
      // reuse buffered input
      inElem = inputBuf[sf];
    }

    // read from the parameter stream
    W_packed = weight.read();
   //if(i<reps * TOTAL_FOLD-9)
   // 	weight.write(W_packed);
    //cout<<"W_packed:"<<W_packed<<endl;
    for (unsigned pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
      w.m_weights[pe] = W_packed((pe+1)*SIMD*TW::width-1,pe*SIMD*TW::width);
    }

    // Threshold Initialisation
    if(sf == 0) {
      for(unsigned pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
      accu[0][pe] = activation.init(nf, pe);
      }
    }

    // compute matrix-vector product for each processing element
    for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
      auto const  act = TSrcI()(inElem, 0);
      auto const  wgt = TWeightI()(w[pe]);
      //auto const  wgt = w[pe];
      accu[0][pe] = mac<SIMD>(accu[0][pe], wgt, act, r, 0);
    }

    // keep track of which folded synapse/neuron we are processing
    ++tile;
    if(++sf == SF) {
      // produce output and clear accumulators
      auto  outElem = TDstI().template operator()<TO>();
      for (unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
      outElem(pe,0,1) = activation.activate(nf, pe, accu[0][pe]);
          }

      out.write(outElem);

      // next folded neuron or image
      sf = 0;
      if(++nf == NF) {
      nf   = 0;
      tile = 0;
      }
    }
  }////
}

#endif
