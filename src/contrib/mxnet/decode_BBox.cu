/*!
 *  Copyright (c) 2019 by Wuwei Lin
 * \file proposal.cu
 * \brief Proposal
 */
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file proposal.cu
 * \brief Proposal Operator
 * \author Shaoqing Ren, Jian Guo, Pengfei Chen, Yuntao Chen
*/
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include <tvm/relay/base.h>
#include <math.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <iostream>

#include "./decode_BBox.h"
#include "./decode_BBox-inl.h"

#define THREAD_PER_BLOCK 256
#define FRCNN_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
} while (0)

using namespace mshadow::cuda;

namespace tvm {
namespace contrib {
namespace mxnet {


using tvm::relay::IndexExpr;

// bbox prediction and clip to the image borders
template<typename DType>
__global__ void BBoxTransformInv(DType* boxes,
                                 DType* bbox_deltas,
                                 const int count;
                                 const int rois_num;
                                 const int num_class;
                                 const DType* bbox_mean,
                                 const DType* bbox_std,
                                 const bool class_agnostic,
                                 const DType* im_info,
                                 DType* out) {

  int bidx = blockIdx.x;
  int tidx = threadIdx.x;
  int cidx = bidx*THREAD_PER_BLOCK+tidx;
  
  //compute if cidx is less than count
  if(cidx<count){
      int n = cidx/(rois_num*num_class);
      int index = cidx%(rois_num*num_class)/num_class;
      int cls = cidx%num_class;

      float width = boxes[n][index][2] - boxes[n][index][0] + 1.0f;
      float height = boxes[n][index][3] - boxes[n][index][1] + 1.0f;
      float ctr_x = boxes[n][index][0] + 0.5f * (width - 1.0f);
      float ctr_y = boxes[n][index][1] + 0.5f * (height - 1.0f);

      int decode_cls = class_agnostic ? 1 : cls;
      float dx = bbox_deltas[n][index][decode_cls*4+0] * bbox_std[0] + bbox_mean[0];
      float dy = bbox_deltas[n][index][decode_cls*4+1] * bbox_std[1] + bbox_mean[1];
      float dw = bbox_deltas[n][index][decode_cls*4+2] * bbox_std[2] + bbox_mean[2];
      float dh = bbox_deltas[n][index][decode_cls*4+3] * bbox_std[3] + bbox_mean[3];

      float pred_ctr_x = dx * width + ctr_x;
      float pred_ctr_y = dy * height + ctr_y;
      float pred_w = exp(dw) * width;
      float pred_h = exp(dh) * height;

      float pred_x1 = pred_ctr_x - 0.5f * (pred_w - 1.0f);
      float pred_y1 = pred_ctr_y - 0.5f * (pred_h - 1.0f);
      float pred_x2 = pred_ctr_x + 0.5f * (pred_w - 1.0f);
      float pred_y2 = pred_ctr_y + 0.5f * (pred_h - 1.0f);
      
      pred_x1 = pred_x1<im_info[n][1] - 1.0f?pred_x1:im_info[n][1] - 1.0f;
      pred_y1 = pred_y1<im_info[n][0] - 1.0f?pred_y1:im_info[n][0] - 1.0f;
      pred_x2 = pred_x2<im_info[n][1] - 1.0f?pred_x2:im_info[n][1] - 1.0f;
      pred_y2 = pred_y2<im_info[n][0] - 1.0f?pred_y2:im_info[n][0] - 1.0f;

      pred_x1 = pred_x1>0.0f?pred_x1:0.0f;
      pred_y1 = pred_y1>0.0f?pred_y1:0.0f;
      pred_x2 = pred_x2>0.0f?pred_x2:0.0f;
      pred_y2 = pred_y2>0.0f?pred_y2:0.0f;

      out[n][index][cls*4+0] = pred_x1;
      out[n][index][cls*4+1] = pred_y1;
      out[n][index][cls*4+2] = pred_x2;
      out[n][index][cls*4+3] = pred_y2;
  }

}

Decode_BBoxOp::Decode_BBoxOp(const Decode_BBoxSign& param) {
    auto boxes_shape = param.rois->shape;
    nbatch = boxes_shape[0];
    class_agnostic = param.class_agnostic;
    bbox_mean = std::move(param.bbox_mean);
    bbox_std = std::move(param.bbox_std);
    bbox_mean_gpu = mshadow::NewTensor<gpu, float, 1>(Shape1(4), 0.);
    bbox_std_gpu = mshadow::NewTensor<gpu, float, 1>(Shape1(4), 0);
}

void Decode_BBoxOp::Forward(
             mshadow::Tensor<gpu, 3, float>& boxes,
             mshadow::Tensor<gpu, 3, float>& bbox_deltas,
             mshadow::Tensor<gpu, 2, float>& im_info,
             mshadow::Tensor<gpu, 3, float>& out) {
  //copy bbox_mean and bbox_std to gpu
  FRCNN_CUDA_CHECK(cudaMemcpy(bbox_mean.data(),
                              bbox_mean_gpu.dptr_,
                              sizeof(float) * bbox_mean.size(),
                              cudaMemcpyHostToDevice));
    
  
  FRCNN_CUDA_CHECK(cudaMemcpy(bbox_std.data(),
                              bbox_std_gpu.dptr_,
                              sizeof(float) * bbox_std.size(),
                              cudaMemcpyHostToDevice));

  //decode bbox
  int rois_num = boxes.size(1);
  int num_class = class_agnostic ? 1 : (bbox_deltas.size(2) / 4);
  int count = nbatch*rois_num*num_class;

  dim3 dimGrid((count + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
  dim3 dimBlock(THREAD_PER_BLOCK);
  BBoxTransformInv<<<dimGrid, dimBlock>>>(boxes.dptr_, bbox_deltas.dptr_, count, rois_num,num_class,
                                          bbox_mean.dptr_, bbox_std.dptr_, class_agnostic,
                                          im_info.dptr_, out.dptr_);

  FRCNN_CUDA_CHECK(cudaPeekAtLastError());
  }
}

ProposalGPUOp::~ProposalGPUOp() {
  mshadow::FreeSpace(&bbox_mean_gpu);
  mshadow::FreeSpace(&bbox_std_gpu);
}


}  // namespace mxnet
}  // namespace contrib
}  // namespace tvm
