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
#include <fstream>
#include "./proposal.h"
#include "./proposal-inl.h"
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

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

// scores are (b, anchor, h, w)
// proposals are (h * w * anchor, 5)
// w defines "x" and h defines "y"
// count should be total anchors numbers, h * w * anchors
template<typename Dtype>
__global__ void ProposalGridKernel(const int count,
                                   const int num_anchors,
                                   const int height,
                                   const int width,
                                   const int feature_stride,
                                   const Dtype* scores,
                                   Dtype* proposals) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % width;
    int h = index / num_anchors / width;

    proposals[index * 5 + 0] = proposals[a * 5 + 0] + w * feature_stride;
    proposals[index * 5 + 1] = proposals[a * 5 + 1] + h * feature_stride;
    proposals[index * 5 + 2] = proposals[a * 5 + 2] + w * feature_stride;
    proposals[index * 5 + 3] = proposals[a * 5 + 3] + h * feature_stride;
    proposals[index * 5 + 4] = scores[(a * height + h) * width + w];
  }
}

// boxes are (h * w * anchor, 5)
// deltas are (b, 4 * anchor, h, w)
// out_pred_boxes are (h * w * anchor, 5)
// count should be total anchors numbers, h * w * anchors
// in-place write: boxes and out_pred_boxes are the same location
template<typename Dtype>
__global__ void BBoxPredKernel(const int count,
                               const int num_anchors,
                               const int feat_height,
                               const int feat_width,
                               const int real_height,
                               const int real_width,
                               const float im_height,
                               const float im_width,
                               const Dtype* boxes,
                               const Dtype* deltas,
                               Dtype* out_pred_boxes) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % feat_width;
    int h = index / num_anchors / feat_width;

    float width = boxes[index * 5 + 2] - boxes[index * 5 + 0] + 1.0f;
    float height = boxes[index * 5 + 3] - boxes[index * 5 + 1] + 1.0f;
    float ctr_x = boxes[index * 5 + 0] + 0.5f * (width - 1.0f);
    float ctr_y = boxes[index * 5 + 1] + 0.5f * (height - 1.0f);

    float dx = deltas[((a * 4) * feat_height + h) * feat_width + w];
    float dy = deltas[((a * 4 + 1) * feat_height + h) * feat_width + w];
    float dw = deltas[((a * 4 + 2) * feat_height + h) * feat_width + w];
    float dh = deltas[((a * 4 + 3) * feat_height + h) * feat_width + w];

    float pred_ctr_x = dx * width + ctr_x;
    float pred_ctr_y = dy * height + ctr_y;
    float pred_w = ::exp(dw) * width;
    float pred_h = ::exp(dh) * height;

    float pred_x1 = pred_ctr_x - 0.5f * (pred_w - 1.0f);
    float pred_y1 = pred_ctr_y - 0.5f * (pred_h - 1.0f);
    float pred_x2 = pred_ctr_x + 0.5f * (pred_w - 1.0f);
    float pred_y2 = pred_ctr_y + 0.5f * (pred_h - 1.0f);

    pred_x1 = ::max(::min(pred_x1, im_width - 1.0f), 0.0f);
    pred_y1 = ::max(::min(pred_y1, im_height - 1.0f), 0.0f);
    pred_x2 = ::max(::min(pred_x2, im_width - 1.0f), 0.0f);
    pred_y2 = ::max(::min(pred_y2, im_height - 1.0f), 0.0f);

    out_pred_boxes[index * 5 + 0] = pred_x1;
    out_pred_boxes[index * 5 + 1] = pred_y1;
    out_pred_boxes[index * 5 + 2] = pred_x2;
    out_pred_boxes[index * 5 + 3] = pred_y2;

    if (h >= real_height || w >= real_width) {
      out_pred_boxes[index * 5 + 4] = -1.0f;
    }
  }
}

// boxes are (h * w * anchor, 5)
// deltas are (b, 4 * anchor, h, w)
// out_pred_boxes are (h * w * anchor, 5)
// count should be total anchors numbers, h * w * anchors
// in-place write: boxes and out_pred_boxes are the same location
template<typename Dtype>
__global__ void IoUPredKernel(const int count,
                              const int num_anchors,
                              const int feat_height,
                              const int feat_width,
                              const int real_height,
                              const int real_width,
                              const float im_height,
                              const float im_width,
                              const Dtype* boxes,
                              const Dtype* deltas,
                              Dtype* out_pred_boxes) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % feat_width;
    int h = index / num_anchors / feat_width;

    float x1 = boxes[index * 5 + 0];
    float y1 = boxes[index * 5 + 1];
    float x2 = boxes[index * 5 + 2];
    float y2 = boxes[index * 5 + 3];

    float dx1 = deltas[((a * 4) * feat_height + h) * feat_width + w];
    float dy1 = deltas[((a * 4 + 1) * feat_height + h) * feat_width + w];
    float dx2 = deltas[((a * 4 + 2) * feat_height + h) * feat_width + w];
    float dy2 = deltas[((a * 4 + 3) * feat_height + h) * feat_width + w];

    float pred_x1 = ::max(::min(x1 + dx1, im_width - 1.0f), 0.0f);
    float pred_y1 = ::max(::min(y1 + dy1, im_height - 1.0f), 0.0f);
    float pred_x2 = ::max(::min(x2 + dx2, im_width - 1.0f), 0.0f);
    float pred_y2 = ::max(::min(y2 + dy2, im_height - 1.0f), 0.0f);

    out_pred_boxes[index * 5 + 0] = pred_x1;
    out_pred_boxes[index * 5 + 1] = pred_y1;
    out_pred_boxes[index * 5 + 2] = pred_x2;
    out_pred_boxes[index * 5 + 3] = pred_y2;

    if (h >= real_height || w >= real_width) {
      out_pred_boxes[index * 5 + 4] = -1.0f;
    }
  }
}

// filter box with stride less than rpn_min_size
// filter: set score to zero
// dets (n, 5)
template<typename Dtype>
__global__ void FilterBoxKernel(const int count,
                                const float min_size,
                                Dtype* dets) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    float iw = dets[index * 5 + 2] - dets[index * 5 + 0] + 1.0f;
    float ih = dets[index * 5 + 3] - dets[index * 5 + 1] + 1.0f;
    if (iw < min_size || ih < min_size) {
      dets[index * 5 + 0] -= min_size / 2;
      dets[index * 5 + 1] -= min_size / 2;
      dets[index * 5 + 2] += min_size / 2;
      dets[index * 5 + 3] += min_size / 2;
      dets[index * 5 + 4] = -1.0f;
    }
  }
}

// copy score and init order
// dets (n, 5); score (n, ); order (n, )
// count should be n (total anchors or proposals)
template<typename Dtype>
__global__ void CopyScoreKernel(const int count,
                                const Dtype* dets,
                                Dtype* score,
                                int* order) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    score[index] = dets[index * 5 + 4];
    order[index] = index;
  }
}

// reorder proposals according to order and keep the top_n proposals
// prev_dets (n, 5); order (n, ); dets (n, 5)
// count should be output anchor numbers (top_n)
template<typename Dtype>
__global__ void ReorderProposalsKernel(const int count,
                                       const Dtype* prev_dets,
                                       const int* order,
                                       Dtype* dets) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    const int order_i = order[index];
    for (int j = 0; j < 5; j ++) {
      dets[index * 5 + j] = prev_dets[order_i * 5 + j];
    }
  }
}

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = ::max(a[0], b[0]), right = ::min(a[2], b[2]);
  float top = ::max(a[1], b[1]), bottom = ::min(a[3], b[3]);
  float width = ::max(right - left + 1, 0.f), height = ::max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, uint64_t *dev_mask) {
  const int threadsPerBlock = sizeof(uint64_t) * 8;
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        ::min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        ::min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    uint64_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _nms(const mshadow::Tensor<mshadow::gpu, 2>& boxes,
          const float nms_overlap_thresh,
          int *keep,
          int *num_out,
          uint64_t *mask_dev,
          uint64_t *mask_host) {
  /*
  @input  boxes: (pre_nms_top_n, 5)
  @return keep
  @return num_out
  @tmp    mask_dev
  @tmp    mask_host
  */
  const int threadsPerBlock = sizeof(uint64_t) * 8;
  const int boxes_num = boxes.size(0);
  const int boxes_dim = boxes.size(1);

  float* boxes_dev = boxes.dptr_;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);
  FRCNN_CUDA_CHECK(cudaPeekAtLastError());

  FRCNN_CUDA_CHECK(cudaMemcpy(mask_host,
                              mask_dev,
                              sizeof(uint64_t) * boxes_num * col_blocks,
                              cudaMemcpyDeviceToHost));

  std::vector<uint64_t> remv(col_blocks);
  memset(&remv[0], 0, sizeof(uint64_t) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep[num_to_keep++] = i;
      uint64_t *p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;
}

// copy proposals to output
// dets (top_n, 5); keep (top_n, ); out (top_n, )
// count should be top_n (total anchors or proposals)
template<typename Dtype>
__global__ void PrepareOutput(const int count,
                              const Dtype* dets,
                              const int* keep,
                              const int out_size,
                              const int batchIdx,
                              Dtype* out
                              ) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    out[index * 5] = batchIdx;
    if (index < out_size) {
      int keep_i = keep[index];
      for (int j = 0; j < 4; ++j) {
        out[index * 5 + j + 1] = dets[keep_i * 5 + j];
      }
    } else {
      //int keep_i = keep[index % out_size];
      for (int j = 0; j < 4; ++j) {
        out[index * 5 + j + 1] = 0.0f;
      }
    }
  }
}

ProposalGPUOp::ProposalGPUOp(const ProposalSign& param) {
    auto scores_shape = param.cls_prob->shape;

    nbatch_ = scores_shape[0];
    num_anchors_ = scores_shape[1] / 2;
    height_ = scores_shape[2];
    width_ = scores_shape[3];
    count_ = num_anchors_ * height_ * width_;  // count of total anchors
    feature_stride_ = param.feature_stride;
    threshold_ = param.threshold;
    rpn_pre_nms_top_n_ = param.rpn_pre_nms_top_n;
    rpn_post_nms_top_n_ = param.rpn_post_nms_top_n;
    rpn_min_size_ = param.rpn_min_size;
    iou_loss_ = param.iou_loss;
    scales_ = std::move(param.scales);
    ratios_ = std::move(param.ratios);


    // Generate first anchors based on base anchor
    std::vector<float> base_anchor(4);
    base_anchor[0] = 0.0;
    base_anchor[1] = 0.0;
    base_anchor[2] = feature_stride_ - 1.0;
    base_anchor[3] = feature_stride_ - 1.0;

    CHECK_EQ(num_anchors_, ratios_.size() * scales_.size());
    GenerateAnchors(base_anchor,
                    ratios_,
                    scales_,
                    &anchors_);

    fg_scores_shape_ = Shape3(scores_shape[1] / 2,
                                      scores_shape[2],
                                      scores_shape[3]);

    proposals_ = mshadow::NewTensor<gpu, float, 3>(Shape3(nbatch_, count_, 5), 0.);
    score_ = mshadow::NewTensor<gpu, float, 1>(Shape1(count_), 0.);
    order_ = mshadow::NewTensor<gpu, int, 1>(Shape1(count_), 0);

    int rpn_pre_nms_top_n = (rpn_pre_nms_top_n_ > 0) ? rpn_pre_nms_top_n_ : count_;
    rpn_pre_nms_top_n = std::min(rpn_pre_nms_top_n, count_);
    ordered_proposals_ = mshadow::NewTensor<gpu, float, 2>(Shape2(rpn_pre_nms_top_n, 5), 0.);

    _keep_.resize(rpn_pre_nms_top_n);
    keep_ = mshadow::NewTensor<gpu, int, 1>(Shape1(_keep_.size()), 0);
    cpu_im_info_.resize(3);
    const int boxes_num = rpn_pre_nms_top_n;
    const int col_blocks = DIVUP(boxes_num, sizeof(uint64_t) * 8);
    // take special care when allocate memory of 8-byte alignment.
    mask_tensor_ = mshadow::NewTensor<gpu, uint64_t, 1>(
            Shape1(boxes_num * col_blocks), uint64_t(0));
    // the following line does not need change since it the only place where requires host workspace
    mask_host_tensor_ = mshadow::NewTensor<cpu, uint64_t, 1>(
            Shape1(boxes_num * col_blocks), uint64_t(0));
  }

void ProposalGPUOp::Forward(
             mshadow::Tensor<gpu, 4, float>& scores,
             mshadow::Tensor<gpu, 4, float>& bbox_deltas,
             mshadow::Tensor<gpu, 2, float>& im_info,
             mshadow::Tensor<gpu, 2, float>& out) {
  // set to -1 for max
  int rpn_pre_nms_top_n = (rpn_pre_nms_top_n_ > 0) ? rpn_pre_nms_top_n_ : count_;
  rpn_pre_nms_top_n = std::min(rpn_pre_nms_top_n, count_);
  int rpn_post_nms_top_n = std::min(rpn_post_nms_top_n_, rpn_pre_nms_top_n);

  // im_info is small, we want to copy them to cpu
  FRCNN_CUDA_CHECK(cudaMemcpy(cpu_im_info_.data(),
                              im_info.dptr_,
                              sizeof(float) * cpu_im_info_.size(),
                              cudaMemcpyDeviceToHost));  // less than 64K

  // prevent padded predictions
  int real_height = static_cast<int>(cpu_im_info_[0] / feature_stride_);
  int real_width = static_cast<int>(cpu_im_info_[1] / feature_stride_);
  CHECK_GE(height_, real_height) << height_ << " " << real_height << std::endl;
  CHECK_GE(width_, real_width) << width_ << " " << real_width << std::endl;
  //zjq debug add
  //std::ofstream zjq_file;
  //zjq_file.open("tvm_record");

  /* copy anchors for all images in batch */
  for (int i = 0; i < nbatch_; i++) {
    float* batch_proposals = proposals_.dptr_ + i * 5 * count_;
    FRCNN_CUDA_CHECK(cudaMemcpy(batch_proposals,
                                &anchors_[0],
                                sizeof(float) * anchors_.size(),
                                cudaMemcpyHostToDevice));  // less than 64K

    /* get current batch foreground score */
    float* foreground_score_ptr = reinterpret_cast<float *>(scores.dptr_) +
        i * 2 * count_ + fg_scores_shape_.Size();
    mshadow::Tensor<gpu, 3> fg_scores = mshadow::Tensor<gpu, 3>(
            foreground_score_ptr, fg_scores_shape_);

    /* copy proposals to a mesh grid */
    dim3 dimGrid((count_ + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock);
    dim3 dimBlock(kMaxThreadsPerBlock);
    ProposalGridKernel<<<dimGrid, dimBlock>>>(
      count_, num_anchors_, height_, width_, feature_stride_,
      fg_scores.dptr_, batch_proposals);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    /* transform anchors and bbox_deltas into bboxes */
    if (iou_loss_) {
      IoUPredKernel<<<dimGrid, dimBlock>>>(
        count_, num_anchors_, height_, width_, real_height, real_width,
        cpu_im_info_[0], cpu_im_info_[1],
        batch_proposals, bbox_deltas.dptr_ + i * 4 * count_, batch_proposals);
    } else {
      BBoxPredKernel<<<dimGrid, dimBlock>>>(
        count_, num_anchors_, height_, width_, real_height, real_width,
        cpu_im_info_[0], cpu_im_info_[1],
        batch_proposals, bbox_deltas.dptr_ + i * 4 * count_, batch_proposals);
        //zjq debug add
        /*
        float temp[5*count_];
        FRCNN_CUDA_CHECK(cudaMemcpy(temp,
                                    batch_proposals,
                                    sizeof(float)*count_*5,
                                    cudaMemcpyDeviceToHost));
        zjq_file<<count_<<std::endl;
        for(int idx=0;idx<count_;idx++){
            zjq_file<<idx<<" "<<temp[idx*5]<<" "<<temp[idx*5+1]<<" "<<temp[idx*5+2]<<" "<<temp[idx*5+3]<<" "<<temp[idx*5+4]<<std::endl;
        }*/
    }
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    /* filter boxes with less than rpn_min_size */
    FilterBoxKernel<<<dimGrid, dimBlock>>>(
      count_, rpn_min_size_ * cpu_im_info_[2], batch_proposals);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    /* copy score to a continuous memory */
    CopyScoreKernel<<<dimGrid, dimBlock>>>(
      count_, batch_proposals, score_.dptr_, order_.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    /* argsort score, save order */
    thrust::stable_sort_by_key(thrust::device,
                               score_.dptr_,
                               score_.dptr_ + score_.size(0),
                               order_.dptr_,
                               thrust::greater<float>());
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());

    /* Reorder proposals according to order */
    dimGrid.x = (rpn_pre_nms_top_n + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    ReorderProposalsKernel<<<dimGrid, dimBlock>>>(
      rpn_pre_nms_top_n, batch_proposals, order_.dptr_, ordered_proposals_.dptr_);
    FRCNN_CUDA_CHECK(cudaPeekAtLastError());
    //zjq debug add
    /*float temp[5*rpn_pre_nms_top_n];
    FRCNN_CUDA_CHECK(cudaMemcpy(temp,
                                ordered_proposals_.dptr_,
                                sizeof(float)*rpn_pre_nms_top_n*5,
                                cudaMemcpyDeviceToHost));
    zjq_file<<rpn_pre_nms_top_n<<std::endl;
    zjq_file<<threshold_<<std::endl;
    for(int idx=0;idx<rpn_pre_nms_top_n;idx++){
         zjq_file<<idx<<" "<<temp[idx*5]<<" "<<temp[idx*5+1]<<" "<<temp[idx*5+2]<<" "<<temp[idx*5+3]<<" "<<temp[idx*5+4]<<std::endl;
    }*/
    /* perform nms */
    int out_size = 0;
    uint64_t *mask_dev = mask_tensor_.dptr_;
    uint64_t *mask_host = mask_host_tensor_.dptr_;
    _nms(ordered_proposals_,
         threshold_,
         &_keep_[0],
         &out_size,
         mask_dev,
         mask_host);

    /* copy nms result to gpu */
    FRCNN_CUDA_CHECK(cudaMemcpy(keep_.dptr_,
                                &_keep_[0],
                                sizeof(int) * _keep_.size(),
                                cudaMemcpyHostToDevice));  // less than 64K

    /* copy results after nms */
    dimGrid.x = (rpn_post_nms_top_n + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
    PrepareOutput<<<dimGrid, dimBlock>>>(
      rpn_post_nms_top_n, ordered_proposals_.dptr_, keep_.dptr_, out_size, i,
      out.dptr_ + i * 5 * rpn_post_nms_top_n);

    FRCNN_CUDA_CHECK(cudaPeekAtLastError());
  }
  //zjq debug add
  //zjq_file.close();
}

ProposalGPUOp::~ProposalGPUOp() {
  mshadow::FreeSpace(&score_);
  mshadow::FreeSpace(&order_);
  mshadow::FreeSpace(&proposals_);
  mshadow::FreeSpace(&ordered_proposals_);
  mshadow::FreeSpace(&mask_tensor_);
  mshadow::FreeSpace(&mask_host_tensor_);
  mshadow::FreeSpace(&keep_);
}


}  // namespace mxnet
}  // namespace contrib
}  // namespace tvm
