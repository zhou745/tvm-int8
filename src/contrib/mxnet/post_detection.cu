#include "./post_detection.h"

#include <stdio.h>
#include <chrono>

#define CUDA_SAFE_CALL(call)                                              \
  {                                                                       \
    const cudaError_t error = call;                                       \
    if (error != cudaSuccess) {                                           \
      printf("Error: %s: %d, ", __FILE__, __LINE__);                      \
      printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                            \
    }                                                                     \
  }

// N = n rois
// B = batch size
// C = classes

namespace tvm {
namespace contrib {
namespace mxnet {

namespace cuda {
//  [in] boxes (rois), shape = (batch_size * N, 5)
//  [in] bbox_deltas, shape = (batch_size, N, 28)
// [out] _pred_boxes, shape = (N*B*(4*C))
template <typename DType>
__global__ void nonlinear_clip_kernel(DType* boxes,       // B, N, 4
                                      DType* box_deltas,  // B, N, C, 4
                                      DType* pred_boxes,  // B, N, C, 4
                                      const DType im_w, const DType im_h) {
  DType* this_box = boxes + blockIdx.x * 5;

  int b_ind = blockIdx.x * blockDim.x;

  DType* this_delta = box_deltas + b_ind * 4 + threadIdx.x * 4;
  DType* this_pred = pred_boxes + b_ind * 4 + threadIdx.x * 4;

  DType w = this_box[3] - this_box[1] + 1.f;
  DType h = this_box[4] - this_box[2] + 1.f;
  DType cx = this_box[1] + 0.5f * (w - 1.f);
  DType cy = this_box[2] + 0.5f * (h - 1.f);

  // nonlinear
  DType pred_cx = this_delta[0] * w + cx;
  DType pred_cy = this_delta[1] * h + cy;
  DType pred_w = ::exp(this_delta[2]) * w;
  DType pred_h = ::exp(this_delta[3]) * h;
  // clip
  this_pred[0] = ::max(::min(pred_cx - 0.5f * (pred_w - 1.0f), im_w - 1.0f), 0.f);
  this_pred[1] = ::max(::min(pred_cy - 0.5f * (pred_h - 1.0f), im_h - 1.0f), 0.f);
  this_pred[2] = ::max(::min(pred_cx + 0.5f * (pred_w - 1.0f), im_w - 1.0f), 0.f);
  this_pred[3] = ::max(::min(pred_cy + 0.5f * (pred_h - 1.0f), im_h - 1.0f), 0.f);
}

template <typename DType>
__global__ void _fore_back_enhance_kernel(const DType* scores,    // B * N * C
                                          DType* enhanced_score,  // B * N * C
                                          const int C) {
  int index = (blockIdx.x * blockDim.x + threadIdx.x) * C;

  DType max_val = 0.f;
  for (int c = 0; c < C; c++) {
    max_val = max_val > scores[index + c] ? max_val : scores[index + c];
  }

  enhanced_score[index] = scores[index];
  DType sum_val = enhanced_score[index];
  for (int c = 1; c < C; c++) {
    enhanced_score[index + c] = scores[index + c] >= max_val ? scores[index + c] : (DType)0.;
    sum_val += enhanced_score[index + c];
  }
  for (int c = 0; c < C; c++) {
    enhanced_score[index + c] /= sum_val;
  }
}

template <typename DType>
int weighted_nms(const DType* boxes, const DType* scores, const int* cls, const int* _order,
                 const int n_box_this_batch, const float thresh_lo, const float thresh_hi,
                 DType* keep_out) {
  DType* areas = new DType[n_box_this_batch];
  for (int i = 0; i < n_box_this_batch; i++) {
    areas[i] = (boxes[4 * i + 2] - boxes[4 * i + 0] + (DType)1.0) *
               (boxes[4 * i + 3] - boxes[4 * i + 1] + (DType)1.0);
  }
  DType* ovr = new DType[n_box_this_batch];
  int keep_num = 0;
  std::vector<int> order(_order, _order + n_box_this_batch);
  std::vector<int> inds;
  inds.reserve(n_box_this_batch);
  while (order.size() > 0) {
    inds.clear();
    int i = order[0];
    DType x1 = boxes[4 * i + 0];
    DType x2 = boxes[4 * i + 2];
    DType y1 = boxes[4 * i + 1];
    DType y2 = boxes[4 * i + 3];
    DType tmp = 0.0;
    DType avg_x1 = 0.0, avg_x2 = 0.0, avg_y1 = 0.0, avg_y2 = 0.0;
    for (unsigned int j = 0; j < order.size(); j++) {
      int oj = order[j];
      DType xx1 = std::max((DType)x1, boxes[4 * oj + 0]);
      DType xx2 = std::min((DType)x2, boxes[4 * oj + 2]);
      DType yy1 = std::max((DType)y1, boxes[4 * oj + 1]);
      DType yy2 = std::min((DType)y2, boxes[4 * oj + 3]);
      DType w = std::max((DType)0.0, xx2 - xx1 + (DType)1.0);
      DType h = std::max((DType)0.0, yy2 - yy1 + (DType)1.0);
      DType inter = w * h;
      ovr[j] = inter / (areas[i] + areas[oj] - inter);  // iou
      if (ovr[j] <= thresh_lo) {
        inds.push_back(oj);
      } else if (ovr[j] > thresh_hi) {
        DType score_j = scores[j];
        tmp += score_j;
        avg_x1 += score_j * boxes[4 * oj + 0];
        avg_x2 += score_j * boxes[4 * oj + 2];
        avg_y1 += score_j * boxes[4 * oj + 1];
        avg_y2 += score_j * boxes[4 * oj + 3];
      }
    }
    if (tmp == 0.0) break;
    keep_out[keep_num * 6 + 0] = avg_x1 / tmp;
    keep_out[keep_num * 6 + 1] = avg_y1 / tmp;
    keep_out[keep_num * 6 + 2] = avg_x2 / tmp;
    keep_out[keep_num * 6 + 3] = avg_y2 / tmp;
    keep_out[keep_num * 6 + 4] = scores[i];
    keep_out[keep_num * 6 + 5] = cls[i];
    keep_num++;
    order.clear();
    order.swap(inds);
  }  // while( n_keep > 0 )
  delete[] areas;
  delete[] ovr;
  return keep_num;
}
}  // namespace cuda

PostDetectionOp::PostDetectionOp(const PostDetectionSign& param) {
  auto scores_shape = param.cls_prob->shape;
  thresh = param.thresh;
  nms_thresh_lo = param.nms_thresh_lo;
  nms_thresh_hi = param.nms_thresh_hi;

  B = scores_shape[0];                          // batches
  N = scores_shape[1];                          // rois
  C = scores_shape[2];                          // classes
  out_batch_boxes = new DType[B * N * 6];       // batch_boxes.dptr_;
  out_batch_boxes_rois = new DType[B * N * 5];  // batch_boxes_rois.dptr_;
  CUDA_SAFE_CALL(cudaMalloc(&_pred_boxes_cu, N * B * (4 * C) * sizeof(DType)));

  _pred_boxes = new DType[N * B * (4 * C)];
  _enhance_scores = new DType[N * B * C];
  CUDA_SAFE_CALL(cudaMalloc(&_enhance_scores_cu, N * B * C * sizeof(DType)));
  _keep = new int[N];
  _class = new int[N];

  int n_box_this_batch = N;
  _keep_score = new DType[N];  // store the score of each box;
  _boxes_batch = new DType[n_box_this_batch * 4];
  _score_batch = new DType[n_box_this_batch];
  _score_batch_copy = new float[n_box_this_batch];
  _class_batch = new int[n_box_this_batch];
  _order_batch = new int[n_box_this_batch];
  cls_next = new int[C];
  _keep_out = new DType[6 * N];
}

void PostDetectionOp::Forward(const mshadow::Tensor<gpu, 2, DType>& rois,
                              const mshadow::Tensor<gpu, 3, DType>& scores,
                              const mshadow::Tensor<gpu, 3, DType>& bbox_deltas,
                              const mshadow::Tensor<gpu, 2, DType>& im_info,
                              const mshadow::Tensor<gpu, 3, DType>& batch_boxes,
                              const mshadow::Tensor<gpu, 2, DType>& batch_boxes_rois) {
  memset(out_batch_boxes, 0, (B * N * 6) * sizeof(DType));
  memset(out_batch_boxes_rois, 0, (B * N * 5) * sizeof(DType));

  std::vector<DType> cpu_im_info(3);
  CUDA_SAFE_CALL(cudaMemcpy(cpu_im_info.data(), im_info.dptr_, sizeof(DType) * cpu_im_info.size(),
                            cudaMemcpyDeviceToHost));

  DType im_h = cpu_im_info[0];
  DType im_w = cpu_im_info[1];
  cuda::nonlinear_clip_kernel<<<B * N, C>>>(rois.dptr_, bbox_deltas.dptr_,  // in
                                            _pred_boxes_cu,                 // out
                                            im_w, im_h);                    // params

  CUDA_SAFE_CALL(cudaMemcpy(_pred_boxes, _pred_boxes_cu, sizeof(DType) * bbox_deltas.shape_.Size(),
                            cudaMemcpyDeviceToHost));
  cuda::_fore_back_enhance_kernel<<<B, N>>>(scores.dptr_, _enhance_scores_cu, C);
  CUDA_SAFE_CALL(cudaMemcpy(_enhance_scores, _enhance_scores_cu, sizeof(DType) * N * B * C,
                            cudaMemcpyDeviceToHost));
  // prepare data for NMS
  int n_box_this_batch = N;

  for (int b = 0; b < B; b++) {
    memset(_keep, 0, N * sizeof(int));  // init as 0
    n_box_this_batch = 0;
    for (int c = 1; c < C; c++) {  // skip background
      for (int n = 0; n < N; n++) {
        int idx = b * N * C + n * C + c;
        if (_enhance_scores[idx] > thresh) {
          _keep[n] = 1;
          _keep_score[n] = _enhance_scores[idx];
          _class[n] = c;
          n_box_this_batch += 1;
        }
      }
    }
    int box_batch_idx = 0;
    for (int n = 0; n < N; n++) {
      if (_keep[n] == 1) {
        int keep_idx = box_batch_idx * 4;
        int class_idx = _class[n];
        int actual_idx = b * (4 * C * N) + n * 4 * C + 4 * class_idx;
        for (int bb = 0; bb < 4; bb++) {
          _boxes_batch[keep_idx + bb] = _pred_boxes[actual_idx + bb];
        }
        _score_batch[box_batch_idx] = _keep_score[n];
        _score_batch_copy[box_batch_idx] = _keep_score[n];
        _order_batch[box_batch_idx] = box_batch_idx;
        _class_batch[box_batch_idx] = _class[n];
        box_batch_idx++;
      }
    }

    // argsort
    thrust::stable_sort_by_key(thrust::host, _score_batch_copy,
                               _score_batch_copy + n_box_this_batch, _order_batch,
                               thrust::greater<float>());
    // nms for this batch
    // prepare output variables
    int keep_num = cuda::weighted_nms(_boxes_batch, _score_batch, _class_batch, _order_batch,
                                      n_box_this_batch, nms_thresh_lo, nms_thresh_hi, _keep_out);

    for (int k = 0; k < keep_num; k++) {
      int out_idx = b * N + k;
      out_batch_boxes_rois[out_idx * 5 + 0] = b;
      for (int l = 0; l < 6; l++) {
        out_batch_boxes[out_idx * 6 + l] = _keep_out[k * 6 + l];  // x1, y1, x2, y2, score, cls
        if (l < 4) {
          out_batch_boxes_rois[out_idx * 5 + l + 1] = _keep_out[k * 6 + l];  // b, x1, y1, x2, y2,
        }
      }
    }
    CUDA_SAFE_CALL(cudaMemcpy(batch_boxes.dptr_, out_batch_boxes,
                              sizeof(DType) * batch_boxes.shape_.Size(), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(batch_boxes_rois.dptr_, out_batch_boxes_rois,
                              sizeof(DType) * batch_boxes_rois.shape_.Size(),
                              cudaMemcpyHostToDevice));
  }  // end of iterate through batch
}

PostDetectionOp::~PostDetectionOp() {
  CUDA_SAFE_CALL(cudaFree(_pred_boxes_cu));
  CUDA_SAFE_CALL(cudaFree(_enhance_scores_cu));
  delete[] cls_next;
  delete[] _keep_out;
  delete[] _boxes_batch;
  delete[] _score_batch;
  delete[] _score_batch_copy;
  delete[] _order_batch;
  delete[] _keep;
  delete[] _class;
  delete[] _pred_boxes;
  delete[] _enhance_scores;
  delete[] out_batch_boxes;
  delete[] out_batch_boxes_rois;
}

}  // namespace mxnet
}  // namespace contrib
}  // namespace tvm
