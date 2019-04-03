#include <stdio.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <tvm/relay/base.h>
#include <chrono>
#include <vector>
#include "./cuda_utils.h"
#include "./op_base.h"

namespace tvm {
namespace contrib {
namespace mxnet {

struct PostDetectionSign : public OpSignature {
  DLTensor* cls_prob;
  double thresh;
  double nms_thresh_lo, nms_thresh_hi;

  explicit PostDetectionSign(const TVMArgs& args) :
      cls_prob(args[1]),
      thresh(args[6]),
      nms_thresh_lo(args[7]),
      nms_thresh_hi(args[8]) {
    AddSign(cls_prob);
    AddSign(thresh);
    AddSign(nms_thresh_lo);
    AddSign(nms_thresh_hi);
  }
};

using DType = float;
using mshadow::gpu;

class PostDetectionOp : public ExternalOpBase<PostDetectionOp, PostDetectionSign> {
 public:
  using BaseType = ExternalOpBase<PostDetectionOp, PostDetectionSign>;
  using BaseType::Get;
  friend BaseType;

  void Forward(const mshadow::Tensor<gpu, 2, DType>& rois,
               const mshadow::Tensor<gpu, 3, DType>& scores,
               const mshadow::Tensor<gpu, 3, DType>& bbox_deltas,
               const mshadow::Tensor<gpu, 2, DType>& im_info,
               const mshadow::Tensor<gpu, 3, DType>& batch_boxes,
               const mshadow::Tensor<gpu, 2, DType>& batch_boxes_rois);

  ~PostDetectionOp();

 private:
  PostDetectionOp(const PostDetectionSign& params);

  int B, N, C;
  DType* out_batch_boxes;
  DType* out_batch_boxes_rois;
  DType* _pred_boxes_cu;
  DType* _pred_boxes;
  DType* _enhance_scores;
  DType* _enhance_scores_cu;
  int* _keep;
  int* _class;

  DType* _keep_score;
  DType* _boxes_batch;
  DType* _score_batch;
  float* _score_batch_copy;
  int* _class_batch;
  int* _order_batch;
  int* cls_next;
  DType* _keep_out;
  double thresh;
  double nms_thresh_lo;
  double nms_thresh_hi;
};

}  // namespace mxnet
}  // namespace contrib
}  // namespace tvm
