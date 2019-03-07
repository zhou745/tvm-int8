/*!
 *  Copyright (c) 2019 by Wuwei Lin
 * \file proposal.h
 * \brief Proposal
 */
#ifndef TVM_CONTRIB_MXNET_PROPOSAL_H_
#define TVM_CONTRIB_MXNET_PROPOSAL_H_

#include <tvm/relay/attrs/vision.h>
#include <vector>
#include "./cuda_utils.h"
#include "./op_base.h"

namespace tvm {
namespace contrib {
namespace mxnet {


using mshadow::gpu;
using namespace mshadow;
using namespace mshadow::expr;

inline std::vector<float> GetFloatVector(const TVMArgs& args, int index) {
  size_t size = args[index];
  std::vector<float> result;
  result.reserve(size);
  while (size-- > 0) {
    result.push_back(static_cast<double>(args[++index]));
  }
  return result;
}

struct ProposalSign : public OpSignature {
  DLTensor* cls_prob;
  std::vector<float> scales;
  std::vector<float> ratios;
  int feature_stride;
  float threshold;
  int rpn_pre_nms_top_n;
  int rpn_post_nms_top_n;
  int rpn_min_size;
  bool iou_loss;

  explicit ProposalSign(const TVMArgs& args) :
    cls_prob(args[0]),
    feature_stride(args[4]),
    threshold(static_cast<float>(static_cast<double>(args[5]))),
    rpn_pre_nms_top_n(args[6]),
    rpn_post_nms_top_n(args[7]),
    rpn_min_size(args[8]),
    iou_loss(args[9]) {

    this->scales = GetFloatVector(args, 10);
    this->ratios = GetFloatVector(args, 11 + this->scales.size());

    Reserve(20);
    AddSign(cls_prob);
    AddSign(feature_stride);
    AddSign(threshold);
    AddSign(rpn_pre_nms_top_n);
    AddSign(rpn_post_nms_top_n);
    AddSign(rpn_min_size);
    AddSign(iou_loss);
    AddSign(scales);
    AddSign(ratios);
  }
};

class ProposalGPUOp : public ExternalOpBase<ProposalGPUOp, ProposalSign> {
 public:
  using BaseType = ExternalOpBase<ProposalGPUOp, ProposalSign>;
  using BaseType::Get;
  friend BaseType;

  void Forward(
        // batch_idx, anchor_idx, height_idx, width_idx
        mshadow::Tensor<gpu, 4, float>& scores,
        // batch_idx, height_idx, width_idx, anchor_idx
        mshadow::Tensor<gpu, 4, float>& bbox_deltas,
        // batch_idx, 3(height, width, scale)
        mshadow::Tensor<gpu, 2, float>& im_info,
        // batch_idx, rois_idx, 5(batch_idx, x1, y1, x2, y2), batch_idx is needed after flatten
        mshadow::Tensor<gpu, 2, float>& out);

  ~ProposalGPUOp();

 private:
  explicit ProposalGPUOp(const ProposalSign& param);

  int nbatch_;
  int num_anchors_;
  int height_;
  int width_;
  int count_;
  int rpn_pre_nms_top_n_;
  int rpn_post_nms_top_n_;
  int rpn_min_size_;
  int feature_stride_;
  double threshold_;
  bool iou_loss_;
  std::vector<float> anchors_;
  std::vector<float> scales_, ratios_;
  std::vector<float> cpu_im_info_;
  Shape<3> fg_scores_shape_;
  std::vector<int> _keep_;
  mshadow::Tensor<gpu, 1, int> keep_;
  mshadow::Tensor<gpu, 3> proposals_;
  mshadow::Tensor<gpu, 1> score_;
  mshadow::Tensor<gpu, 1, int> order_;
  mshadow::Tensor<gpu, 2> ordered_proposals_;
  mshadow::Tensor<gpu, 1, uint64_t> mask_tensor_;
  mshadow::Tensor<cpu, 1, uint64_t> mask_host_tensor_;
};  // class ProposalGPUOp


}  // namespace mxnet
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MXNET_PROPOSAL_H_
