/*!
 *  Copyright (c) 2019 by Wuwei Lin
 * \file proposal.h
 * \brief Proposal
 */
#ifndef TVM_CONTRIB_MXNET_DECODE_BBOX_H_
#define TVM_CONTRIB_MXNET_DECODE_BBOX_H_

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

inline std::vector<float> GetFloatVector_BBox(const TVMArgs& args, int index) {
  size_t size = args[index];
  std::vector<float> result;
  result.reserve(size);
  while (size-- > 0) {
    result.push_back(static_cast<double>(args[++index]));
  }
  return result;
}

struct Decode_BBoxSign : public OpSignature {
  DLTensor* rois;
  DLTensor* bbox_pred;
  DLTensor* im_info;
  bool class_agnostic;
  std::vector<float> bbox_mean;
  std::vector<float> bbox_std;


  explicit Decode_BBoxSign(const TVMArgs& args) :
    rois(args[0]),
    bbox_pred(args[1]),
    im_info(args[2]),
    class_agnostic(args[4]) {
    this->bbox_mean = GetFloatVector_BBox(args, 5);
    this->bbox_std = GetFloatVector_BBox(args, 6 + this->bbox_mean.size());

    Reserve(24);
    AddSign(rois);
    AddSign(bbox_pred);
    AddSign(im_info);
    AddSign(class_agnostic);
    AddSign(bbox_mean);
    AddSign(bbox_std);
  }
};

class Decode_BBoxOp : public ExternalOpBase<Decode_BBoxOp, Decode_BBoxSign> {
 public:
  using BaseType = ExternalOpBase<Decode_BBoxOp, Decode_BBoxSign>;
  using BaseType::Get;
  friend BaseType;
  void Forward(
        // batch_idx, anchor_idx, height_idx, width_idx
        mshadow::Tensor<gpu, 2, float>& boxes,
        // batch_idx, height_idx, width_idx, anchor_idx
        mshadow::Tensor<gpu, 3, float>& bbox_deltas,
        // batch_idx, 3(height, width, scale)
        mshadow::Tensor<gpu, 2, float>& im_info,
        // batch_idx, rois_idx, 5(batch_idx, x1, y1, x2, y2), batch_idx is needed after flatten
        mshadow::Tensor<gpu, 3, float>& out);

  ~Decode_BBoxOp();

 private:
  explicit Decode_BBoxOp(const Decode_BBoxSign& param);

  //parameters
  std::vector<float> bbox_mean, bbox_std;
  int nbatch;
  bool class_agnostic;

  mshadow::Tensor<gpu, 1> bbox_mean_gpu;
  mshadow::Tensor<gpu, 1> bbox_std_gpu;
};  // class ProposalGPUOp


}  // namespace mxnet
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MXNET_DECODE_BBOX_H_
