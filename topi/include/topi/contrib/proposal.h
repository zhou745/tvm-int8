/*!
 *  Copyright (c) 2019 by Wuwei Lin
 * \file topi/contrib/proposal.h
 * \brief Proposal
 */
#ifndef TOPI_CONTRIB_PROPOSAL_H_
#define TOPI_CONTRIB_PROPOSAL_H_

#include "tvm/tvm.h"
#include "topi/detail/extern.h"

namespace topi {
using namespace tvm;
using namespace topi::detail;

inline void UnpackArray(Array<Expr>* target, const Array<Expr>& src) {
  target->push_back(Expr(src.size()));
  for (auto&& v : src) {
    target->push_back(Expr(v));
  }
}

inline Tensor proposal(const Tensor& cls_prob, const Tensor& bbox_pred, const Tensor& im_info,
                                                    Array<Expr> scales,
                             Array<Expr> ratios,
                             int feature_stride,
                             double threshold,
                             int rpn_pre_nms_top_n,
                             int rpn_post_nms_top_n,
                             int rpn_min_size,
                             bool iou_loss) {
  return make_extern(
    {{cls_prob->shape[0] * rpn_post_nms_top_n, 5}},
    {cls_prob->dtype},
    {cls_prob, bbox_pred, im_info},
    [&](Array<Buffer> ins, Array<Buffer> outs) {
      Array<Expr> args {
          Expr("tvm.contrib.mxnet.proposal"),
          pack_buffer(ins[0]),
          pack_buffer(ins[1]),
          pack_buffer(ins[2]),
          pack_buffer(outs[0]),
          Expr(feature_stride),
          Expr(threshold),
          Expr(rpn_pre_nms_top_n),
          Expr(rpn_post_nms_top_n),
          Expr(rpn_min_size),
          Expr(iou_loss)
      };
      UnpackArray(&args, scales);
      UnpackArray(&args, ratios);

      return call_packed(args);
    }, "proposal", "", {})[0];
}

}  // namespace topi

#endif  // TOPI_CONTRIB_PROPOSAL_H_
