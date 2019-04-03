/*!
 *  Copyright (c) 2019 by Wuwei Lin
 * \file topi/contrib/post_detection.h
 * \brief post_detection.h
 */
#ifndef TOPI_CONTRIB_POST_DETECTION_H_
#define TOPI_CONTRIB_POST_DETECTION_H_

#include "tvm/tvm.h"
#include "topi/detail/extern.h"

namespace topi {
using namespace tvm;
using namespace topi::detail;

inline Array<Tensor> post_detection(const Tensor& rois,
                                    const Tensor& cls_prob,
                                    const Tensor& bbox_pred,
                                    const Tensor& im_info,
                                    double thresh,
                                    double nms_thresh_lo,
                                    double nms_thresh_hi) {
  auto batch = cls_prob->shape[0];
  auto img_rois = cls_prob->shape[1];
  return make_extern(
    {{batch, img_rois, 6}, {batch*img_rois, 5}},
    {cls_prob->dtype, cls_prob->dtype},
    {rois, cls_prob, bbox_pred, im_info},
    [&](Array<Buffer> ins, Array<Buffer> outs) {
      Array<Expr> args {
          Expr("tvm.contrib.mxnet.post_detection"),
          pack_buffer(ins[0]),
          pack_buffer(ins[1]),
          pack_buffer(ins[2]),
          pack_buffer(ins[3]),
          pack_buffer(outs[0]),
          pack_buffer(outs[1]),
          Expr(thresh),
          Expr(nms_thresh_lo),
          Expr(nms_thresh_hi)
      };
      return call_packed(args);
    }, "proposal", "", {});
}

}  // namespace topi

#endif  // TOPI_CONTRIB_POST_DETECTION_H_
