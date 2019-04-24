/*!
 *  Copyright (c) 2019 by Wuwei Lin
 * \file topi/contrib/proposal.h
 * \brief Proposal
 */
#ifndef TOPI_CONTRIB_DECODE_BBOX_H_
#define TOPI_CONTRIB_DECODE_BBOX_H_

#include "tvm/tvm.h"
#include "topi/detail/extern.h"

namespace topi {
using namespace tvm;
using namespace topi::detail;

inline void UnpackArray_BBox(Array<Expr>* target, const Array<Expr>& src) {
  target->push_back(Expr(src.size()));
  for (auto&& v : src) {
    target->push_back(Expr(v));
  }
}

inline Tensor decode_BBox(const Tensor& rois, const Tensor& bbox_pred, const Tensor& im_info,
                        Array<Expr> bbox_mean,
                        Array<Expr> bbox_std,
                        bool class_agnostic) {
  if(class_agnostic){
  return make_extern(
    {{bbox_pred->shape[0],bbox_pred->shape[1],4}},
    {bbox_pred->dtype},
    {rois, bbox_pred, im_info},
    [&](Array<Buffer> ins, Array<Buffer> outs) {
      Array<Expr> args {
          Expr("tvm.contrib.mxnet.decode_BBox"),
          pack_buffer(ins[0]),
          pack_buffer(ins[1]),
          pack_buffer(ins[2]),
          pack_buffer(outs[0]),
          Expr(class_agnostic)
      };
      UnpackArray_BBox(&args, bbox_mean);
      UnpackArray_BBox(&args, bbox_std);

      return call_packed(args);
    }, "decode_BBox", "", {})[0];
  } else {
    return make_extern(
    {bbox_pred->shape},
    {bbox_pred->dtype},
    {rois, bbox_pred, im_info},
    [&](Array<Buffer> ins, Array<Buffer> outs) {
      Array<Expr> args {
          Expr("tvm.contrib.mxnet.decode_BBox"),
          pack_buffer(ins[0]),
          pack_buffer(ins[1]),
          pack_buffer(ins[2]),
          pack_buffer(outs[0]),
          Expr(class_agnostic)
      };
      UnpackArray_BBox(&args, bbox_mean);
      UnpackArray_BBox(&args, bbox_std);

      return call_packed(args);
    }, "decode_BBox", "", {})[0];
  }
  
}

}  // namespace topi

#endif  // TOPI_CONTRIB_DECODE_BBOX_H_
