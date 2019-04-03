/*!
 *  Copyright (c) 2019 by Wuwei Lin
 * \file post_detection.cc
 * \brief Post Detection
 */
#include "./post_detection.h"
#include <tvm/relay/attrs/vision.h>
#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <topi/contrib/post_detection.h>
#include "./cuda_utils.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(PostDetectionAttrs);

bool PostDetectionRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 5);
  const auto* rois = types[0].as<TensorTypeNode>();
  const auto* cls_prob = types[1].as<TensorTypeNode>();
  const auto* bbox_pred = types[2].as<TensorTypeNode>();
  const auto* im_info = types[3].as<TensorTypeNode>();

  if (!rois || !cls_prob || !bbox_pred || !im_info) {
    return false;
  }

  CHECK_EQ(rois->shape.size(), 2);
  CHECK_EQ(cls_prob->shape.size(), 3);
  CHECK_EQ(bbox_pred->shape.size(), 3);
  CHECK_EQ(im_info->shape.size(), 2);

  auto batch = cls_prob->shape[0];
  auto img_rois = cls_prob->shape[1];
  auto num_cls = cls_prob->shape[2];

  auto dtype = cls_prob->dtype;
  auto out_bboxes = TensorTypeNode::make(Array<IndexExpr>{batch, img_rois, IndexExpr(6)}, dtype);
  auto out_rois = TensorTypeNode::make(Array<IndexExpr>{batch * img_rois, IndexExpr(5)}, dtype);
  reporter->Assign(types[4], TupleTypeNode::make({out_bboxes, out_rois}));

  return true;
}

Expr MakePostDetection(Expr rois, Expr cls_prob, Expr bbox_pred, Expr im_info, double thresh,
                       double nms_thresh_lo, double nms_thresh_hi) {
  auto attrs = make_node<PostDetectionAttrs>();
  attrs->thresh = thresh;
  attrs->nms_thresh_lo = nms_thresh_lo;
  attrs->nms_thresh_hi = nms_thresh_hi;
  static const Op& op = Op::Get("vision.post_detection");
  return CallNode::make(op, {rois, cls_prob, bbox_pred, im_info}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.vision._make.post_detection")
.set_body([](const TVMArgs args, TVMRetValue* rv) {
  runtime::detail::unpack_call<Expr, 7>(MakePostDetection, args, rv);
});

RELAY_REGISTER_OP("vision.post_detection")
.describe(R"code(Post detection.
see https://github.com/TuSimple/mxnet/blob/master/src/operator/post_detection_op-inl.h
)code" TVM_ADD_FILELINE)
.set_num_inputs(4)
.add_argument("rois", "Tensor", "Rois")
.add_argument("cls_prob", "Tensor", "Score of how likely proposal is object")
.add_argument("bbox_pred", "Tensor", "BBox predicted deltas from anchors for proposals")
.add_argument("im_info", "Tensor", "Image size and scale")
.set_support_level(5)
.add_type_rel("PostDetection", PostDetectionRel)
.set_attr<FTVMCompute>("FTVMCompute",
         [](const Attrs& attrs, const Array<Tensor>& inputs,const Type& out_dtype, const Target& target) -> Array<Tensor> { auto* param = attrs.as<PostDetectionAttrs>();
         return topi::post_detection(inputs[0], inputs[1], inputs[2], inputs[3], param->thresh, param->nms_thresh_lo, param->nms_thresh_hi);});

}  // namespace relay

namespace contrib {
namespace mxnet {

TVM_REGISTER_GLOBAL("tvm.contrib.mxnet.post_detection")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* rois = args[0];
  DLTensor* cls_prob = args[1];
  DLTensor* bbox_delta = args[2];
  DLTensor* im_info = args[3];
  DLTensor* out_boxes = args[4];
  DLTensor* out_rois = args[5];

  auto rois_tensor = ToTensor<mshadow::gpu, 2, float>(rois);
  auto cls_prob_tensor = ToTensor<mshadow::gpu, 3, float>(cls_prob);
  auto bbox_delta_tensor = ToTensor<mshadow::gpu, 3, float>(bbox_delta);
  auto im_info_tensor = ToTensor<mshadow::gpu, 2, float>(im_info);
  auto out_boxes_tensor = ToTensor<mshadow::gpu, 3, float>(out_boxes);
  auto out_rois_tensor = ToTensor<mshadow::gpu, 2, float>(out_rois);

  PostDetectionOp::Get(PostDetectionSign(args))
      ->Forward(rois_tensor, cls_prob_tensor, bbox_delta_tensor, im_info_tensor,
                out_boxes_tensor, out_rois_tensor);
});

}  // namespace mxnet
}  // namespace contrib
}  // namespace tvm
