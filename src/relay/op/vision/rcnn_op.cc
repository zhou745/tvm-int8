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
 *  Copyright (c) 2019 by Contributors
 * \file rcnn_op.cc
 * \brief Faster RCNN and Mask RCNN operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/attrs/vision.h>
#include <topi/contrib/proposal.h>
#include <topi/contrib/decode_BBox.h>
#include "../../../contrib/mxnet/proposal.h"
#include "../../../contrib/mxnet/decode_BBox.h"
namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(ROIAlignAttrs);

bool ROIAlignRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  auto roi_align_attrs = attrs.as<ROIAlignAttrs>();
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* rois = types[1].as<TensorTypeNode>();
  const auto& dshape = data->shape;
  const auto& rshape = rois->shape;
  CHECK(roi_align_attrs);
  CHECK_EQ(dshape.size(), 4) << "Input data should be 4-D.";
  CHECK_EQ(rshape.size(), 2) << "Input rois should be 2-D.";
  CHECK_EQ(roi_align_attrs->layout, "NCHW") << "ROI Align only supports NCHW layout";
  // assign output type
  std::vector<IndexExpr> oshape(
      {rshape[0], dshape[1], roi_align_attrs->pooled_size[0], roi_align_attrs->pooled_size[1]});
  reporter->Assign(types[2], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

Expr MakeROIAlign(Expr data, Expr rois, Array<IndexExpr> pooled_size, double spatial_scale,
                  int sample_ratio, std::string layout) {
  auto attrs = make_node<ROIAlignAttrs>();
  attrs->pooled_size = pooled_size;
  attrs->spatial_scale = spatial_scale;
  attrs->sample_ratio = sample_ratio;
  attrs->layout = layout;
  static const Op& op = Op::Get("vision.roi_align");
  return CallNode::make(op, {data, rois}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.vision._make.roi_align")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 6>(MakeROIAlign, args, rv);
  });

RELAY_REGISTER_OP("vision.roi_align")
    .describe(R"doc(ROI Align operator.

 - **data**: This depends on the `layout` parameter. Input is 4D array of shape
             (batch_size, channels, height, width) if `layout` is `NCHW`.
 - **rois**: 2D array of shape (num_roi, 5). The last dimension should be in format of
             [batch_index, w_start, h_start, w_end, h_end].
 - **out**: This depends on the `layout` parameter. Output is 4D array of shape
            (num_roi, channels, pooled_height, pooled_width) if `layout` is `NCHW`.
 )doc" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("rois", "Tensor", "The input rois")
.set_support_level(5)
.add_type_rel("ROIAlign", ROIAlignRel);

TVM_REGISTER_NODE_TYPE(ROIPoolAttrs);

bool ROIPoolRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  auto roi_pool_attrs = attrs.as<ROIPoolAttrs>();
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* rois = types[1].as<TensorTypeNode>();
  const auto& dshape = data->shape;
  const auto& rshape = rois->shape;
  CHECK(roi_pool_attrs);
  CHECK_EQ(dshape.size(), 4) << "Input data should be 4-D.";
  CHECK_EQ(rshape.size(), 2) << "Input rois should be 2-D.";
  CHECK_EQ(roi_pool_attrs->layout, "NCHW") << "ROI Pool only supports NCHW layout";
  // assign output type
  std::vector<IndexExpr> oshape(
      {rshape[0], dshape[1], roi_pool_attrs->pooled_size[0], roi_pool_attrs->pooled_size[1]});
  reporter->Assign(types[2], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

Expr MakeROIPool(Expr data, Expr rois, Array<IndexExpr> pooled_size, double spatial_scale,
                 std::string layout) {
  auto attrs = make_node<ROIPoolAttrs>();
  attrs->pooled_size = pooled_size;
  attrs->spatial_scale = spatial_scale;
  attrs->layout = layout;
  static const Op& op = Op::Get("vision.roi_pool");
  return CallNode::make(op, {data, rois}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.vision._make.roi_pool")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 5>(MakeROIPool, args, rv);
  });

RELAY_REGISTER_OP("vision.roi_pool")
    .describe(R"doc(ROI Pool operator.

 - **data**: This depends on the `layout` parameter. Input is 4D array of shape
             (batch_size, channels, height, width) if `layout` is `NCHW`.
 - **rois**: 2D array of shape (num_roi, 5). The last dimension should be in format of
             [batch_index, w_start, h_start, w_end, h_end].
 - **out**: This depends on the `layout` parameter. Output is 4D array of shape
            (num_roi, channels, pooled_height, pooled_width) if `layout` is `NCHW`.
 )doc" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("rois", "Tensor", "The input rois")
.set_support_level(5)
.add_type_rel("ROIPool", ROIPoolRel);

TVM_REGISTER_NODE_TYPE(ProposalAttrs);

bool ProposalRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  auto proposal_attrs = attrs.as<ProposalAttrs>();
  CHECK_EQ(types.size(), 4);
  const auto* cls_prob = types[0].as<TensorTypeNode>();
  const auto* bbox_pred = types[1].as<TensorTypeNode>();
  const auto* im_info = types[2].as<TensorTypeNode>();

  if (!cls_prob || !bbox_pred || !im_info) {
    return false;
  }

  CHECK_EQ(cls_prob->shape.size(), 4U)
      << "The dimension of class probability should be 4, but received " << cls_prob->shape.size();
  CHECK_EQ(bbox_pred->shape.size(), 4U)
      << "The dimension of box prediction should be 4, but received " << bbox_pred->shape.size();
  CHECK_EQ(im_info->shape.size(), 2U)
      << "The dimension of image info should be 2, but received " << im_info->shape.size();
  CHECK(reporter->AssertEQ(im_info->shape[1], 3));

  auto batch = cls_prob->shape[0];

  std::vector<IndexExpr> oshape(
      {batch * proposal_attrs->rpn_post_nms_top_n, 5});
  reporter->Assign(types[3], TensorTypeNode::make(oshape, cls_prob->dtype));
  return true;
}

Expr MakeProposal(Expr cls_prob, Expr bbox_pred, Expr im_info, Array<IndexExpr> scales,
                  Array<IndexExpr> ratios, int feature_stride, double threshold,
                  int rpn_pre_nms_top_n, int rpn_post_nms_top_n, int rpn_min_size,
                  bool iou_loss) {
  auto attrs = make_node<ProposalAttrs>();
  attrs->scales = scales;
  attrs->ratios = ratios;
  attrs->feature_stride = feature_stride;
  attrs->threshold = threshold;
  attrs->rpn_pre_nms_top_n = rpn_pre_nms_top_n;
  attrs->rpn_post_nms_top_n = rpn_post_nms_top_n;
  attrs->rpn_min_size = rpn_min_size;
  attrs->iou_loss = iou_loss;
  static const Op& op = Op::Get("vision.proposal");
  return CallNode::make(op, {cls_prob, bbox_pred, im_info}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.vision._make.proposal")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 11>(MakeProposal, args, rv);
  });

RELAY_REGISTER_OP("vision.proposal")
    .describe(R"code(Generate region proposals via RPN.

 - **cls_prob**: 4-D with shape [batch, 2 * num_anchors, height, width].
 - **bbox_pred**: 4-D with shape [batch, 4 * num_anchors, height, width].
 - **im_info**: 2-D with shape [batch, 3].
 - **out**: 2-D with shape [batch * rpn_post_nms_top_n, 5].
 )code" TVM_ADD_FILELINE)
.set_num_inputs(3)
.add_argument("cls_prob", "Tensor", "Score of how likely proposal is object")
.add_argument("bbox_pred", "Tensor", "BBox predicted deltas from anchors for proposals")
.add_argument("im_info", "Tensor", "Image size and scale")
.set_support_level(5)
.add_type_rel("Proposal", ProposalRel)
.set_attr<FTVMCompute>("FTVMCompute",
                       [](const Attrs& attrs, const Array<Tensor>& inputs,
                          const Type& out_dtype, const Target& target) -> Array<Tensor> {
                         auto* param = attrs.as<ProposalAttrs>();
                         return Array<Tensor>{topi::proposal(
                                 inputs[0], inputs[1], inputs[2], param->scales, param->ratios,
                                 param->feature_stride, param->threshold, param->rpn_pre_nms_top_n,
                                 param->rpn_post_nms_top_n, param->rpn_min_size, param->iou_loss)};
                       });

TVM_REGISTER_NODE_TYPE(Decode_BBoxAttrs);

bool Decode_BBoxRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  auto decode_BBox_attrs = attrs.as<Decode_BBoxAttrs>();
  CHECK_EQ(types.size(), 4);
  const auto* rois = types[0].as<TensorTypeNode>();
  const auto* bbox_pred = types[1].as<TensorTypeNode>();
  const auto* im_info = types[2].as<TensorTypeNode>();

  if (!rois || !bbox_pred || !im_info) {
    return false;
  }

  CHECK_EQ(rois->shape.size(), 3U)
      << "The dimension of class probability should be 3, but received " << rois->shape.size();
  CHECK_EQ(bbox_pred->shape.size(), 3U)
      << "The dimension of box prediction should be 3, but received " << bbox_pred->shape.size();
  CHECK_EQ(im_info->shape.size(), 2U)
      << "The dimension of image info should be 2, but received " << im_info->shape.size();
  CHECK(reporter->AssertEQ(im_info->shape[1], 3));

  
  if(decode_BBox_attrs->class_agnostic){
        std::vector<IndexExpr> oshape(
        {rois->shape[0],rois->shape[1],4});
        reporter->Assign(types[3], TensorTypeNode::make(oshape, bbox_pred->dtype));
  } else {
        std::vector<IndexExpr> oshape(
        {bbox_pred->shape[0],bbox_pred->shape[1],bbox_pred->shape[2]});
        reporter->Assign(types[3], TensorTypeNode::make(oshape, bbox_pred->dtype));
  }
  return true;
}

Expr MakeDecode_BBox(Expr rois, Expr bbox_pred, Expr im_info, Array<IndexExpr> bbox_mean,
                  Array<IndexExpr> bbox_std, bool class_agnostic) {
  auto attrs = make_node<Decode_BBoxAttrs>();
  attrs->bbox_mean = bbox_mean;
  attrs->bbox_std = bbox_std;
  attrs->class_agnostic = class_agnostic;
  static const Op& op = Op::Get("vision.decode_BBox");
  return CallNode::make(op, {rois, bbox_pred, im_info}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.vision._make.decode_BBox")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 6>(MakeDecode_BBox, args, rv);
  });

RELAY_REGISTER_OP("vision.decode_BBox")
    .describe(R"code(perform decode on bounding box.

 - **rois**: 3-D.
 - **bbox_pred**: 3-D.
 - **im_info**: 2-D with shape [batch, 3].
 - **out**: 3-D.
 )code" TVM_ADD_FILELINE)
.set_num_inputs(3)
.add_argument("rois", "Tensor", "...")
.add_argument("bbox_pred", "Tensor", "BBox predicted deltas from anchors for proposals")
.add_argument("im_info", "Tensor", "Image size and scale")
.set_support_level(5)
.add_type_rel("Decode_BBox", Decode_BBoxRel)
.set_attr<FTVMCompute>("FTVMCompute",
                       [](const Attrs& attrs, const Array<Tensor>& inputs,
                          const Type& out_dtype, const Target& target) -> Array<Tensor> {
                         auto* param = attrs.as<Decode_BBoxAttrs>();
                         return Array<Tensor>{topi::decode_BBox(
                                 inputs[0], inputs[1], inputs[2], param->bbox_mean, param->bbox_std,
                                 param->class_agnostic)};
                       });


Expr MakeROIAlignV2(Expr data, Expr rois, Array<IndexExpr> pooled_size, double spatial_scale,
                    std::string layout) {
  auto attrs = make_node<ROIAlignAttrs>();
  attrs->pooled_size = pooled_size;
  attrs->spatial_scale = spatial_scale;
  attrs->layout = layout;
  static const Op& op = Op::Get("vision.roi_align_v2");
  return CallNode::make(op, {data, rois}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.vision._make.roi_align_v2")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 5>(MakeROIAlignV2, args, rv);
  });

RELAY_REGISTER_OP("vision.roi_align_v2")
    .describe(R"code("ROI Align.
 - **data**: This depends on the `layout` parameter. Input is 4D array of shape
             (batch_size, channels, height, width) if `layout` is `NCHW`.
 - **rois**: 2D array of shape (num_roi, 5). The last dimension should be in format of
          [batch_index, w_start, h_start, w_end, h_end].
 - **out**: This depends on the `layout` parameter. Output is 4D array of shape
            (num_roi, channels, pooled_height, pooled_width) if `layout` is `NCHW`.

 )code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("rois", "Tensor", "The input rois")
    .set_support_level(5)
    .add_type_rel("ROIAlign", ROIAlignRel);

}  // namespace relay
}  // namespace tvm
