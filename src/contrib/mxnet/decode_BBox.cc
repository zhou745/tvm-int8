/*!
 *  Copyright (c) 2019 by Wuwei Lin
 * \file proposal.cc
 * \brief Proposal
 */
#include <tvm/relay/base.h>
#include "./cuda_utils.h"
#include "./decode_BBox.h"

namespace tvm {
namespace contrib {
namespace mxnet {

TVM_REGISTER_GLOBAL("tvm.contrib.mxnet.decode_BBox")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor* boxes = args[0];
  DLTensor* bbox_delta = args[1];
  DLTensor* im_info = args[2];
  DLTensor* out = args[3];

  auto boxes_tensor = ToTensor<mshadow::gpu, 2, float>(boxes);
  auto bbox_delta_tensor = ToTensor<mshadow::gpu, 3, float>(bbox_delta);
  auto im_info_tensor = ToTensor<mshadow::gpu, 2, float>(im_info);
  auto out_tensor = ToTensor<mshadow::gpu, 3, float>(out);

  Decode_BBoxOp::Get(Decode_BBoxSign(args))->Forward(
    boxes_tensor, bbox_delta_tensor, im_info_tensor, out_tensor);
});


}  // namespace mxnet
}  // namespace contrib
}  // namespace tvm
