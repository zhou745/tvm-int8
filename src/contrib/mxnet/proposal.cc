/*!
 *  Copyright (c) 2019 by Wuwei Lin
 * \file proposal.cc
 * \brief Proposal
 */
#include <tvm/relay/base.h>
#include "./cuda_utils.h"
#include "./proposal.h"

namespace tvm {
namespace contrib {
namespace mxnet {

TVM_REGISTER_GLOBAL("tvm.contrib.mxnet.proposal")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor* cls_prob = args[0];
  DLTensor* bbox_delta = args[1];
  DLTensor* im_info = args[2];
  DLTensor* out = args[3];

  auto cls_prob_tensor = ToTensor<mshadow::gpu, 4, float>(cls_prob);
  auto bbox_delta_tensor = ToTensor<mshadow::gpu, 4, float>(bbox_delta);
  auto im_info_tensor = ToTensor<mshadow::gpu, 2, float>(im_info);
  auto out_tensor = ToTensor<mshadow::gpu, 2, float>(out);

  ProposalGPUOp::Get(ProposalSign(args))->Forward(
    cls_prob_tensor, bbox_delta_tensor, im_info_tensor, out_tensor);
});


}  // namespace mxnet
}  // namespace contrib
}  // namespace tvm
