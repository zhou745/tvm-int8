
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
 * \file proposal-inl.h
 * \brief Proposal Operator
 * \author Piotr Teterwak, Bing Xu, Jian Guo, Pengfei Chen, Yuntao Chen
*/
#ifndef TVM_CONTRIB_MXNET_DECODE_BBOX_INL_H_
#define TVM_CONTRIB_MXNET_DECODE_BBOX_INL_H_

#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cstring>
#include <iostream>
#include "./cuda_utils.h"

// extend NumericalParam
namespace tvm {
namespace contrib {
namespace mxnet {

/*!
* \brief structure for numerical tuple input
* \tparam VType data type of param
*/
template<typename VType>
struct NumericalParam {
  NumericalParam() {}
  explicit NumericalParam(VType *begin, VType *end) {
    int32_t size = static_cast<int32_t>(end - begin);
    info.resize(size);
    for (int i = 0; i < size; ++i) {
      info[i] = *(begin + i);
    }
  }
  inline size_t ndim() const {
    return info.size();
  }
  std::vector<VType> info;
};

template<typename VType>
inline std::istream &operator>>(std::istream &is, NumericalParam<VType> &param) {
  while (true) {
    char ch = is.get();
    if (ch == '(') break;
    if (!isspace(ch)) {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  VType idx;
  std::vector<VType> tmp;
  // deal with empty case
  size_t pos = is.tellg();
  char ch = is.get();
  if (ch == ')') {
    param.info = tmp;
    return is;
  }
  is.seekg(pos);
  // finish deal
  while (is >> idx) {
    tmp.push_back(idx);
    char ch;
    do {
      ch = is.get();
    } while (isspace(ch));
    if (ch == ',') {
      while (true) {
        ch = is.peek();
        if (isspace(ch)) {
          is.get(); continue;
        }
        if (ch == ')') {
          is.get(); break;
        }
        break;
      }
      if (ch == ')') break;
    } else if (ch == ')') {
      break;
    } else {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  param.info = tmp;
  return is;
}

template<typename VType>
inline std::ostream &operator<<(std::ostream &os, const NumericalParam<VType> &param) {
  os << '(';
  for (size_t i = 0; i < param.info.size(); ++i) {
    if (i != 0) os << ',';
    os << param.info[i];
  }
  // python style tuple
  if (param.info.size() == 1) os << ',';
  os << ')';
  return os;
}


struct Decode_BBoxParam {
  NumericalParam<float> bbox_mean;
  NumericalParam<float> bbox_std;
};

//========================
// Anchor Generation Utils
//========================

inline void _MakeAnchor(float w,
                        float h,
                        float x_ctr,
                        float y_ctr,
                        std::vector<float> *out_anchors) {
  out_anchors->push_back(x_ctr - 0.5f * (w - 1.0f));
  out_anchors->push_back(y_ctr - 0.5f * (h - 1.0f));
  out_anchors->push_back(x_ctr + 0.5f * (w - 1.0f));
  out_anchors->push_back(y_ctr + 0.5f * (h - 1.0f));
  out_anchors->push_back(0.0f);
}

inline void _Transform(float scale,
                       float ratio,
                       const std::vector<float>& base_anchor,
                       std::vector<float>  *out_anchors) {
  float w = base_anchor[2] - base_anchor[0] + 1.0f;
  float h = base_anchor[3] - base_anchor[1] + 1.0f;
  float x_ctr = base_anchor[0] + 0.5 * (w - 1.0f);
  float y_ctr = base_anchor[1] + 0.5 * (h - 1.0f);
  float size = w * h;
  float size_ratios = std::floor(size / ratio);
  float new_w = std::floor(std::sqrt(size_ratios) + 0.5f) * scale;
  float new_h = std::floor((new_w / scale * ratio) + 0.5f) * scale;

  _MakeAnchor(new_w, new_h, x_ctr, y_ctr, out_anchors);
}

// out_anchors must have shape (n, 5), where n is ratios.size() * scales.size()
inline void GenerateAnchors(const std::vector<float>& base_anchor,
                            const std::vector<float>& ratios,
                            const std::vector<float>& scales,
                            std::vector<float> *out_anchors) {
  for (size_t j = 0; j < ratios.size(); ++j) {
    for (size_t k = 0; k < scales.size(); ++k) {
      _Transform(scales[k], ratios[j], base_anchor, out_anchors);
    }
  }
}

}  // namespace mxnet
}  // namespace contrib
}  // namespace tvm

#endif  //  TVM_CONTRIB_MXNET_DECODE_BBOX_INL_H_
