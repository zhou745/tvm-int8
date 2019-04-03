/*!
 *  Copyright (c) 2019 by Wuwei Lin
 * \file op_base.h
 * \brief Base class of external operations
 */
#ifndef TVM_CONTRIB_MXNET_OP_BASE_H_
#define TVM_CONTRIB_MXNET_OP_BASE_H_

#include <dmlc/thread_local.h>
#include <vector>
#include <unordered_map>

class OpSignature {
  std::vector<int64_t> eles;
  uint64_t hash;

 public:
  OpSignature() {
    hash = 0;
  }

  explicit OpSignature(uint64_t hash) {
    this->hash = hash;
  }

  /*
   * This is to reserve space for the vector.
   */
  void Reserve(size_t num) {
    eles.reserve(num);
  }

  /*
   * We provide different methods to add signature to an op.
   * For operations, such as convolutin and fully connected, which determines
   * the optimal data layout for the op, we only need to use the shape and data
   * type to sign the op. For other operations, such as activation, which uses
   * whatever layout in the input array, we have to use the shape, the data type
   * and the layout to sign the op.
   */

  void AddSign(const DLTensor* tensor) {
    for (int i = 0; i < tensor->ndim; i++) {
      AddSign(tensor->shape[i]);
    }
  }

  void AddSign(int val) {
    hash = (hash << 1) + val;
    eles.push_back(val);
  }

  void AddSign(int64_t val) {
    hash = (hash << 1) + val;
    eles.push_back(val);
  }

  void AddSign(float val) {
    hash = (hash << 1) ^ std::hash<float>{}(val);
  }

  void AddSign(double val) {
    hash = (hash << 1) ^ std::hash<double>{}(val);
  }

  template <typename T>
  void AddSign(const std::vector<T>& vals) {
    for (auto&& v : vals) {
      AddSign(v);
    }
  }

  bool operator==(const OpSignature &sign) const {
    if (hash != sign.hash)
      return false;
    if (eles.size() != sign.eles.size())
      return false;
    for (size_t i = 0; i < eles.size(); i++)
      if (eles[i] != sign.eles[i])
        return false;
    return true;
  }

  uint64_t GetHash() const {
    return hash;
  }
};

struct OpHash {
  size_t operator()(const OpSignature &sign) const {
    return sign.GetHash();
  }
};

template <typename Derived, typename Signature>
class ExternalOpBase {
 public:
  static Derived* Get(const Signature& signature) {
    static thread_local std::unordered_map<Signature,
                                           std::unique_ptr<Derived>, OpHash> ops;
    auto it = ops.find(signature);
    if (it != ops.end()) {
      return it->second.get();
    }
    ops[signature] = std::unique_ptr<Derived>(new Derived(signature));
    return ops[signature].get();
  }
};

#endif  // TVM_CONTRIB_MXNET_OP_BASE_H_
