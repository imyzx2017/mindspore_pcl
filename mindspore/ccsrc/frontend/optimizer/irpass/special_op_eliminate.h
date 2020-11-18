/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SPECIAL_OP_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SPECIAL_OP_ELIMINATE_H_

#include <securec.h>
#include <algorithm>
#include <memory>
#include <vector>
#include <string>

#include "frontend/optimizer/optimizer_caller.h"
#include "ir/pattern_matcher.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/prim_eliminate.h"
#include "frontend/optimizer/optimizer.h"
#include "utils/comm_manager.h"
#include "frontend/parallel/context.h"
#include "pipeline/jit/parse/resolve.h"

namespace mindspore {
namespace opt {
namespace irpass {
class SpecialOpEliminater : public OptimizerCaller {
 public:
  SpecialOpEliminater()
      : insert_gradient_of_(std::make_shared<PrimEliminater>(prim::kPrimInsertGradientOf)),
        stop_gradient_(std::make_shared<PrimEliminater>(prim::kPrimStopGradient)),
        hook_backward_(std::make_shared<PrimEliminater>(prim::kPrimHookBackward)),
        print_shape_type_(std::make_shared<PrimEliminater>(prim::kPrimPrintShapeType)),
        get_ref_value_(std::make_shared<PrimEliminater>(prim::kPrimGetRefValue)),
        mirror_(std::make_shared<PrimEliminater>(prim::kPrimMirror)),
        virtual_div_(std::make_shared<PrimEliminater>(prim::kPrimVirtualDiv)) {
    eliminaters_.emplace_back(insert_gradient_of_);
    eliminaters_.emplace_back(stop_gradient_);
    eliminaters_.emplace_back(hook_backward_);
    eliminaters_.emplace_back(print_shape_type_);
    eliminaters_.emplace_back(get_ref_value_);
    eliminaters_.emplace_back(mirror_);
    eliminaters_.emplace_back(virtual_div_);
  }
  ~SpecialOpEliminater() = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    AnfNodePtr new_node;
    for (auto &eliminater : eliminaters_) {
      new_node = (*eliminater)(optimizer, node);
      if (new_node != nullptr) {
        return new_node;
      }
    }
    return nullptr;
  }

 private:
  OptimizerCallerPtr insert_gradient_of_, stop_gradient_, hook_backward_, print_shape_type_, get_ref_value_, mirror_,
    virtual_div_;
  std::vector<OptimizerCallerPtr> eliminaters_{};
};

// {PrimVirtualDataset, X} -> X
// {PrimVirtualDataset, Xs} -> {prim::kPrimMakeTuple, Xs}
class VirtualDatasetEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!IsPrimitiveCNode(node, prim::kPrimVirtualDataset) || node->func_graph() == nullptr) {
      return nullptr;
    }

    auto &inputs = node->cast<CNodePtr>()->inputs();
    if (inputs.size() < 1) {
      return nullptr;
    }

    std::vector<AnfNodePtr> args;
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args));
    if (args.size() == 1) {
      return args.front();
    }

    (void)args.insert(args.begin(), NewValueNode(prim::kPrimMakeTuple));

    return node->func_graph()->NewCNode(args);
  }

  void Visit(const AnfNodePtr &) override {}
};

// {prim::kPrimSameTypeShape, X, Y} -> X
class SameEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    x_ = nullptr;
    AnfVisitor::Match(prim::kPrimSameTypeShape, {IsNode, IsNode})(node);
    return x_;
  }

  void Visit(const AnfNodePtr &node) override {
    if (x_ == nullptr) {
      x_ = node;
    }
  }

 private:
  AnfNodePtr x_{nullptr};
};

// {prim::kPrimCheckBprop, X, Y} -> X
class CheckBpropEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    x_ = nullptr;
    AnfVisitor::Match(prim::kPrimCheckBprop, {IsNode, IsNode})(node);
    return x_;
  }

  void Visit(const AnfNodePtr &node) override {
    if (x_ == nullptr) {
      x_ = node;
    }
  }

 private:
  AnfNodePtr x_{nullptr};
};

// Reset defer_inline flag
class ResetDeferInline : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (IsValueNode<FuncGraph>(node)) {
      auto fg = GetValueNode<FuncGraphPtr>(node);
      fg->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, false);
    }
    return nullptr;
  }
};

// {PrimZerosLike, Y} ->
// {PrimFill, {PrimDType, Y}, {PrimShape, Y}, 0}
class ZeroLikeFillZero : public AnfVisitor {
 public:
  ZeroLikeFillZero()
      : PrimFill_(prim::GetPythonOps("fill", "mindspore.ops.functional")->cast<PrimitivePtr>()),
        PrimShape_(prim::GetPythonOps("shape", "mindspore.ops.functional")->cast<PrimitivePtr>()),
        PrimDType_(prim::GetPythonOps("dtype", "mindspore.ops.functional")->cast<PrimitivePtr>()) {}
  ~ZeroLikeFillZero() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    y_ = nullptr;
    AnfVisitor::Match(prim::kPrimZerosLike, {IsNode})(node);
    if (y_ == nullptr || node->func_graph() == nullptr) {
      return nullptr;
    }
    if ((y_->abstract() == nullptr) || !y_->abstract()->isa<abstract::AbstractTensor>()) {
      auto fg = node->func_graph();
      auto dtype = fg->NewCNode({NewValueNode(PrimDType_), y_});
      auto shape = fg->NewCNode({NewValueNode(PrimShape_), y_});
      return fg->NewCNode({NewValueNode(PrimFill_), dtype, shape, NewValueNode(MakeValue(static_cast<int64_t>(0)))});
    }

    abstract::AbstractTensorPtr tensor_abstract = y_->abstract()->cast<abstract::AbstractTensorPtr>();

    TypePtr tensor_type_ptr = tensor_abstract->element()->BuildType();
    std::vector<int64_t> tensor_shape = tensor_abstract->shape()->shape();

    tensor::TensorPtr new_tensor_ptr = std::make_shared<tensor::Tensor>(tensor_type_ptr->type_id(), tensor_shape);
    size_t mem_size = GetTypeByte(tensor_type_ptr) * LongToSize(new_tensor_ptr->ElementsNum());
    char *data = reinterpret_cast<char *>(new_tensor_ptr->data_c());
    (void)memset_s(data, mem_size, 0, mem_size);

    auto new_cnode = NewValueNode(new_tensor_ptr);
    new_cnode->set_abstract(new_tensor_ptr->ToAbstract());

    return new_cnode;
  }

  void Visit(const AnfNodePtr &node) override { y_ = node; }

 private:
  AnfNodePtr y_{nullptr};
  PrimitivePtr PrimFill_, PrimShape_, PrimDType_;
};

// {prim::kPrimDepend, X, ValueCond}->X
class DependValueElim : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode<AnfNodePtr> x, cond;
    MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimDepend, x, cond), x, IsVNode(cond.GetNode(node)));
    return nullptr;
  }
};

// {{prim:getattr, {prim::resolve, SymbolStr, C}, zeros_like}, Xy} ->Tensor(0, shape(Xy))
// {prim:getattr, {prim::resolve, SymbolStr, zeros_like}, Xy} ->Tensor(0, shape(Xy))
// {{prim::resolve, CommonOPS, getitem}, (tensor0, tensor1,...), 0} -> tensor0
class PynativeEliminater : public OptimizerCaller {
  bool CheckNameSpaceVNode(const AnfNodePtr &node, const std::string &str_value) {
    ValueNodePtr value_node = node->cast<ValueNodePtr>();
    if (value_node == nullptr) {
      return false;
    }
    return GetValueNode<parse::NameSpacePtr>(value_node)->module() == str_value;
  }

  bool CheckSymbolVNode(const AnfNodePtr &node, const std::string &str_value) {
    ValueNodePtr value_node = node->cast<ValueNodePtr>();
    if (value_node == nullptr) {
      return false;
    }
    return GetValueNode<parse::SymbolPtr>(value_node)->symbol() == str_value;
  }
  bool CheckStrVNode(const AnfNodePtr &node, const std::string &str_value) {
    ValueNodePtr value_node = node->cast<ValueNodePtr>();
    if (value_node == nullptr) {
      return false;
    }
    return GetValueNode<StringImmPtr>(value_node)->value() == str_value;
  }

  ValuePtr FillGetItem(const ValuePtr &value, const ValuePtr &idx) {
    MS_LOG(DEBUG) << "Start FillGetItem" << value->ToString() << idx->ToString();
    if (!idx->isa<Int64Imm>()) {
      MS_LOG(EXCEPTION) << "Getitem idx must int:" << idx->ToString();
    }

    if (!value->isa<ValueTuple>()) {
      MS_LOG(EXCEPTION) << "Getitem value must tuple:" << value->ToString();
    }

    auto value_tuple = value->cast<ValueTuplePtr>();
    int idx_t = idx->cast<Int64ImmPtr>()->value();
    MS_LOG(DEBUG) << "Fill getitem" << idx_t << (*value_tuple)[idx_t]->ToString();
    return (*value_tuple)[idx_t];
  }

  ValuePtr FillZero(const ValuePtr &value) {
    MS_LOG(DEBUG) << "Start FillZero";
    ValuePtr out = nullptr;
    if (value->isa<Int64Imm>()) {
      return MakeValue(value->cast<Int64ImmPtr>()->value());
    }

    if (value->isa<tensor::Tensor>()) {
      MS_LOG(DEBUG) << "Start FillZero Tensor";
      auto tensor = value->cast<tensor::TensorPtr>();
      tensor::TensorPtr out_t = std::make_shared<tensor::Tensor>(tensor->Dtype()->type_id(), tensor->shape());
      char *data = reinterpret_cast<char *>(out_t->data_c());
      std::fill(data, data + out_t->data().nbytes(), 0);
      out = out_t;
    }

    std::vector<ValuePtr> value_list;
    if (value->isa<ValueTuple>()) {
      MS_LOG(DEBUG) << "Start FillZero Tuple" << value->ToString();
      auto value_tuple = value->cast<ValueTuplePtr>();
      for (size_t i = 0; i < value_tuple->size(); i++) {
        value_list.push_back(FillZero((*value_tuple)[i]));
      }
      out = std::make_shared<ValueTuple>(value_list);
    }
    if (out == nullptr) {
      MS_LOG(EXCEPTION) << "FillZero failed:" << value->ToString();
    }
    MS_LOG(DEBUG) << "Result: " << out->ToString();
    return out;
  }

 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    MS_LOG(DEBUG) << "Start replace node " << node->DebugString(4);
    PatternNode<AnfNodePtr> symbol_str_vnode, c_vnode, zeros_like_vnode, getitem_vnode, arg, arg1;
    auto resolve = PPrimitive(prim::kPrimResolve, symbol_str_vnode, c_vnode);
    auto getattr = PPrimitive(prim::kPrimGetAttr, resolve, zeros_like_vnode);
    auto pattern = PCNode(getattr, arg);

    if ((pattern).TryCapture(node) &&
        (CheckNameSpaceVNode(symbol_str_vnode.GetNode(node), "SymbolStr") &&
         CheckSymbolVNode(c_vnode.GetNode(node), "C") && CheckStrVNode(zeros_like_vnode.GetNode(node), "zeros_like"))) {
      auto rep = (arg).GetNode(node);
      if (rep != nullptr) {
        if (rep->isa<ValueNode>()) {
          auto value_node = rep->cast<ValueNodePtr>();
          auto new_value_node = NewValueNode(FillZero(value_node->value()));
          new_value_node->set_has_new_value(value_node->has_new_value());
          MS_LOG(DEBUG) << "Zeros_like replace ok " << rep->DebugString(4);
          return new_value_node;
        }
      }
    }

    MS_LOG(DEBUG) << "End replace 1 " << node->DebugString(4);
    auto resolve1 = PPrimitive(prim::kPrimResolve, symbol_str_vnode, zeros_like_vnode);
    auto pattern1 = PCNode(resolve1, arg);

    if ((pattern1).TryCapture(node) && (CheckNameSpaceVNode(symbol_str_vnode.GetNode(node), "SymbolStr") &&
                                        CheckSymbolVNode(zeros_like_vnode.GetNode(node), "zeros_like"))) {
      auto rep = (arg).GetNode(node);
      if (rep != nullptr) {
        if (rep->isa<ValueNode>()) {
          auto value_node = rep->cast<ValueNodePtr>();
          auto new_value_node = NewValueNode(FillZero(value_node->value()));
          new_value_node->set_has_new_value(value_node->has_new_value());
          MS_LOG(DEBUG) << "Zeros_like replace ok 2 " << rep->DebugString(4);
          return new_value_node;
        }
      }
    }

    // resolve(CommonOPS, getitem)((tensors), 3)
    auto resolve2 = PPrimitive(prim::kPrimResolve, symbol_str_vnode, getitem_vnode);
    auto pattern2 = PCNode(resolve2, arg, arg1);
    if ((pattern2).TryCapture(node) && (CheckNameSpaceVNode(symbol_str_vnode.GetNode(node), "CommonOPS") &&
                                        CheckSymbolVNode(getitem_vnode.GetNode(node), "getitem"))) {
      auto rep = (arg).GetNode(node);
      if (rep != nullptr) {
        if (rep->isa<ValueNode>()) {
          MS_LOG(DEBUG) << "Rep is " << rep->DebugString(4);
          ValueNodePtr new_node;
          auto value_node = rep->cast<ValueNodePtr>();
          auto rep1 = (arg1).GetNode(node);
          if (rep1 != nullptr) {
            if (rep1->isa<ValueNode>()) {
              auto idx = rep1->cast<ValueNodePtr>();
              if (!value_node->value()->isa<ValueTuple>()) {
                return nullptr;
              }
              new_node = NewValueNode(FillGetItem(value_node->value(), idx->value()));
              new_node->set_has_new_value(value_node->has_new_value());
            }
          }
          MS_LOG(DEBUG) << "Fill getitem  replace ok " << new_node->DebugString(4);
          return new_node;
        }
      }
    }

    MS_LOG(DEBUG) << "End Replace " << node->DebugString(4);
    return nullptr;
  }
};

class AllReduceConstElim : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode<AnfNodePtr> x;
    auto pattern = PPrimitive(prim::kPrimAllReduce, x);
    // If AllReduce takes contant value as input and values across devices are all the same(ensured by parallel mode)
    if (pattern.TryCapture(node) && IsVNode(x.GetNode(node)) &&
        (pattern.GetFuncGraph()->has_flag(parallel::AUTO_PARALLEL) ||
         pattern.GetFuncGraph()->has_flag(parallel::SEMI_AUTO_PARALLEL))) {
      auto cur_func_graph = pattern.GetFuncGraph();
      // If reduce operation is sum, then multiply constant by number of devices, otherwise just return the contant
      auto prim_cnode = pattern.GetOriginalNode();
      MS_EXCEPTION_IF_NULL(prim_cnode);
      auto primitive = GetCNodePrimitive(prim_cnode);
      auto reduce_op = primitive->GetAttr("op");
      auto group = primitive->GetAttr("group")->ToString();
      // For sum operation, multiply constant tensor by number of devices
      if (reduce_op->ToString() == "sum") {
        uint32_t num_of_devices;
        // Get number of devices
        if (!CommManager::GetInstance().GetRankSize(group, &num_of_devices)) {
          MS_LOG(EXCEPTION) << "Failed to get num of devices for group [" + group + "]";
        }
        // Multiply constant by number of devices then return
        std::vector<AnfNodePtr> mul_inputs;
        auto constant_node = x.GetNode(node);
        MS_EXCEPTION_IF_NULL(constant_node);
        auto constant_value_node = constant_node->cast<ValueNodePtr>();
        MS_EXCEPTION_IF_NULL(constant_value_node);
        if (!constant_value_node->value()->isa<tensor::Tensor>()) {
          MS_LOG(EXCEPTION) << "Expect the constant input for AllReduce to be a Tensor. Got " +
                                 constant_value_node->value()->ToString();
        }
        auto constant_tensor = constant_value_node->value()->cast<tensor::TensorPtr>();
        auto tensor_dtype = constant_tensor->Dtype();
        auto num_of_device_node = NewValueNode(std::make_shared<tensor::Tensor>((int64_t)num_of_devices, tensor_dtype));
        // Multiply nodes
        auto mul_prim = prim::GetPythonOps("tensor_mul", "mindspore.ops.functional");
        MS_EXCEPTION_IF_NULL(mul_prim);
        mul_inputs.push_back(NewValueNode(mul_prim));
        mul_inputs.push_back(constant_node);
        mul_inputs.push_back(num_of_device_node);
        return cur_func_graph->NewCNode(mul_inputs);
      } else {
        return x.GetNode(node);
      }
    }
    return nullptr;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SPECIAL_OP_ELIMINATE_H_
