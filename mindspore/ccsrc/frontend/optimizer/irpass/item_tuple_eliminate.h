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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ITEM_TUPLE_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ITEM_TUPLE_ELIMINATE_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
namespace irpass {
// (a, b, c, ...)[0] => a
// (a, b, c, ...)[1] => b
// {prim::kPrimTupleGetItem, {prim::kPrimMakeTuple, Xs}, C}
class GetitemEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsVNode})(node);
    AnfVisitor::Match(prim::kPrimListGetItem, {IsCNode, IsVNode})(node);

    if (is_match_) {
      return tuple_->input(id_);
    }
    return nullptr;
  }

  void Visit(const CNodePtr &cnode) override {
    if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) || IsPrimitiveCNode(cnode, prim::kPrimMakeList)) {
      tuple_ = cnode;
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (tuple_ != nullptr && IsValueNode<Int64Imm>(vnode)) {
      int64_t idx = GetValue<int64_t>(vnode->value());
      if (idx < 0) {
        idx = idx + tuple_->size() - 1;
      }
      id_ = LongToSize(idx + 1);
      if (tuple_->size() > id_) {
        is_match_ = true;
      }
    }
  }

  void Reset() {
    id_ = 0;
    tuple_ = nullptr;
    is_match_ = false;
  }

 private:
  bool is_match_{false};
  size_t id_{0};
  CNodePtr tuple_{nullptr};
};

// (a, b, c, ...)[0] => a
// (a, b, c, ...)[1] => b
// {prim::kPrimTupleGetItem, C1, C}
class GetitemConstEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsVNode, IsVNode})(node);
    AnfVisitor::Match(prim::kPrimListGetItem, {IsVNode, IsVNode})(node);

    if (is_match_) {
      auto out = NewValueNode((*tuple_)[id_]);
      out->set_has_new_value(has_new_value_);
      return out;
    }
    return nullptr;
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (IsValueNode<ValueTuple>(vnode)) {
      tuple_ = GetValueNode<ValueTuplePtr>(vnode);
      has_new_value_ = vnode->has_new_value();
    }
    if (tuple_ != nullptr && IsValueNode<Int64Imm>(vnode)) {
      id_ = LongToSize(GetValue<int64_t>(vnode->value()));
      if (tuple_->size() > id_) {
        is_match_ = true;
      }
    }
  }

  void Reset() {
    id_ = 0;
    tuple_ = nullptr;
    is_match_ = false;
  }

 private:
  bool is_match_{false};
  size_t id_{0};
  ValueTuplePtr tuple_{nullptr};
  bool has_new_value_{false};
};

// setitem((a, b, c, ...), 0, z) => (z, b, c, ...)
// setitem((a, b, c, ...), 1, z) => (a, z, c, ...)
// {prim::kPrimTupleSetItem, {prim::kPrimMakeTuple, Xs}, C, Z}
class SetitemEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleSetItem, {IsCNode, IsVNode, IsNode})(node);

    auto fg = node->func_graph();
    if (fg != nullptr && z_ != nullptr) {
      args_[id_] = z_;
      return fg->NewCNode(args_);
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (is_match_) {
      z_ = node;
      return;
    }

    AnfVisitor::Visit(node);
  }

  void Visit(const CNodePtr &cnode) override {
    if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) || IsPrimitiveCNode(cnode, prim::kPrimMakeList)) {
      auto &inputs = cnode->inputs();
      (void)std::copy(inputs.begin(), inputs.end(), std::back_inserter(args_));
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (args_.size() > 0 && IsValueNode<Int64Imm>(vnode)) {
      id_ = LongToSize(GetValue<int64_t>(vnode->value()) + 1);
      if (id_ < args_.size()) {
        is_match_ = true;
      }
    }
  }

  void Reset() {
    id_ = 0;
    z_ = nullptr;
    is_match_ = false;
    args_.clear();
  }

 private:
  bool is_match_{false};
  size_t id_{0};
  AnfNodePtr z_{nullptr};
  std::vector<AnfNodePtr> args_{};
};

// {prim::kPrimTupleGetItem, {prim::kPrimTupleSetItem, Y, C1, X}, C2}
class GetSetitemEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsVNode})(node);

    auto fg = node->func_graph();
    if (fg != nullptr && key1_ >= 0 && key2_ >= 0) {
      if (key1_ == key2_) {
        return last_;
      }
      return fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), tuple_, c2_});
    }
    return nullptr;
  }

  void Visit(const CNodePtr &cnode) override {
    if (IsPrimitiveCNode(cnode, prim::kPrimTupleSetItem)) {
      if (cnode->size() < 4) {
        return;
      }

      tuple_ = cnode->input(1);
      last_ = cnode->input(3);

      // key of setitem
      is_in_set_ = true;
      AnfVisitor::Visit(cnode->input(2));
      is_in_set_ = false;
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (IsValueNode<Int64Imm>(vnode)) {
      auto key = GetValue<int64_t>(vnode->value());
      if (is_in_set_) {
        key1_ = key;
      } else {
        c2_ = vnode;
        key2_ = key;
      }
    }
  }

  void Reset() {
    key1_ = -1;
    key2_ = -1;
    c2_ = nullptr;
    last_ = nullptr;
    tuple_ = nullptr;
    is_in_set_ = false;
  }

 private:
  bool is_in_set_{false};
  int64_t key1_{-1}, key2_{-1};
  AnfNodePtr tuple_{nullptr}, last_{nullptr}, c2_{nullptr};
};

// {prim::kPrimTupleGetItem, {prim::kPrimDepend, X, Y}, C} ->
// {prim::kPrimDepend, {prim::kPrimTupleGetItem, X, C}, Y}
class GetitemDependReorder : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    AnfVisitor::Match(prim::kPrimListGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    if (x_ == nullptr) {
      return nullptr;
    }

    auto fg = node->func_graph();
    auto item_node = NewCNode({NewValueNode(prim::kPrimTupleGetItem), x_, c_}, fg);
    return NewCNode({NewValueNode(prim::kPrimDepend), item_node, y_}, fg);
  }

  void Visit(const CNodePtr &cnode) override {
    // {prim::kPrimDepend, X, Y}
    if (IsPrimitiveCNode(cnode, prim::kPrimDepend) && cnode->size() == 3) {
      x_ = cnode->input(1);
      y_ = cnode->input(2);
    }
  }

  void Visit(const ValueNodePtr &vnode) override { c_ = vnode; }

  void Reset() {
    x_ = nullptr;
    y_ = nullptr;
    c_ = nullptr;
  }

 private:
  AnfNodePtr x_{nullptr}, y_{nullptr}, c_{nullptr};
};

class ItemTupleEliminater : public OptimizerCaller {
 public:
  ItemTupleEliminater()
      : get_item_eliminater_(std::make_shared<GetitemEliminater>()),
        get_item_const_eliminater_(std::make_shared<GetitemConstEliminater>()),
        set_item_eliminater_(std::make_shared<SetitemEliminater>()),
        get_set_item_eliminater_(std::make_shared<GetSetitemEliminater>()),
        get_item_depend_reorder_(std::make_shared<GetitemDependReorder>()) {
    eliminaters_.emplace_back(get_item_eliminater_);
    eliminaters_.emplace_back(get_item_const_eliminater_);
    eliminaters_.emplace_back(set_item_eliminater_);
    eliminaters_.emplace_back(get_set_item_eliminater_);
    eliminaters_.emplace_back(get_item_depend_reorder_);
  }
  ~ItemTupleEliminater() = default;

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
  OptimizerCallerPtr get_item_eliminater_, get_item_const_eliminater_, set_item_eliminater_, get_set_item_eliminater_,
    get_item_depend_reorder_;
  std::vector<OptimizerCallerPtr> eliminaters_{};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ITEM_TUPLE_ELIMINATE_H_
