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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INLINE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INLINE_H_

#include <vector>
#include <utility>
#include <algorithm>
#include <unordered_map>

#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "ir/tensor.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
class ReplaceApplicator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!IsValueNode<FuncGraph>(node)) {
      return nullptr;
    }

    auto fg = GetValueNode<FuncGraphPtr>(node);
    if (fg->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE) || fg->stub() || *(fg->switch_layer_input())) {
      return nullptr;
    }

    auto out = fg->output();
    MS_EXCEPTION_IF_NULL(out);
    if (!out->isa<CNode>()) {
      return nullptr;
    }

    auto &inputs = out->cast<CNodePtr>()->inputs();
    auto params = fg->parameters();

    // Exclude first elements of inputs which is fn.
    auto input_size = inputs.size();
    auto param_size = params.size();
    if ((input_size == 1 && param_size == 0) || (input_size > 1 && (input_size - 1) == param_size &&
                                                 std::equal(inputs.begin() + 1, inputs.end(), params.begin()))) {
      auto inner = inputs[0];
      if (IsValueNode<Primitive>(inner) ||
          (IsValueNode<FuncGraph>(inner) && GetValueNode<FuncGraphPtr>(inner)->parent() == nullptr)) {
        return inner;
      }
    }

    return nullptr;
  }
};

using CriterionFuncType = std::function<bool(FuncGraphPtr, AnfNodePtr)>;

bool IsTrivial(const FuncGraphPtr &fg, AnfNodePtr) {
  auto n_cnode = fg->nodes().size() - fg->parameters().size();
  // There is at least one CNode(return, other_node).
  return n_cnode <= 2;
}

bool IsUniqueUse(const FuncGraphPtr &fg, AnfNodePtr) {
  auto &cnodes = fg->func_graph_cnodes_index();
  int64_t n_use = std::accumulate(
    cnodes.begin(), cnodes.end(), 0,
    [](int64_t sum, const std::pair<const CNodeIndexPairPtr, int64_t> &item) { return sum + item.second; });
  return n_use == 1;
}

bool IsInside(FuncGraphPtr, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node->func_graph());
  return node->func_graph()->has_flag("inline_inside");
}

bool IsCore(const FuncGraphPtr &fg, AnfNodePtr) { return fg->has_flag("core"); }

bool NoCriterion(FuncGraphPtr, AnfNodePtr) { return true; }

bool IsDirectParentCall(FuncGraphPtr fg, AnfNodePtr node) {
  bool unique_use = IsUniqueUse(fg, nullptr);
  bool is_recursive = fg->recursive();
  if (fg->parent() != nullptr && is_recursive) {
    if (fg->parent() == node->func_graph() && unique_use) {
      return true;
    }
  }
  return false;
}

// {G, Xs}
class InlinerBase : public AnfVisitor {
 public:
  explicit InlinerBase(std::vector<std::pair<CriterionFuncType, bool>> criterions, bool use_move = true)
      : use_move_(use_move), criterions_(criterions) {}
  ~InlinerBase() override = default;
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!node->isa<CNode>()) {
      return nullptr;
    }

    auto &inputs = node->cast<CNodePtr>()->inputs();
    if (inputs.size() < 1 || !IsValueNode<FuncGraph>(inputs[0])) {
      return nullptr;
    }

    // G
    auto fg = GetValueNode<FuncGraphPtr>(inputs[0]);
    if (fg->has_flag(FUNC_GRAPH_FLAG_DEFER_INLINE) || fg->stub()) {
      return nullptr;
    }

    // Do not inline GraphKernel to Cell.
    if (fg->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL) && !node->func_graph()->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
      // If the GraphKernel only contains a return node, we make it inlined.
      if (fg->nodes().size() - fg->parameters().size() > 1) {
        return nullptr;
      }
    }

    Reset();
    bool is_match = false;
    for (auto &criterion : criterions_) {
      if (!criterion.first(fg, node)) {
        continue;
      }

      if (criterion.second && IsRecursive(fg)) {
        continue;
      }

      is_match = true;
      break;
    }

    if (!is_match) {
      return nullptr;
    }

    std::vector<AnfNodePtr> args;
    (void)std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(args));
    // compare size to avoid the case that the function has default value after grad.
    // for which after renormalize, the function default value will be an input
    if (fg->parameters().size() != args.size()) {
      return nullptr;
    }
    auto is_unique_use = IsUniqueUse(fg, nullptr);
    // Not to inline after block if it has switch call inside, to avoid switch expansion.
    if (!is_unique_use && fg->has_flag(FUNC_GRAPH_FLAG_AFTER_BLOCK)) {
      auto has_branch_call = GraphHasBranch(fg);
      if (has_branch_call) {
        return TransformBranchCall(fg, node, args);
      }
    }

    if (use_move_ && is_unique_use) {
      auto mng = fg->manager();
      MS_EXCEPTION_IF_NULL(mng);
      ReplaceParams(mng, args, fg);
      auto out_node = fg->output();
      mng->MoveAllCNodeDropGraph(fg, node->func_graph(), inputs[0]->scope());
      return out_node;
    }

    return InlineClone(fg, node->func_graph(), args, inputs[0]->scope());
  }

  void ReplaceParams(const FuncGraphManagerPtr &mng, const std::vector<AnfNodePtr> &new_params,
                     const FuncGraphPtr &fg) {
    auto params = fg->parameters();
    auto old_size = params.size();
    if (old_size != new_params.size()) {
      MS_LOG(EXCEPTION) << "Parameter size not match." << old_size << " new " << new_params.size()
                        << fg->output()->DebugString(10);
    }
    for (size_t i = 0; i < old_size; i++) {
      (void)mng->Replace(params[i], new_params[i]);
    }
  }

  bool IsRecursive(const FuncGraphPtr &fg) {
    if (!is_checked_) {
      is_checked_ = true;
      is_recursive_ = fg->recursive();
    }
    return is_recursive_;
  }

  void Reset() {
    is_checked_ = false;
    is_recursive_ = false;
  }
  // For after block which contains branch call, delete the parameters which is not used.
  // In most cases, it may be a `Module` or other constant input.
  AnfNodePtr TransformBranchCall(const FuncGraphPtr &fg, const AnfNodePtr &node, const std::vector<AnfNodePtr> &args) {
    auto &fg_params = fg->parameters();
    std::vector<int64_t> used_param_index;
    auto mng = fg->manager();
    for (size_t i = 0; i < fg_params.size(); i++) {
      if (mng->node_users()[fg_params[i]].size() != 0) {
        used_param_index.emplace_back(i);
      }
    }
    // If all parameters are used by cnodes
    if (used_param_index.size() == fg_params.size()) {
      return nullptr;
    }
    if (transformed_branch_chache_.find(fg) == transformed_branch_chache_.end()) {
      MS_LOG(DEBUG) << "Parameter not used found for graph :" << fg->ToString();
      // clone a new graph and ignore the not used parameters
      FuncGraphPtr new_fg = TransformableClone(fg);
      auto &new_fg_params = new_fg->parameters();
      std::vector<AnfNodePtr> new_params;
      std::transform(used_param_index.begin(), used_param_index.end(), std::back_inserter(new_params),
                     [&new_fg_params](size_t i) { return new_fg_params[i]; });
      new_fg->set_parameters(new_params);
      // New func graph must set FUNC_GRAPH_FLAG_AFTER_BLOCK flag otherwise the new graph will be inlined.
      new_fg->set_flag(FUNC_GRAPH_FLAG_AFTER_BLOCK, true);
      // Add new graph to the cache to improve perfomance when call HasBranchCall.
      graph_branch_cache_[new_fg] = true;
      // If a graph be called at two or more locations, it should not be cloned once again, so add it to the cache.
      transformed_branch_chache_[fg] = new_fg;
    }
    std::vector<AnfNodePtr> node_inputs;
    node_inputs.push_back(NewValueNode(transformed_branch_chache_[fg]));
    std::transform(used_param_index.begin(), used_param_index.end(), std::back_inserter(node_inputs),
                   [&args](size_t i) { return args[i]; });
    return node->func_graph()->NewCNode(node_inputs);
  }

  // This is a try-best algorithm to find a graph which may generate branch call.
  // It does not handle high-order function call. For high-orderer call branch, it still may be inlined.
  bool GraphHasBranch(FuncGraphPtr fg) {
    if (graph_branch_cache_.find(fg) != graph_branch_cache_.end()) {
      return graph_branch_cache_[fg];
    }
    bool has_branch = false;
    auto nodes = fg->nodes();
    for (auto &item : nodes) {
      if (IsPrimitiveCNode(item, prim::kPrimSwitch)) {
        auto sw_inputs = item->cast<CNodePtr>()->inputs();
        if (sw_inputs.size() != 4) {
          MS_LOG(EXCEPTION) << "switch inputs should be 4";
        }
        if (!sw_inputs[1]->isa<ValueNode>() || IsValueNode<tensor::Tensor>(sw_inputs[1])) {
          has_branch = true;
          break;
        }
      } else if (IsCNodeGraph(item)) {
        auto cinputs = item->cast<CNodePtr>()->inputs();
        if (cinputs.size() < 1) {
          MS_LOG(EXCEPTION) << "graph call inputs should greater than 1";
        }
        FuncGraphPtr call_fg = GetValueNode<FuncGraphPtr>(cinputs[0]);
        bool call_fg_has_branch = GraphHasBranch(call_fg);
        if (call_fg_has_branch) {
          has_branch = true;
          break;
        }
      } else if (IsPrimitiveCNode(item, prim::kPrimPartial)) {
        auto cinputs = item->cast<CNodePtr>()->inputs();
        if (cinputs.size() < 2) {
          MS_LOG(EXCEPTION) << "partial call inputs should greater than 2";
        }
        FuncGraphPtr call_fg = GetValueNode<FuncGraphPtr>(cinputs[1]);
        if (call_fg == nullptr) {
          continue;
        }
        bool call_fg_has_branch = GraphHasBranch(call_fg);
        if (call_fg_has_branch) {
          has_branch = true;
          break;
        }
      }
    }
    graph_branch_cache_[fg] = has_branch;
    return has_branch;
  }

 private:
  bool is_checked_{false}, is_recursive_{false};
  bool use_move_;
  std::vector<std::pair<CriterionFuncType, bool>> criterions_;
  std::unordered_map<FuncGraphPtr, bool> graph_branch_cache_;
  // Key is the old func graph, and the value is the new func_graph
  std::unordered_map<FuncGraphPtr, FuncGraphPtr> transformed_branch_chache_;
};

class Inliner : public InlinerBase {
 public:
  explicit Inliner(bool use_move = true)
      : InlinerBase(
          {
            {IsUniqueUse, true},
            {IsTrivial, false},
            {IsInside, false},
            {IsCore, false},
            {IsDirectParentCall, false},
            {NoCriterion, true},
          },
          use_move) {}
  ~Inliner() override = default;
};

class DirectInliner : public InlinerBase {
 public:
  explicit DirectInliner(bool use_move = true)
      : InlinerBase(
          {
            {IsDirectParentCall, false},
          },
          use_move) {}
  ~DirectInliner() override = default;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INLINE_H_
