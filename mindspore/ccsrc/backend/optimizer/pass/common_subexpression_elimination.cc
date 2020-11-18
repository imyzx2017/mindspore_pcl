/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/pass/common_subexpression_elimination.h"

#include <memory>
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/kernel_info.h"
#include "utils/flags.h"

namespace mindspore {
namespace opt {
namespace {
bool HasSideEffectAttr(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!AnfAlgo::HasNodeAttr(GRAPH_FLAG_SIDE_EFFECT, cnode)) {
    return false;
  }
  return AnfAlgo::GetNodeAttr<bool>(cnode, GRAPH_FLAG_SIDE_EFFECT);
}
}  // namespace

bool BackendCSE::CheckEqualKernelBuildInfo(const AnfNodePtr &main, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(main);
  MS_EXCEPTION_IF_NULL(node);
  auto main_kernel_info = dynamic_cast<device::KernelInfo *>(main->kernel_info());
  auto node_kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  if (main_kernel_info == nullptr && node_kernel_info == nullptr) {
    return true;
  }
  if (main_kernel_info != nullptr && node_kernel_info != nullptr) {
    return *main_kernel_info == *node_kernel_info;
  }
  return false;
}

bool BackendCSE::CheckEqualCnodeInputs(const AnfNodePtr &main, const AnfNodePtr &node) const {
  auto c_main = main->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(c_main);
  auto c_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(c_node);
  const auto &inp1 = c_main->inputs();
  const auto &inp2 = c_node->inputs();
  if (inp1.size() != inp2.size()) {
    return false;
  }
  for (size_t j = 0; j < inp1.size(); j++) {
    auto inp1_j = inp1[j];
    auto inp2_j = inp2[j];
    MS_EXCEPTION_IF_NULL(inp1_j);
    MS_EXCEPTION_IF_NULL(inp2_j);
    if (!(*inp1_j == *inp2_j)) {
      return false;
    }
  }
  return true;
}

bool BackendCSE::CheckReplace(const AnfNodePtr &main, const AnfNodePtr &node, bool check_side_effect) const {
  MS_EXCEPTION_IF_NULL(main);
  MS_EXCEPTION_IF_NULL(node);

  if (main->isa<ValueNode>() && node->isa<ValueNode>()) {
    auto main_value = GetValueNode(main);
    auto node_value = GetValueNode(node);
    if (main_value->isa<Primitive>() && node_value->isa<Primitive>()) {
      return false;
    } else if (main_value->isa<tensor::Tensor>() && node_value->isa<tensor::Tensor>()) {
      return (AbsOf(main) == AbsOf(node)) && CheckEqualKernelBuildInfo(main, node);
    } else {
      return (AbsOf(main) == AbsOf(node)) && (*main_value == *node_value);
    }
  } else if (main->isa<CNode>() && node->isa<CNode>()) {
    if (check_side_effect && HasSideEffectAttr(main)) {
      return false;
    }
    if (!CheckEqualKernelBuildInfo(main, node)) {
      return false;
    }
    return CheckEqualCnodeInputs(main, node);
  }
  return false;
}

bool CommonSubexpressionElimination::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto backend_cse = std::make_shared<BackendCSE>();
  return backend_cse->Cse(func_graph, func_graph->manager());
}
}  // namespace opt
}  // namespace mindspore
