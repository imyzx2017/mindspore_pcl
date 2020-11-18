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

#include "backend/optimizer/gpu/remove_redundant_format_transform.h"
#include <memory>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
const BaseRef RemoveRedundantFormatTransform::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  MS_EXCEPTION_IF_NULL(X);
  VectorRef transpose = VectorRef({prim::kPrimTranspose, X});
  return transpose;
}

const AnfNodePtr RemoveRedundantFormatTransform::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                         const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  MS_LOG(DEBUG) << "Process node:" << node->fullname_with_scope();
  auto input_node = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  MS_EXCEPTION_IF_NULL(input_node);
  AnfNodePtr first_transpose = nullptr;
  auto used_node_list = GetRealNodeUsedList(graph, input_node);
  for (size_t j = 0; j < used_node_list->size(); j++) {
    auto used_node = used_node_list->at(j).first;
    if (AnfAlgo::GetCNodeName(used_node) == prim::kPrimTranspose->name()) {
      first_transpose = used_node;
      break;
    }
  }
  auto first_transpose_perm = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(first_transpose, "perm");
  auto node_perm = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "perm");
  if ((first_transpose != node) && (first_transpose_perm == node_perm)) {
    return first_transpose;
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
