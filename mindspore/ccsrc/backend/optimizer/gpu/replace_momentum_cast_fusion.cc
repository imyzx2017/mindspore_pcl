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
#include "backend/optimizer/gpu/replace_momentum_cast_fusion.h"

#include <memory>
#include <vector>
#include <string>

#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
const BaseRef ReplaceMomentumCastFusion::DefinePattern() const {
  VectorRef grad_cast = VectorRef({prim::kPrimCast, grad_});
  VectorRef momentum = VectorRef({prim::kPrimApplyMomentum, var_, acc_, lr_, grad_cast, mom_});
  return momentum;
}

const AnfNodePtr ReplaceMomentumCastFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                    const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);

  auto grad_cast = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 3);
  auto grad = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(grad_cast), 0);
  MS_EXCEPTION_IF_NULL(grad_cast);
  MS_EXCEPTION_IF_NULL(grad);

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->Replace(utils::cast<CNodePtr>(grad_cast), utils::cast<CNodePtr>(grad));
  std::vector<TypeId> outputs_type;
  std::vector<std::vector<size_t>> outputs_shape;
  auto output_num = AnfAlgo::GetOutputTensorNum(node);
  for (size_t i = 0; i < output_num; i++) {
    outputs_type.push_back(AnfAlgo::GetOutputInferDataType(node, i));
    outputs_shape.push_back(AnfAlgo::GetOutputInferShape(node, i));
  }
  outputs_type[3] = AnfAlgo::GetPrevNodeOutputInferDataType(grad_cast, 0);

  AnfAlgo::SetOutputInferTypeAndShape(outputs_type, outputs_shape, node.get());

  return node;
}
}  // namespace opt
}  // namespace mindspore
