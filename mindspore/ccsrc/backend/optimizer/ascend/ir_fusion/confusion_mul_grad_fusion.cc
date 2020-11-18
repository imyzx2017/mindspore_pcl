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
#include "backend/optimizer/ascend/ir_fusion/confusion_mul_grad_fusion.h"
#include <utility>
#include <memory>
#include <vector>
#include <algorithm>
#include <string>
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "abstract/abstract_value.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
const size_t kConfusionMulGradOutputNum = 2;

CNodePtr CreateFusionNode(const FuncGraphPtr &graph, const CNodePtr &reduce_sum, const AnfNodePtr &mul0_anf,
                          const AnfNodePtr &input3) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(reduce_sum);
  MS_EXCEPTION_IF_NULL(mul0_anf);
  MS_EXCEPTION_IF_NULL(input3);
  auto mul0 = mul0_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul0);

  auto prim = std::make_shared<Primitive>(kConfusionMulGradOpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), mul0->input(1), mul0->input(2), input3};
  auto fusion_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(fusion_node);
  fusion_node->set_scope(reduce_sum->scope());
  AnfAlgo::CopyNodeAttr(kAttrAxis, reduce_sum, fusion_node);
  AnfAlgo::CopyNodeAttr(kAttrKeepDims, reduce_sum, fusion_node);
  auto types = {AnfAlgo::GetOutputInferDataType(mul0, 0), AnfAlgo::GetOutputInferDataType(reduce_sum, 0)};
  auto shapes = {AnfAlgo::GetOutputInferShape(mul0, 0), AnfAlgo::GetOutputInferShape(reduce_sum, 0)};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, fusion_node.get());
  return fusion_node;
}

AnfNodePtr GetMul0(const FuncGraphPtr &graph, const AnfNodePtr &input2, const AnfNodePtr &mul1) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input2);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(input2) == manager->node_users().end()) {
    MS_LOG(EXCEPTION) << "node has no output in manager";
  }

  AnfNodePtr mul0 = nullptr;
  const AnfNodeIndexSet &outputs_set = manager->node_users()[input2];
  // input2 must be the 2rd input of mul0
  auto it = std::find_if(outputs_set.begin(), outputs_set.end(), [&mul1](const std::pair<AnfNodePtr, int> &node_index) {
    return node_index.first != mul1 && node_index.second == 2;
  });
  if (it != outputs_set.end() && AnfAlgo::GetCNodeName(it->first) == prim::kPrimMul->name()) {
    mul0 = it->first;
  }
  return mul0;
}

bool QuitFusion(const FuncGraphPtr &graph, const AnfNodePtr &mul0_anf, const AnfNodePtr &mul1_anf,
                const AnfNodePtr &reduce_sum, const AnfNodePtr &input2) {
  MS_EXCEPTION_IF_NULL(mul0_anf);
  MS_EXCEPTION_IF_NULL(mul1_anf);
  MS_EXCEPTION_IF_NULL(reduce_sum);
  MS_EXCEPTION_IF_NULL(input2);
  auto addn = input2->cast<CNodePtr>();
  if (addn == nullptr || AnfAlgo::GetCNodeName(addn) != prim::kPrimAddN->name()) {
    MS_LOG(INFO) << "mul's second input is not addn";
    return true;
  }
  if (AnfAlgo::IsDynamicShape(addn)) {
    return true;
  }
  std::vector<size_t> shape = AnfAlgo::GetOutputInferShape(addn, 0);
  if (shape.size() != 2 || !(shape[1] == 1024 || shape[1] == 768)) {
    MS_LOG(INFO) << "Addn's infer shape is not equal [x,1024] or [x,768]";
    return true;
  }
  if (!mul0_anf->isa<CNode>() || !mul1_anf->isa<CNode>()) {
    return true;
  }
  auto mul1 = mul1_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul1);
  auto mul0 = mul0_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul0);

  if (IsDepend(*graph, mul0->input(1), {reduce_sum})) {
    MS_LOG(INFO) << "mul0->input(1) depends on reduce_sum, quit fusion";
    return true;
  }
  if (IsDepend(*graph, mul1->input(1), {mul0})) {
    MS_LOG(INFO) << "mul1->input(1) depends on mul0, quit fusion";
    return true;
  }
  return false;
}
}  // namespace

const BaseRef ConfusionMulGradFusion::DefinePattern() const {
  VectorRef mul1({prim::kPrimMul, input3_, input2_});
  VectorRef reduce_sum({prim::kPrimReduceSum, mul1});
  return reduce_sum;
}

const AnfNodePtr ConfusionMulGradFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                 const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto input2 = utils::cast<AnfNodePtr>((*equiv)[input2_]);
  auto input3 = utils::cast<AnfNodePtr>((*equiv)[input3_]);
  auto reduce_sum = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(reduce_sum);
  auto mul1 = reduce_sum->input(1);
  if (IsUsedByOthers(graph, mul1)) {
    MS_LOG(INFO) << "Mul1 is used by others, quit fusion!";
    return nullptr;
  }
  auto mul0 = GetMul0(graph, input2, mul1);
  if (mul0 == nullptr) {
    MS_LOG(INFO) << "Mul0 do not exist, quit fusion";
    return nullptr;
  }
  if (QuitFusion(graph, mul0, mul1, node, input2)) {
    return nullptr;
  }

  auto fusion_node = CreateFusionNode(graph, reduce_sum, mul0, input3);
  std::vector<AnfNodePtr> fusion_node_outputs;
  CreateMultipleOutputsOfAnfNode(graph, fusion_node, kConfusionMulGradOutputNum, &fusion_node_outputs);

  auto manage = graph->manager();
  MS_EXCEPTION_IF_NULL(manage);
  manage->Replace(mul0, fusion_node_outputs[0]);
  return fusion_node_outputs[1];
}
}  // namespace opt
}  // namespace mindspore
