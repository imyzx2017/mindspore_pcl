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
#include "tools/optimizer/fusion/sigmoid_mul_fusion.h"
#include <memory>
#include "src/ops/primitive_c.h"
#include "src/ops/activation.h"
#include "src/param_value_lite.h"
#include "schema/inner/model_generated.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
namespace {
bool IsActivationNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_Activation;
  }
  return false;
}
bool IsMulNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_Mul;
  }
  return false;
}
}  // namespace
const BaseRef SigmoidMulFusion::DefinePattern() const {
  auto input_var = std::make_shared<Var>();
  auto activation_var = std::make_shared<CondVar>(IsActivationNode);
  auto mul_var = std::make_shared<CondVar>(IsMulNode);
  auto activation_input = VectorRef({activation_var, input_var});
  return VectorRef({mul_var, input_var, activation_input});
}

// x * sigmoid(x) ->swish(x)
const AnfNodePtr SigmoidMulFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                           const EquivPtr &) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  auto mul_cnode = node->cast<CNodePtr>();
  MS_ASSERT(mul_cnode != nullptr);
  auto activation_cnode = mul_cnode->input(2)->cast<CNodePtr>();
  MS_ASSERT(activation_cnode != nullptr);
  // activation must sigmoid
  auto primitive = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(activation_cnode->input(0));
  auto activation_prim = utils::cast<std::shared_ptr<mindspore::lite::Activation>>(primitive);
  if (activation_prim->GetType() != schema::ActivationType_SIGMOID) {
    return nullptr;
  }
  activation_prim->SetType(schema::ActivationType_SWISH);
  return activation_cnode;
}
}  // namespace mindspore::opt
