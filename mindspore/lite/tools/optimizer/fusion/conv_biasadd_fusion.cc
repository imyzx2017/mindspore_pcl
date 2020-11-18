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
#include "tools/optimizer/fusion/conv_biasadd_fusion.h"
#include <memory>
#include "src/ops/conv2d.h"
#include "src/ops/depthwise_conv2d.h"
#include "src/ops/deconv2d.h"
#include "src/ops/primitive_c.h"
#include "src/param_value_lite.h"
#include "schema/inner/model_generated.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"
#include "src/ops/add.h"

namespace mindspore::opt {
namespace {
constexpr size_t kAddInputsLength = 3;
constexpr size_t kAddWEIGHTINDEX = 2;
constexpr size_t kConvWeightIndex = 2;
constexpr size_t kConvBiasIndex = 3;
constexpr size_t kConvNoBiasLen = 3;
constexpr size_t kConvWithBiasLen = 4;
bool IsConvExtendNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_Conv2D || type == schema::PrimitiveType_DepthwiseConv2D ||
           type == schema::PrimitiveType_DeConv2D;
  }
  return false;
}
bool IsAddNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_Add || type == schema::PrimitiveType_BiasAdd;
  }
  return false;
}

int Get_Kenrnel_nums(const CNodePtr &conv_node) {
  MS_ASSERT(conv_node != nullptr);
  auto value_primitive = conv_node->input(0);
  auto value_node = value_primitive->cast<ValueNodePtr>();
  MS_ASSERT(value_node != nullptr);
  auto value = value_node->value();
  MS_ASSERT(value != nullptr);
  auto primitive = value->cast<PrimitiveCPtr>();
  MS_ASSERT(primitive != nullptr);
  auto type = (schema::PrimitiveType)primitive->Type();
  if (type == schema::PrimitiveType_Conv2D) {
    MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::Conv2D>>(primitive));
    auto primc = utils::cast<std::shared_ptr<mindspore::lite::Conv2D>>(primitive);
    MS_ASSERT(primc != nullptr);
    return primc->GetChannelOut();
  } else if (type == schema::PrimitiveType_DepthwiseConv2D) {
    MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::DepthwiseConv2D>>(primitive));
    auto primc = utils::cast<std::shared_ptr<mindspore::lite::DepthwiseConv2D>>(primitive);
    MS_ASSERT(primc != nullptr);
    return primc->GetChannelMultiplier() * primc->GetChannelIn();
  } else if (type == schema::PrimitiveType_DeConv2D) {
    MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::DeConv2D>>(primitive));
    auto primc = utils::cast<std::shared_ptr<mindspore::lite::DeConv2D>>(primitive);
    MS_ASSERT(primc != nullptr);
    return primc->GetChannelOut();
  } else {
    MS_LOG(ERROR) << "Unsupported opType, " << type;
    return 0;
  }
}
int GenConvNewBias(const FuncGraphPtr &func_graph, const CNodePtr &conv_node, const CNodePtr &bias_node) {
  AnfNodePtr conv_bias_node = nullptr;
  AnfNodePtr conv_weight_node = nullptr;
  if (conv_node->inputs().size() == kConvNoBiasLen) {
    conv_weight_node = conv_node->input(kConvWeightIndex);
  } else if (conv_node->inputs().size() == kConvWithBiasLen) {
    conv_weight_node = conv_node->input(kConvWeightIndex);
    conv_bias_node = conv_node->input(kConvBiasIndex);
  } else {
    MS_LOG(ERROR) << "conv node:" << conv_node->DebugString() << "inputs size must 3 or 4";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  auto kernel_nums = Get_Kenrnel_nums(conv_node);
  if (kernel_nums <= 0) {
    MS_LOG(ERROR) << "kernel num less than 0";
    return lite::RET_INVALID_OP_ATTR;
  }
  auto add_bias_data = new (std::nothrow) float[kernel_nums];
  if (add_bias_data == nullptr) {
    MS_LOG(ERROR) << "tensor_data is nullptr";
    return lite::RET_MEMORY_FAILED;
  }
  auto bias_add_weight = bias_node->input(kAddWEIGHTINDEX);
  if (CheckIfNodeIsParam(bias_add_weight) != lite::RET_OK) {
    delete[] add_bias_data;
    return lite::RET_INVALID_OP_ATTR;
  }
  auto add_weight_param = bias_add_weight->cast<ParameterPtr>()->default_param();
  auto add_weight_tensor = std::dynamic_pointer_cast<ParamValueLite>(add_weight_param);
  auto add_weight_data = reinterpret_cast<float *>(add_weight_tensor->tensor_addr());
  auto add_weight_shape = add_weight_tensor->tensor_shape();
  if (add_weight_shape.empty() || (add_weight_shape.size() == 1 && add_weight_shape[0] == 1)) {
    for (int i = 0; i < kernel_nums; i++) {
      add_bias_data[i] = *add_weight_data;
    }
  } else {
    if (EOK != memcpy_s(add_bias_data, kernel_nums * sizeof(float), add_weight_data, kernel_nums * sizeof(float))) {
      MS_LOG(ERROR) << "memcpy_s conv_bias_data failed";
      delete[] add_bias_data;
      return lite::RET_MEMORY_FAILED;
    }
  }
  if (conv_bias_node != nullptr) {
    if (CheckIfNodeIsParam(conv_bias_node) != lite::RET_OK) {
      delete[] add_bias_data;
      return lite::RET_INVALID_OP_ATTR;
    }
    auto conv_bias_param = conv_bias_node->cast<ParameterPtr>()->default_param();
    auto conv_bias_tensor = std::dynamic_pointer_cast<ParamValueLite>(conv_bias_param);
    if (conv_bias_tensor->tensor_shape().empty() || conv_bias_tensor->tensor_shape()[0] != kernel_nums) {
      MS_LOG(ERROR) << "conv_bias_node shape error";
      delete[] add_bias_data;
      return lite::RET_INVALID_OP_ATTR;
    }
    auto conv_bias_data = reinterpret_cast<float *>(conv_bias_tensor->tensor_addr());
    for (int i = 0; i < kernel_nums; i++) {
      conv_bias_data[i] += add_bias_data[i];
    }
    delete[] add_bias_data;
  } else {
    auto conv_weight_param = conv_weight_node->cast<ParameterPtr>()->default_param();
    auto conv_weight_tensor = std::dynamic_pointer_cast<ParamValueLite>(conv_weight_param);
    auto conv_new_bias = AddNewBiasNode(add_bias_data, func_graph, kernel_nums, conv_weight_tensor);
    conv_new_bias->set_name(conv_node->fullname_with_scope() + "_bias");
    conv_node->add_input(conv_new_bias);
  }
  return lite::RET_OK;
}
}  // namespace
const BaseRef ConvBiasaddFusion::DefinePattern() const {
  auto conv_var = std::make_shared<CondVar>(IsConvExtendNode);
  auto add_var = std::make_shared<CondVar>(IsAddNode);
  auto weight_var = std::make_shared<CondVar>(IsParamNode);
  return VectorRef({add_var, conv_var, weight_var});
}

const AnfNodePtr ConvBiasaddFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                            const EquivPtr &) const {
  MS_LOG(DEBUG) << "Enter pass process";
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
    return nullptr;
  }
  auto add_node = node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(add_node) != lite::RET_OK || CheckInputSize(add_node, kAddInputsLength) != lite::RET_OK) {
    return nullptr;
  }
  if (GetCNodeType(add_node) == schema::PrimitiveType_Add) {
    auto primitive_c = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(add_node->input(0));
    MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::Add>>(primitive_c));
    auto primc = utils::cast<std::shared_ptr<mindspore::lite::Add>>(primitive_c);
    MS_ASSERT(primc != nullptr);
    if (primc->GetActivationType() != schema::ActivationType_NO_ACTIVATION) {
      return add_node;
    }
  }

  AnfNodePtr conv_node_anf = add_node->input(1);
  if (CheckIfAnfNodeIsNull(conv_node_anf) != lite::RET_OK || IsMultiOutputTensors(func_graph, conv_node_anf)) {
    return nullptr;
  }
  auto conv_node = conv_node_anf->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(conv_node) != lite::RET_OK) {
    return nullptr;
  }
  int ret = GenConvNewBias(func_graph, conv_node, add_node);
  if (ret != lite::RET_OK) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return nullptr;
  }
  auto primitive_c = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(conv_node->input(0));
  MS_ASSERT(primitive_c != nullptr);
  auto type = primitive_c->Type();
  if (type == schema::PrimitiveType_Conv2D) {
    MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::Conv2D>>(primitive_c));
    auto primc = utils::cast<std::shared_ptr<mindspore::lite::Conv2D>>(primitive_c);
    MS_ASSERT(primc != nullptr);
    primc->SetHasBias(true);
  } else if (type == schema::PrimitiveType_DepthwiseConv2D) {
    MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::DepthwiseConv2D>>(primitive_c));
    auto primc = utils::cast<std::shared_ptr<mindspore::lite::DepthwiseConv2D>>(primitive_c);
    MS_ASSERT(primc != nullptr);
    primc->SetHasBias(true);
  } else if (type == schema::PrimitiveType_DeConv2D) {
    MS_ASSERT(utils::isa<std::shared_ptr<mindspore::lite::DeConv2D>>(primitive_c));
    auto primc = utils::cast<std::shared_ptr<mindspore::lite::DeConv2D>>(primitive_c);
    MS_ASSERT(primc != nullptr);
    primc->SetHasBias(true);
  } else {
    MS_LOG(ERROR) << "Unsupported opType, " << type;
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return nullptr;
  }
  return conv_node;
}
}  // namespace mindspore::opt
