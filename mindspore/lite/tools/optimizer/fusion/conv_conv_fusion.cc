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

#include "tools/optimizer/fusion/conv_conv_fusion.h"
#include <memory>
#include <functional>
#include "src/ops/primitive_c.h"
#include "src/ops/conv2d.h"
#include "schema/inner/model_generated.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
namespace {
constexpr size_t kConvNoBiasLen = 3;
constexpr size_t kConvWithBiasLen = 4;
constexpr size_t kConvWeightIndex = 2;
constexpr size_t kConvBiasIndex = 3;
constexpr size_t kNHWC_DIMS = 4;
constexpr size_t kNHWC_NDim = 0;
constexpr size_t kNHWC_HDim = 1;
constexpr size_t kNHWC_WDim = 2;
constexpr size_t kNHWC_CDim = 3;

bool IsCommonConvNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n) || utils::isa<ValueNodePtr>(n)) {
    auto type = opt::GetCNodeType(n);
    return type == schema::PrimitiveType_Conv2D;
  }
  return false;
}
STATUS GenNewConvBias(const ParameterPtr &down_bias_node, const ParameterPtr &down_weight_node,
                      const ParameterPtr &up_bias_node, const ParameterPtr &new_bias_node) {
  float *down_bias_data = nullptr;
  if (down_bias_node != nullptr) {
    auto down_bias_param = std::dynamic_pointer_cast<ParamValueLite>(down_bias_node->default_param());
    auto down_bias_shape = down_bias_param->tensor_shape();
    if (down_bias_shape.size() != 1) {
      MS_LOG(ERROR) << "cur conv_conv fusion only support scalar bias shape";
      return RET_FAILED;
    }
    down_bias_data = static_cast<float *>(down_bias_param->tensor_addr());
  }
  auto up_bias_param = std::dynamic_pointer_cast<ParamValueLite>(up_bias_node->default_param());
  auto up_bias_shape = up_bias_param->tensor_shape();
  if (up_bias_shape.size() != 1) {
    MS_LOG(ERROR) << "cur conv_conv fusion only support scalar bias shape";
    return RET_FAILED;
  }
  auto down_weight_param = std::dynamic_pointer_cast<ParamValueLite>(down_weight_node->default_param());
  auto down_weight_data = static_cast<float *>(down_weight_param->tensor_addr());
  auto down_weight_shape = down_weight_param->tensor_shape();
  auto up_bias_data = static_cast<float *>(up_bias_param->tensor_addr());
  int new_bias_size = down_weight_shape[0];
  auto new_bias_data = new (std::nothrow) float[new_bias_size];
  if (new_bias_data == nullptr) {
    MS_LOG(ERROR) << "tensor_data is nullptr";
    return RET_ERROR;
  }
  memset(new_bias_data, 0, new_bias_size * sizeof(float));
  auto up_bias_size = up_bias_shape[0];
  for (int i = 0; i < new_bias_size; i++) {
    for (int j = 0; j < up_bias_size; j++) {
      new_bias_data[i] += up_bias_data[j] * down_weight_data[i * up_bias_size + j];
    }
    if (down_bias_node != nullptr) {
      new_bias_data[i] += down_bias_data[i];
    }
  }
  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  MS_ASSERT(param_value != nullptr);
  param_value->set_tensor_shape({new_bias_size});
  param_value->set_tensor_type(up_bias_param->tensor_type());
  param_value->set_format(up_bias_param->format());
  param_value->set_tensor_addr(new_bias_data);
  param_value->set_tensor_size(sizeof(float) * new_bias_size);
  new_bias_node->set_name(down_bias_node->fullname_with_scope());
  new_bias_node->set_default_param(param_value);
  new_bias_node->set_abstract(down_bias_node->abstract());
  return RET_OK;
}
// up weight shape[cout0,h,w,cin0] down weight shape[cout1,1,1,cout0],new weight shape [cout1,h,w,cin0]
STATUS GenNewConvWeight(const ParameterPtr &down_weight_node, const ParameterPtr &up_weight_node,
                        const ParameterPtr &new_weight_node) {
  auto down_weight_param = std::dynamic_pointer_cast<ParamValueLite>(down_weight_node->default_param());
  auto down_weight_shape = down_weight_param->tensor_shape();
  auto up_weight_param = std::dynamic_pointer_cast<ParamValueLite>(up_weight_node->default_param());
  auto up_weight_shape = up_weight_param->tensor_shape();
  auto up_weight_data = static_cast<float *>(up_weight_param->tensor_addr());
  auto down_weight_data = static_cast<float *>(down_weight_param->tensor_addr());
  int cout0 = up_weight_shape[0];
  int cin0 = up_weight_shape[kNHWC_CDim];
  int cout1 = down_weight_shape[0];
  int window_size = up_weight_shape[kNHWC_WDim] * up_weight_shape[kNHWC_HDim];
  auto new_weight_shape = up_weight_shape;
  new_weight_shape[0] = down_weight_shape[0];
  int size = std::accumulate(new_weight_shape.begin(), new_weight_shape.end(), 1, std::multiplies<>());
  auto new_weight_data = new (std::nothrow) float[size];
  if (new_weight_data == nullptr) {
    MS_LOG(ERROR) << "tensor_data is nullptr";
    return RET_ERROR;
  }
  memset(new_weight_data, 0, size * sizeof(float));

  for (int i = 0; i < cout1; i++) {
    auto down_weight_base = i * cout0;
    auto new_weight_base = i * window_size * cin0;
    for (int j = 0; j < cin0; j++) {
      for (int k = 0; k < cout0; k++) {
        auto up_weight_offset = k * window_size * cin0 + j;
        auto down_weight_offset = down_weight_base + k;
        auto new_weight_offset = new_weight_base + j;
        for (int m = 0; m < window_size; m++) {
          new_weight_data[new_weight_offset + cin0 * m] +=
            up_weight_data[up_weight_offset + cin0 * m] * down_weight_data[down_weight_offset];
        }
      }
    }
  }
  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  MS_ASSERT(param_value != nullptr);
  param_value->set_tensor_shape(new_weight_shape);
  param_value->set_tensor_type(up_weight_param->tensor_type());
  param_value->set_format(up_weight_param->format());
  param_value->set_tensor_addr(new_weight_data);
  param_value->set_tensor_size(sizeof(float) * size);
  new_weight_node->set_name(down_weight_node->fullname_with_scope());
  new_weight_node->set_default_param(param_value);
  new_weight_node->set_abstract(down_weight_node->abstract());
  return RET_OK;
}
}  // namespace
const BaseRef ConvConvFusion::DefinePattern() const {
  auto up_conv_var = std::make_shared<CondVar>(IsCommonConvNode);
  auto down_conv_var = std::make_shared<CondVar>(IsCommonConvNode);
  auto down_weight_var = std::make_shared<CondVar>(IsParamNode);
  auto down_bias_var = std::make_shared<SeqVar>();
  return VectorRef({down_conv_var, up_conv_var, down_weight_var, down_bias_var});
}

// conv->conv1x1 fusion conv (w1x+b)w2+c = (w1*w2)*x+(w2*b+c)
const AnfNodePtr ConvConvFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                         const EquivPtr &) const {
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
    return nullptr;
  }
  auto down_conv_cnode = node->cast<CNodePtr>();
  if (down_conv_cnode->inputs().size() != kConvWithBiasLen && down_conv_cnode->inputs().size() != kConvNoBiasLen) {
    MS_LOG(WARNING) << "conv node inputs error ,name:" << down_conv_cnode->fullname_with_scope();
    return nullptr;
  }
  auto down_weight_parameter = down_conv_cnode->input(kConvWeightIndex)->cast<ParameterPtr>();
  auto down_weight_value = std::dynamic_pointer_cast<ParamValueLite>(down_weight_parameter->default_param());
  auto down_weight_shape = down_weight_value->tensor_shape();
  auto down_weight_type = down_weight_value->tensor_type();
  // down conv node filter must 1x1,only support float32
  if (down_weight_shape.size() != kNHWC_DIMS || down_weight_type != kNumberTypeFloat32 ||
      (down_weight_shape[kNHWC_HDim] != 1 || down_weight_shape[kNHWC_WDim] != 1)) {
    return nullptr;
  }

  auto up_conv_cnode = down_conv_cnode->input(1)->cast<CNodePtr>();
  auto up_weight_parameter = up_conv_cnode->input(kConvWeightIndex)->cast<ParameterPtr>();
  auto up_weight_value = std::dynamic_pointer_cast<ParamValueLite>(up_weight_parameter->default_param());
  auto up_weight_shape = up_weight_value->tensor_shape();
  auto up_weight_type = up_weight_value->tensor_type();
  if (up_weight_shape.size() != kNHWC_DIMS || up_weight_type != kNumberTypeFloat32 ||
      (up_weight_shape[kNHWC_HDim] != 1 || up_weight_shape[kNHWC_WDim] != 1)) {
    return nullptr;
  }
  if (up_conv_cnode->inputs().size() != kConvWithBiasLen && up_conv_cnode->inputs().size() != kConvNoBiasLen) {
    MS_LOG(WARNING) << "conv node inputs error ,name:" << up_conv_cnode->fullname_with_scope();
    return nullptr;
  }
  auto cin0 = up_weight_shape[kNHWC_CDim];
  auto cout0 = up_weight_shape[0];
  auto cout1 = down_weight_shape[0];
  if (cout0 != down_weight_shape[kNHWC_CDim]) {
    MS_LOG(WARNING) << "conv_conv_fusion up conv and down conv node shape not fit";
    return nullptr;
  }
  if (cin0 * (cout1 - cout0) > cout0 * cout1) {
    MS_LOG(INFO) << "conv_conv_fusion up conv and down conv node channel requirment not fit";
    return nullptr;
  }
  // multi output need skip
  if (IsMultiOutputTensors(func_graph, up_conv_cnode)) {
    return nullptr;
  }
  auto down_primitive = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(down_conv_cnode->input(0));
  auto down_conv_primitive = utils::cast<std::shared_ptr<mindspore::lite::Conv2D>>(down_primitive);
  auto up_primitive = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(up_conv_cnode->input(0));
  auto up_conv_primitive = utils::cast<std::shared_ptr<mindspore::lite::Conv2D>>(up_primitive);
  // up conv node must no activation
  if (up_conv_primitive == nullptr || up_conv_primitive->GetActivationType() != schema::ActivationType_NO_ACTIVATION) {
    return nullptr;
  }
  if (up_conv_primitive->GetGroup() != 1 || down_conv_primitive->GetGroup() != 1) {
    return nullptr;
  }
  auto new_weight_paramter = func_graph->add_parameter();
  if (GenNewConvWeight(down_weight_parameter, up_weight_parameter, new_weight_paramter) != RET_OK) {
    MS_LOG(ERROR) << "GenNewConvWeight failed.";
    return nullptr;
  }
  auto manager = func_graph->manager();
  manager->Replace(down_weight_parameter, new_weight_paramter);
  // up conv node no bias
  if (up_conv_cnode->inputs().size() == kConvWithBiasLen) {
    ParameterPtr down_bias_parameter;
    if (down_conv_cnode->inputs().size() == kConvWithBiasLen) {
      down_bias_parameter = down_conv_cnode->input(kConvBiasIndex)->cast<ParameterPtr>();
    }
    auto up_bias_parameter = up_conv_cnode->input(kConvBiasIndex)->cast<ParameterPtr>();
    auto new_bias_paramter = func_graph->add_parameter();
    if (GenNewConvBias(down_bias_parameter, down_weight_parameter, up_bias_parameter, new_bias_paramter) != RET_OK) {
      MS_LOG(ERROR) << "GenNewConvBias failed.";
      return nullptr;
    }
    if (down_conv_cnode->inputs().size() == kConvWithBiasLen) {
      manager->Replace(down_bias_parameter, new_bias_paramter);
    } else {
      down_conv_cnode->add_input(new_bias_paramter);
    }
  } else {
    MS_LOG(INFO) << "up conv node has no bias,no need replace bias.";
  }
  MS_LOG(INFO) << "fusion node success:" << down_conv_cnode->fullname_with_scope();
  // delete up conv node
  manager->Replace(up_conv_cnode, up_conv_cnode->input(1));
  return nullptr;
}
}  // namespace mindspore::opt
