/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "nnacl/layer_norm_parameter.h"
#include "src/ops/layer_norm.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateLayerNormParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto layer_norm_parameter = reinterpret_cast<LayerNormParameter *>(malloc(sizeof(LayerNormParameter)));
  if (layer_norm_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc LayerNormParameter failed.";
    return nullptr;
  }
  memset(layer_norm_parameter, 0, sizeof(LayerNormParameter));
  layer_norm_parameter->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::LayerNorm *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  auto normalized_shape = param->GetNormalizedShape();
  layer_norm_parameter->normalized_dims_ = normalized_shape.size();
  layer_norm_parameter->normalized_shape_ = reinterpret_cast<int *>(malloc(normalized_shape.size() * sizeof(int)));
  if (layer_norm_parameter->normalized_shape_ == nullptr) {
    MS_LOG(ERROR) << "malloc layer_norm_parameter->normalized_shape_ failed.";
    free(layer_norm_parameter);
    layer_norm_parameter = nullptr;
    return nullptr;
  }
  for (size_t i = 0; i < normalized_shape.size(); i++) {
    layer_norm_parameter->normalized_shape_[i] = normalized_shape[i];
  }
  layer_norm_parameter->epsilon_ = param->GetEpsilon();
  layer_norm_parameter->elementwise_affine_ = param->GetElementwiseAffine();

  return reinterpret_cast<OpParameter *>(layer_norm_parameter);
}

Registry LayerNormParameterRegistry(schema::PrimitiveType_LayerNorm, PopulateLayerNormParameter);
}  // namespace lite
}  // namespace mindspore
