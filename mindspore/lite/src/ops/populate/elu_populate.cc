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

#include "src/ops/elu.h"
#include "nnacl/fp32/elu.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateEluParameter(const mindspore::lite::PrimitiveC *primitive) {
  EluParameter *elu_parameter = reinterpret_cast<EluParameter *>(malloc(sizeof(EluParameter)));
  if (elu_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc EluParameter failed.";
    return nullptr;
  }
  memset(elu_parameter, 0, sizeof(EluParameter));
  elu_parameter->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Elu *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  elu_parameter->alpha_ = param->GetAlpha();
  return reinterpret_cast<OpParameter *>(elu_parameter);
}
Registry EluParameterRegistry(schema::PrimitiveType_Elu, PopulateEluParameter);
}  // namespace lite
}  // namespace mindspore
