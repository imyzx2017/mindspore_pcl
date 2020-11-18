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
#include "src/ops/lsh_projection.h"
#include "nnacl/lsh_projection_parameter.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateLshProjectionParameter(const mindspore::lite::PrimitiveC *primitive) {
  LshProjectionParameter *lsh_project_param =
    reinterpret_cast<LshProjectionParameter *>(malloc(sizeof(LshProjectionParameter)));
  if (lsh_project_param == nullptr) {
    MS_LOG(ERROR) << "malloc LshProjectionParameter failed.";
    return nullptr;
  }
  memset(lsh_project_param, 0, sizeof(LshProjectionParameter));
  lsh_project_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::LshProjection *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  lsh_project_param->lsh_type_ = param->GetLshType();
  return reinterpret_cast<OpParameter *>(lsh_project_param);
}
Registry LshProjectionParameterRegistry(schema::PrimitiveType_LshProjection, PopulateLshProjectionParameter);

}  // namespace lite
}  // namespace mindspore
