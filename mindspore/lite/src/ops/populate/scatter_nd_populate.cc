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

#include "src/ops/scatter_nd.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/scatter_nd.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateScatterNDParameter(const mindspore::lite::PrimitiveC *primitive) {
  ScatterNDParameter *scatter_nd_param = reinterpret_cast<ScatterNDParameter *>(malloc(sizeof(ScatterNDParameter)));
  if (scatter_nd_param == nullptr) {
    MS_LOG(ERROR) << "malloc ScatterNDParameter failed.";
    return nullptr;
  }
  memset(scatter_nd_param, 0, sizeof(ScatterNDParameter));
  scatter_nd_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(scatter_nd_param);
}
Registry ScatterNDParameterRegistry(schema::PrimitiveType_ScatterND, PopulateScatterNDParameter);

}  // namespace lite
}  // namespace mindspore
