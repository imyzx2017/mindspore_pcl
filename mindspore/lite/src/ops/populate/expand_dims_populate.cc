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

#include "src/ops/expand_dims.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/expandDims.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateExpandDimsParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto param = reinterpret_cast<mindspore::lite::ExpandDims *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  ExpandDimsParameter *expand_dims_param = reinterpret_cast<ExpandDimsParameter *>(malloc(sizeof(ExpandDimsParameter)));
  if (expand_dims_param == nullptr) {
    MS_LOG(ERROR) << "malloc ExpandDimsParameter failed.";
    return nullptr;
  }
  memset(expand_dims_param, 0, sizeof(ExpandDimsParameter));
  expand_dims_param->op_parameter_.type_ = primitive->Type();
  expand_dims_param->dim_ = param->GetDim();
  return reinterpret_cast<OpParameter *>(expand_dims_param);
}

Registry ExpandDimsParameterRegistry(schema::PrimitiveType_ExpandDims, PopulateExpandDimsParameter);

}  // namespace lite
}  // namespace mindspore
