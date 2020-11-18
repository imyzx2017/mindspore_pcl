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

#include "src/ops/concat.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/concat_parameter.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateConcatParameter(const mindspore::lite::PrimitiveC *primitive) {
  ConcatParameter *concat_param = reinterpret_cast<ConcatParameter *>(malloc(sizeof(ConcatParameter)));
  if (concat_param == nullptr) {
    MS_LOG(ERROR) << "malloc ConcatParameter failed.";
    return nullptr;
  }
  memset(concat_param, 0, sizeof(ConcatParameter));
  concat_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Concat *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  concat_param->axis_ = param->GetAxis();
  return reinterpret_cast<OpParameter *>(concat_param);
}

Registry ConcatParameterRegistry(schema::PrimitiveType_Concat, PopulateConcatParameter);
}  // namespace lite
}  // namespace mindspore
