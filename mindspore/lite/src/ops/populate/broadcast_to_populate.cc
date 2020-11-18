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

#include "src/ops/broadcast_to.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/broadcast_to.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateBroadcastToParameter(const mindspore::lite::PrimitiveC *primitive) {
  BroadcastToParameter *broadcast_param =
    reinterpret_cast<BroadcastToParameter *>(malloc(sizeof(BroadcastToParameter)));
  if (broadcast_param == nullptr) {
    MS_LOG(ERROR) << "malloc BroadcastToParameter failed.";
    return nullptr;
  }
  memset(broadcast_param, 0, sizeof(BroadcastToParameter));
  auto param = reinterpret_cast<mindspore::lite::BroadcastTo *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  broadcast_param->op_parameter_.type_ = primitive->Type();
  auto dst_shape = param->GetDstShape();
  broadcast_param->shape_size_ = dst_shape.size();
  for (size_t i = 0; i < broadcast_param->shape_size_; ++i) {
    broadcast_param->shape_[i] = dst_shape[i];
  }
  return reinterpret_cast<OpParameter *>(broadcast_param);
}

Registry BroadcastToParameterRegistry(schema::PrimitiveType_BroadcastTo, PopulateBroadcastToParameter);
}  // namespace lite
}  // namespace mindspore
