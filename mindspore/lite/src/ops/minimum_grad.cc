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

#include "include/errorcode.h"
#include "src/ops/minimum_grad.h"
#include "src/common/log_adapter.h"
#ifdef PRIMITIVE_WRITEABLE
#include <float.h>
#include "tools/converter/quantizer/quantize_util.h"
#endif

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int MinimumGrad::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_MinimumGrad;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_MinimumGrad) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::MinimumGradT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "primitive value is nullptr";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

#else
PrimitiveC *MinimumGradCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<MinimumGrad>(primitive);
}
Registry MinimumGradRegistry(schema::PrimitiveType_MinimumGrad, MinimumGradCreator);

int MinimumGrad::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto val_offset = schema::CreateMinimumGrad(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_MinimumGrad, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
#endif
}  // namespace lite
}  // namespace mindspore
