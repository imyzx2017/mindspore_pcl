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

#include "src/ops/eltwise.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Eltwise::GetMode() const { return this->primitive_->value.AsEltwise()->mode; }

void Eltwise::SetMode(int mode) { this->primitive_->value.AsEltwise()->mode = (schema::EltwiseMode)mode; }

#else
int Eltwise::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Eltwise();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Eltwise return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateEltwise(*fbb, attr->mode());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Eltwise, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int Eltwise::GetMode() const { return this->primitive_->value_as_Eltwise()->mode(); }

PrimitiveC *EltwiseCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Eltwise>(primitive); }
Registry EltwiseRegistry(schema::PrimitiveType_Eltwise, EltwiseCreator);
#endif

}  // namespace lite
}  // namespace mindspore
