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

#include "src/ops/fake_quant_with_min_max_vars.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
bool FakeQuantWithMinMaxVars::GetNarrowRange() const {
  return this->primitive_->value.AsFakeQuantWithMinMaxVars()->narrowRange;
}
int FakeQuantWithMinMaxVars::GetNumBits() const { return this->primitive_->value.AsFakeQuantWithMinMaxVars()->numBits; }

void FakeQuantWithMinMaxVars::SetNarrowRange(bool narrow_range) {
  this->primitive_->value.AsFakeQuantWithMinMaxVars()->narrowRange = narrow_range;
}
void FakeQuantWithMinMaxVars::SetNumBits(int num_bits) {
  this->primitive_->value.AsFakeQuantWithMinMaxVars()->numBits = num_bits;
}

#else
int FakeQuantWithMinMaxVars::UnPackToFlatBuilder(const schema::Primitive *primitive,
                                                 flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_FakeQuantWithMinMaxVars();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_FakeQuantWithMinMaxVars return nullptr";
    return RET_ERROR;
  }

  auto val_offset = schema::CreateFakeQuantWithMinMaxVars(*fbb, attr->narrowRange(), attr->numBits());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_FakeQuantWithMinMaxVars, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
bool FakeQuantWithMinMaxVars::GetNarrowRange() const {
  return this->primitive_->value_as_FakeQuantWithMinMaxVars()->narrowRange();
}
int FakeQuantWithMinMaxVars::GetNumBits() const {
  return this->primitive_->value_as_FakeQuantWithMinMaxVars()->numBits();
}

PrimitiveC *FakeQuantWithMinMaxVarsCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<FakeQuantWithMinMaxVars>(primitive);
}
Registry FakeQuantWithMinMaxVarsRegistry(schema::PrimitiveType_FakeQuantWithMinMaxVars, FakeQuantWithMinMaxVarsCreator);
#endif
}  // namespace lite
}  // namespace mindspore
