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

#include "src/ops/permute.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int64_t> Permute::GetOrder() const { return this->primitive_->value.AsPermute()->order; }

void Permute::SetOrder(const std::vector<int64_t> &order) { this->primitive_->value.AsPermute()->order = order; }

#else

std::vector<int64_t> Permute::GetOrder() const {
  auto fb_vector = this->primitive_->value_as_Permute()->order();
  return std::vector<int64_t>(fb_vector->begin(), fb_vector->end());
}

void Permute::SetOrder(const std::vector<int64_t> &order) {}
int Permute::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Permute();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Permute return nullptr";
    return RET_ERROR;
  }
  std::vector<int64_t> order;
  if (attr->order() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->order()->size()); i++) {
      order.push_back(attr->order()->data()[i]);
    }
  }
  auto val_offset = schema::CreatePermuteDirect(*fbb, &order);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Permute, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *PermuteCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Permute>(primitive); }
Registry PermuteRegistry(schema::PrimitiveType_Permute, PermuteCreator);
#endif
}  // namespace lite
}  // namespace mindspore
