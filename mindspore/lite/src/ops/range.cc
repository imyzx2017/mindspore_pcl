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

#include "src/ops/range.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Range::GetDType() const { return this->primitive_->value.AsRange()->dType; }
int Range::GetStart() const { return this->primitive_->value.AsRange()->start; }
int Range::GetLimit() const { return this->primitive_->value.AsRange()->limit; }
int Range::GetDelta() const { return this->primitive_->value.AsRange()->delta; }

void Range::SetDType(int d_type) { this->primitive_->value.AsRange()->dType = d_type; }
void Range::SetStart(int start) { this->primitive_->value.AsRange()->start = start; }
void Range::SetLimit(int limit) { this->primitive_->value.AsRange()->limit = limit; }
void Range::SetDelta(int delta) { this->primitive_->value.AsRange()->delta = delta; }

#else

int Range::GetDType() const { return this->primitive_->value_as_Range()->dType(); }
int Range::GetStart() const { return this->primitive_->value_as_Range()->start(); }
int Range::GetLimit() const { return this->primitive_->value_as_Range()->limit(); }
int Range::GetDelta() const { return this->primitive_->value_as_Range()->delta(); }
int Range::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Range();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Range return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateRange(*fbb, attr->dType(), attr->start(), attr->limit(), attr->delta());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Range, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *RangeCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Range>(primitive); }
Registry RangeRegistry(schema::PrimitiveType_Range, RangeCreator);
#endif

int Range::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);

  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }

  int shape_size = std::ceil(static_cast<float>(GetLimit() - GetStart()) / GetDelta());
  std::vector<int> in_shape(1);
  in_shape.push_back(shape_size);
  output->set_shape(in_shape);

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
