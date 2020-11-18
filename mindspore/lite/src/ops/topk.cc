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

#include "src/ops/topk.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int TopK::GetK() const { return this->primitive_->value.AsTopK()->k; }
bool TopK::GetSorted() const { return this->primitive_->value.AsTopK()->sorted; }

void TopK::SetK(int k) { this->primitive_->value.AsTopK()->k = k; }
void TopK::SetSorted(bool sorted) { this->primitive_->value.AsTopK()->sorted = sorted; }

#else

int TopK::GetK() const { return this->primitive_->value_as_TopK()->k(); }
bool TopK::GetSorted() const { return this->primitive_->value_as_TopK()->sorted(); }
int TopK::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_TopK();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_TopK return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateTopK(*fbb, attr->k(), attr->sorted());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_TopK, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *TopKCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<TopK>(primitive); }
Registry TopKRegistry(schema::PrimitiveType_TopK, TopKCreator);

#endif

int TopK::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  if ((inputs_.size() != kSingleNum && inputs_.size() != kDoubleNum) || outputs_.size() != kDoubleNum) {
    MS_LOG(ERROR) << "input size: " << inputs_.size() << ", output size: " << outputs_.size();
    return RET_INPUT_TENSOR_ERROR;
  }
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output0 = outputs_.front();
  MS_ASSERT(output0 != nullptr);
  auto output1 = outputs_.at(1);
  MS_ASSERT(output1 != nullptr);
  output0->set_data_type(input->data_type());
  output0->SetFormat(input->GetFormat());
  output1->set_data_type(kNumberTypeInt32);
  output1->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  auto out_shape = input->shape();
  out_shape[out_shape.size() - 1] = GetK();
  if (inputs_.size() == kDoubleNum && inputs_.at(1)->data_c() != nullptr) {
    out_shape[out_shape.size() - 1] = reinterpret_cast<int *>(inputs_.at(1)->data_c())[0];
  }
  output0->set_shape(out_shape);
  output1->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
