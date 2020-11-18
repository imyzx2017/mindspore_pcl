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

#include "src/ops/argmax.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int ArgMax::GetAxis() const { return this->primitive_->value.AsArgMax()->axis; }
bool ArgMax::GetOutMaxValue() const { return this->primitive_->value.AsArgMax()->outMaxValue; }
int ArgMax::GetTopK() const { return this->primitive_->value.AsArgMax()->topK; }
bool ArgMax::GetKeepDims() const { return this->primitive_->value.AsArgMax()->keepDims; }
int ArgMax::GetAxisType() const { return this->primitive_->value.AsArgMax()->axisType; }

void ArgMax::SetAxis(int axis) { this->primitive_->value.AsArgMax()->axis = axis; }
void ArgMax::SetOutMaxValue(bool out_max_value) { this->primitive_->value.AsArgMax()->outMaxValue = out_max_value; }
void ArgMax::SetTopK(int top_k) { this->primitive_->value.AsArgMax()->topK = top_k; }
void ArgMax::SetKeepDims(bool keep_dims) { this->primitive_->value.AsArgMax()->keepDims = keep_dims; }
void ArgMax::SetAxisType(int axis_type) { this->primitive_->value.AsArgMax()->axisType = axis_type; }

#else
int ArgMax::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_ArgMax();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_ArgMax return nullptr";
    return RET_ERROR;
  }
  auto val_offset =
    schema::CreateArgMax(*fbb, attr->axis(), attr->outMaxValue(), attr->topK(), attr->keepDims(), attr->axisType());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_ArgMax, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int ArgMax::GetAxis() const { return this->primitive_->value_as_ArgMax()->axis(); }
bool ArgMax::GetOutMaxValue() const { return this->primitive_->value_as_ArgMax()->outMaxValue(); }
int ArgMax::GetTopK() const { return this->primitive_->value_as_ArgMax()->topK(); }
bool ArgMax::GetKeepDims() const { return this->primitive_->value_as_ArgMax()->keepDims(); }
int ArgMax::GetAxisType() const { return this->primitive_->value_as_ArgMax()->axisType(); }

PrimitiveC *ArgMaxCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<ArgMax>(primitive); }
Registry ArgMaxRegistry(schema::PrimitiveType_ArgMax, ArgMaxCreator);
#endif

int ArgMax::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  if (inputs_.size() != kSingleNum || outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "tensor number is error.";
    return RET_ERROR;
  }

  output->SetFormat(input->GetFormat());
  output->set_data_type(input->data_type());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  std::vector<int> output_shape(input->shape());
  auto input_shape_size = input->shape().size();
  auto axis = GetAxis() < 0 ? GetAxis() + input_shape_size : GetAxis();
  if (axis >= input_shape_size || axis < 0) {
    MS_LOG(ERROR) << "Invalid axis " << GetAxis() << ", input shape size: " << input_shape_size;
    return RET_PARAM_INVALID;
  }
  if (GetTopK() == 1 && !GetKeepDims()) {
    output_shape.erase(output_shape.begin() + axis);
  } else {
    output_shape[axis] = GetTopK();
  }

  output->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
