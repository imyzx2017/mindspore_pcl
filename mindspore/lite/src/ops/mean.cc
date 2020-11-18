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

#include "src/ops/mean.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Mean::GetAxis() const { return this->primitive_->value.AsMean()->axis; }
bool Mean::GetKeepDims() const { return this->primitive_->value.AsMean()->keepDims; }

void Mean::SetAxis(const std::vector<int> &axis) { this->primitive_->value.AsMean()->axis = axis; }
void Mean::SetKeepDims(bool keep_dims) { this->primitive_->value.AsMean()->keepDims = keep_dims; }

#else

std::vector<int> Mean::GetAxis() const {
  auto fb_vector = this->primitive_->value_as_Mean()->axis();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
bool Mean::GetKeepDims() const { return this->primitive_->value_as_Mean()->keepDims(); }

int Mean::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Mean();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Mean return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> axis;
  if (attr->axis() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->axis()->size()); i++) {
      axis.push_back(attr->axis()->data()[i]);
    }
  }
  auto val_offset = schema::CreateMeanDirect(*fbb, &axis, attr->keepDims());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Mean, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *MeanCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Mean>(primitive); }
Registry MeanRegistry(schema::PrimitiveType_Mean, MeanCreator);
#endif

namespace {
constexpr size_t kInputSize = 1;
constexpr size_t kOutputSize = 1;
}  // namespace
int Mean::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  if (inputs_.size() != kInputSize || outputs_.size() != kOutputSize) {
    return RET_ERROR;
  }
  auto input = inputs_.front();
  auto output = outputs_.front();
  if (input == nullptr || output == nullptr) {
    return RET_NULL_PTR;
  }
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  if (this->primitive_ == nullptr) {
    return RET_NULL_PTR;
  }

  bool keep_dims = static_cast<bool>(GetKeepDims());
  std::vector<int> in_shape = input->shape();
  std::vector<int> out_shape;
  const auto &axes = GetAxis();
  auto num_axes = axes.size();
  // reduce on all axes
  if (num_axes == 0) {
    if (keep_dims) {
      for (size_t i = 0; i < in_shape.size(); i++) {
        out_shape.push_back(1);
      }
    }
    output->set_shape(out_shape);
    output->set_data_type(input->data_type());
    return RET_OK;
  }
  // reduce on selected axes
  for (size_t i = 0; i < in_shape.size(); i++) {
    bool reduce_axis = false;
    for (size_t idx = 0; idx < num_axes; ++idx) {
      if (static_cast<size_t>(axes[idx]) == i) {
        reduce_axis = true;
        break;
      }
    }
    if (reduce_axis) {
      if (keep_dims) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(in_shape[i]);
    }
  }
  output->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
