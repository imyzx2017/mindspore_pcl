/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/ops/layer_norm.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> LayerNorm::GetNormalizedShape() const {
  return this->primitive_->value.AsLayerNorm()->normalizedShape;
}
float LayerNorm::GetEpsilon() const { return this->primitive_->value.AsLayerNorm()->epsilon; }
bool LayerNorm::GetElementwiseAffine() const { return this->primitive_->value.AsLayerNorm()->elementwiseAffine; }

void LayerNorm::SetNormalizedShape(const std::vector<int> &normalizedShape) {
  this->primitive_->value.AsLayerNorm()->normalizedShape = normalizedShape;
}
void LayerNorm::SetEpsilon(float epsilon) { this->primitive_->value.AsLayerNorm()->epsilon = epsilon; }
void LayerNorm::SetElementwiseAffine(bool elementwiseAffine) {
  this->primitive_->value.AsLayerNorm()->elementwiseAffine = elementwiseAffine;
}

#else
int LayerNorm::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_LayerNorm();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_LayerNorm return nullptr";
    return RET_ERROR;
  }

  std::vector<int32_t> normalizedShape;
  if (attr->normalizedShape() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->normalizedShape()->size()); i++) {
      normalizedShape.push_back(attr->normalizedShape()->data()[i]);
    }
  }
  auto val_offset = schema::CreateLayerNormDirect(*fbb, &normalizedShape, attr->epsilon(), attr->elementwiseAffine());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_LayerNorm, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
std::vector<int> LayerNorm::GetNormalizedShape() const {
  auto fb_vector = this->primitive_->value_as_LayerNorm()->normalizedShape();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
float LayerNorm::GetEpsilon() const { return this->primitive_->value_as_LayerNorm()->epsilon(); }
bool LayerNorm::GetElementwiseAffine() const { return this->primitive_->value_as_LayerNorm()->elementwiseAffine(); }
PrimitiveC *LayerNormCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<LayerNorm>(primitive);
}
Registry LayerNormRegistry(schema::PrimitiveType_LayerNorm, LayerNormCreator);

#endif
int LayerNorm::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  if (outputs_.size() != kSingleNum || (inputs_.size() != kSingleNum && inputs_.size() != kMultiNum)) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs_.size() << ",input size: " << inputs_.size();
    return RET_PARAM_INVALID;
  }
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.at(0);
  MS_ASSERT(output != nullptr);
  output->SetFormat(input->GetFormat());
  output->set_data_type(input->data_type());

  if (GetElementwiseAffine() && inputs_.size() != kMultiNum) {
    MS_LOG(INFO) << "input tensor amount error";
    return RET_INPUT_TENSOR_ERROR;
  }
  if (!GetElementwiseAffine() && inputs_.size() != kSingleNum) {
    MS_LOG(INFO) << "input tensor amount error";
    return RET_INPUT_TENSOR_ERROR;
  }
  auto input_shape = input->shape();
  auto normalized_shape = GetNormalizedShape();
  if (normalized_shape.size() > input_shape.size() || normalized_shape.size() == 0) {
    MS_LOG(INFO) << "normalized_shape attr invalid";
    return RET_PARAM_INVALID;
  }
  size_t first_index = input_shape.size() - normalized_shape.size();
  for (size_t i = first_index; i < input_shape.size(); ++i) {
    if (input_shape[i] != normalized_shape[i - first_index]) {
      MS_LOG(INFO) << "normalized_shape attr invalid";
      return RET_PARAM_INVALID;
    }
  }
  if (!GetInferFlag()) {
    return RET_OK;
  }

  output->set_shape(input_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
