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

#include "src/ops/slice.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
namespace {
constexpr int kSliceInputNum = 1;
constexpr int kSliceOutputNum = 1;
constexpr int kSliceMaxInputNum = 5;
}  // namespace
#ifdef PRIMITIVE_WRITEABLE
int Slice::GetFormat() const { return this->primitive_->value.AsSlice()->format; }
std::vector<int> Slice::GetBegin() const { return this->primitive_->value.AsSlice()->begin; }
std::vector<int> Slice::GetSize() const { return this->primitive_->value.AsSlice()->size; }
std::vector<int> Slice::GetAxes() const { return this->primitive_->value.AsSlice()->axes; }

void Slice::SetFormat(int format) { this->primitive_->value.AsSlice()->format = (schema::Format)format; }
void Slice::SetBegin(const std::vector<int> &begin) { this->primitive_->value.AsSlice()->begin = begin; }
void Slice::SetSize(const std::vector<int> &size) { this->primitive_->value.AsSlice()->size = size; }

int Slice::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Slice;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Slice) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::SliceT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    if (inputs.size() >= kAnfPopulaterThree) {
      auto beginNode = inputs[kAnfPopulaterOne];
      MS_ASSERT(beginNode != nullptr);
      if (beginNode->isa<ValueNode>()) {
        auto valueNode = beginNode->cast<ValueNodePtr>();
        MS_ASSERT(valueNode != nullptr);
        auto value = valueNode->value();
        MS_ASSERT(value != nullptr);
        if (value->isa<ValueTuple>()) {
          auto valTuplPtr = dyn_cast<ValueTuple>(value);
          MS_ASSERT(valTuplPtr != nullptr);
          for (size_t i = 0; i < valTuplPtr->size(); i++) {
            auto elem = dyn_cast<Int32Imm>((*valTuplPtr)[i]);
            MS_ASSERT(elem != nullptr);
            attr->begin.emplace_back(elem->value());
          }
        }
      }
      auto sizeNode = inputs[kAnfPopulaterTwo];
      MS_ASSERT(sizeNode != nullptr);
      if (sizeNode->isa<ValueNode>()) {
        auto valueNode = sizeNode->cast<ValueNodePtr>();
        MS_ASSERT(valueNode != nullptr);
        auto value = valueNode->value();
        MS_ASSERT(value != nullptr);
        if (value->isa<ValueTuple>()) {
          auto valTuplPtr = dyn_cast<ValueTuple>(value);
          MS_ASSERT(valTuplPtr != nullptr);
          for (size_t i = 0; i < valTuplPtr->size(); i++) {
            auto elem = dyn_cast<Int32Imm>((*valTuplPtr)[i]);
            MS_ASSERT(elem != nullptr);
            attr->size.emplace_back(elem->value());
          }
        }
      }
      std::vector<int> axes;
      axes.clear();
      for (size_t i = 0; i < attr->begin.size(); i++) {
        axes.push_back(i);
      }
      attr->axes = axes;
    }
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}

#else

int Slice::GetFormat() const { return this->primitive_->value_as_Slice()->format(); }
std::vector<int> Slice::GetBegin() const {
  auto fb_vector = this->primitive_->value_as_Slice()->begin();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> Slice::GetSize() const {
  auto fb_vector = this->primitive_->value_as_Slice()->size();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

std::vector<int> Slice::GetAxes() const {
  auto fb_vector = this->primitive_->value_as_Slice()->axes();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

int Slice::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);

  auto attr = primitive->value_as_Slice();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Slice return nullptr";
    return RET_ERROR;
  }

  std::vector<int32_t> axes;
  if (attr->axes() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->axes()->size()); i++) {
      axes.push_back(attr->axes()->data()[i]);
    }
  }
  std::vector<int32_t> begin;
  if (attr->begin() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->begin()->size()); i++) {
      begin.push_back(attr->begin()->data()[i]);
    }
  }
  std::vector<int32_t> size;
  if (attr->size() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->size()->size()); i++) {
      size.push_back(attr->size()->data()[i]);
    }
  }

  auto val_offset = schema::CreateSliceDirect(*fbb, attr->format(), &axes, &begin, &size);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Slice, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *SliceCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Slice>(primitive); }
Registry SliceRegistry(schema::PrimitiveType_Slice, SliceCreator);

#endif

std::vector<int> Slice::GetPostProcessBegin() const { return this->begin; }
std::vector<int> Slice::GetPostProcessSize() const { return this->size; }
int Slice::InferShape(std::vector<lite::Tensor *> inputs, std::vector<lite::Tensor *> outputs) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (inputs.size() < kSliceInputNum || outputs.size() != kSliceOutputNum) {
    MS_LOG(ERROR) << "input size:" << inputs.size() << ",output size:" << outputs.size();
    return RET_PARAM_INVALID;
  }
  auto input = inputs.at(0);
  outputs[0]->set_data_type(input->data_type());
  outputs[0]->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  auto input_shape = input->shape();
  std::vector<int32_t> slice_begin(GetBegin());
  std::vector<int32_t> slice_size(GetSize());
  std::vector<int32_t> slice_axes(GetAxes());
  std::vector<int32_t> output_shape(input_shape.size());
  if (inputs.size() == kSliceMaxInputNum) {
    if (slice_begin.empty() && inputs.at(1)->data_c() != nullptr) {
      for (int i = 0; i < inputs.at(1)->ElementsNum(); i++) {
        slice_begin.emplace_back(static_cast<int *>(inputs.at(1)->data_c())[i]);
      }
    }
    if (slice_size.empty() && inputs.at(2)->data_c() != nullptr) {
      for (int i = 0; i < inputs.at(2)->ElementsNum(); i++) {
        auto end = static_cast<int *>(inputs.at(2)->data_c())[i];
        auto size = end < 0 ? end : (end == INT32_MAX ? -1 : end - slice_begin[i]);
        slice_size.emplace_back(size);
      }
    }
    if (slice_axes.empty() && inputs.at(3)->data_c() != nullptr) {
      for (int i = 0; i < inputs.at(3)->ElementsNum(); i++) {
        slice_axes.emplace_back(static_cast<int *>(inputs.at(3)->data_c())[i]);
      }
    }
  }
  if (slice_begin.empty() || slice_size.empty() || slice_axes.empty()) {
    MS_LOG(ERROR) << "Infershape failed.";
    return RET_INFER_INVALID;
  }
  begin.assign(input_shape.size(), 0);
  size.assign(input_shape.size(), -1);
  for (size_t i = 0; i < slice_axes.size(); ++i) {
    begin[slice_axes[i]] = slice_begin[i];
    size[slice_axes[i]] = slice_size[i];
  }
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (size[i] < 0 && size[i] != -1) {
      MS_LOG(ERROR) << "Invalid size input!size[" << i << "]=" << size[i];
      return RET_PARAM_INVALID;
    }
    if (begin[i] < 0) {
      MS_LOG(ERROR) << "Invalid begin input " << begin[i] << " which should be >= 0";
      return RET_PARAM_INVALID;
    }
    if (input_shape[i] <= begin[i]) {
      MS_LOG(ERROR) << "Invalid begin input!begin[" << i << "]=" << begin[i]
                    << " which should be <= " << input_shape[i];
      return RET_PARAM_INVALID;
    }
    if (size[i] > (input_shape[i] - begin[i])) {
      MS_LOG(ERROR) << "Invalid size input " << size[i] << " which should be <= " << input_shape[i] - begin[i];
      return RET_PARAM_INVALID;
    }

    output_shape[i] = size[i] < 0 ? input_shape[i] - begin[i] : size[i];
  }

  outputs[0]->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
