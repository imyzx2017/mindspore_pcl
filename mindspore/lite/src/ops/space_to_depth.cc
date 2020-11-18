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

#include "src/ops/space_to_depth.h"
#include "src/common/common.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int SpaceToDepth::GetBlockSize() const { return this->primitive_->value.AsSpaceToDepth()->blockSize; }
int SpaceToDepth::GetFormat() const { return this->primitive_->value.AsSpaceToDepth()->format; }

void SpaceToDepth::SetBlockSize(int block_size) { this->primitive_->value.AsSpaceToDepth()->blockSize = block_size; }
void SpaceToDepth::SetFormat(int format) { this->primitive_->value.AsSpaceToDepth()->format = (schema::Format)format; }

#else

int SpaceToDepth::GetBlockSize() const { return this->primitive_->value_as_SpaceToDepth()->blockSize(); }
int SpaceToDepth::GetFormat() const { return this->primitive_->value_as_SpaceToDepth()->format(); }
int SpaceToDepth::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_SpaceToDepth();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_SpaceToDepth return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateSpaceToDepth(*fbb, attr->blockSize(), attr->format());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_SpaceToDepth, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *SpaceToDepthCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<SpaceToDepth>(primitive);
}
Registry SpaceToDepthRegistry(schema::PrimitiveType_SpaceToDepth, SpaceToDepthCreator);
#endif

namespace {
constexpr int kSpaceToDepthOutputNum = 1;
constexpr int kSpaceToDepthInputNum = 1;
}  // namespace

int SpaceToDepth::InferShape(std::vector<lite::Tensor *> inputs, std::vector<lite::Tensor *> outputs) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (outputs.size() != kSpaceToDepthOutputNum || inputs.size() != kSpaceToDepthInputNum) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs.size() << ",input size: " << inputs.size();
    return 1;
  }

  auto input = inputs.at(0);
  if (input->GetFormat() != schema::Format::Format_NHWC) {
    MS_LOG(ERROR) << "space_to_depth only support NHWC now!";
    return 1;
  }
  outputs[0]->SetFormat(input->GetFormat());
  outputs[0]->set_data_type(input->data_type());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  auto input_shape = input->shape();
  if (input_shape.size() != kDimension_4d) {
    MS_LOG(ERROR) << "input shape dimension size should == " << kDimension_4d;
    return 1;
  }

  int32_t block_size = GetBlockSize();
  if (input_shape[NHWC_H] % block_size != 0 || input_shape[NHWC_H] == 0 || input_shape[NHWC_W] % block_size != 0 ||
      input_shape[NHWC_W] == 0) {
    MS_LOG(ERROR) << "input dimension h or w size error!";
    return 1;
  }
  std::vector<int32_t> output_shape(input_shape.size());
  output_shape[NHWC_N] = input_shape[NHWC_N];
  output_shape[NHWC_H] = input_shape[NHWC_H] / block_size;
  output_shape[NHWC_W] = input_shape[NHWC_W] / block_size;
  output_shape[NHWC_C] = input_shape[NHWC_C] * (block_size * block_size);
  outputs[0]->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
