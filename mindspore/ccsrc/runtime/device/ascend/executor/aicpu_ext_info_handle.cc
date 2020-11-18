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

#include "runtime/device/ascend/executor/aicpu_ext_info_handle.h"
#include <algorithm>
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/aicpu/aicpu_util.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
// if dim count is not reach kMaxShapeDims(8), use INT64_MIN to mark dim end.
constexpr int64_t kDimEndFlag = INT64_MIN;
}  // namespace
bool AicpuExtInfoHandler::Parse(const std::string &ext_info) {
  MS_LOG(INFO) << "Parse Node:" << node_name_ << " start";
  if (ext_info.empty()) {
    MS_LOG(ERROR) << "Node:" << node_name_ << " ext_info is empty";
    return false;
  }

  ext_info_len_ = ext_info.size();
  ext_info_.reset(new (std::nothrow) uint8_t[ext_info_len_]);
  MS_EXCEPTION_IF_NULL(ext_info_);

  (void)memcpy_s(ext_info_.get(), ext_info_len_, ext_info.c_str(), ext_info.size());

  input_shape_and_type_.clear();
  output_shape_and_type_.clear();

  auto ext_info_data = ext_info_.get();
  size_t offset = 0;
  while (offset + sizeof(AicpuExtInfo) <= ext_info_len_) {
    auto aicpu_ext_info = reinterpret_cast<AicpuExtInfo *>(ext_info_data + offset);
    MS_EXCEPTION_IF_NULL(aicpu_ext_info);
    switch (aicpu_ext_info->infoType) {
      case kernel::FWK_ADPT_EXT_SHAPE_TYPE:
        if (!ParseExtShapeType(aicpu_ext_info)) {
          MS_LOG(EXCEPTION) << "Parse ext shape type failed.";
        }
        break;
      case kernel::FWK_ADPT_EXT_INPUT_SHAPE:
        if (!ParseExtInputShape(aicpu_ext_info)) {
          MS_LOG(EXCEPTION) << "Parse ext input shape failed.";
        }
        break;
      case kernel::FWK_ADPT_EXT_OUTPUT_SHAPE:
        if (!ParseExtOutputShape(aicpu_ext_info)) {
          MS_LOG(EXCEPTION) << "Parse ext output shape failed.";
        }
        break;
      default:
        MS_LOG(INFO) << "Ignore Node:" << node_name_ << " infoType:" << aicpu_ext_info->infoType
                     << " infoLen:" << aicpu_ext_info->infoLen;
        break;
    }
    offset += sizeof(AicpuExtInfo);
    offset += aicpu_ext_info->infoLen;
  }

  if (offset != ext_info_len_) {
    MS_LOG(EXCEPTION) << "Node:" << node_name_ << " ext_info format error, parse not reach end, offset=" << offset
                      << ", ext_info_len" << ext_info_len_;
  }
  MS_LOG(INFO) << "Node:" << node_name_ << " parse ext info end.";
  return true;
}

bool AicpuExtInfoHandler::ParseExtShapeType(AicpuExtInfo *aicpu_ext_info) {
  if (aicpu_ext_info->infoLen != sizeof(int32_t)) {
    MS_LOG(ERROR) << "Node:" << node_name_ << " parse ext shape type failed as infoLen must be " << sizeof(int32_t)
                  << " but got:" << aicpu_ext_info->infoLen;
    return false;
  }

  auto type = reinterpret_cast<const int32_t *>(aicpu_ext_info->infoMsg);

  if (*type != unknown_type_) {
    MS_LOG(ERROR) << "Node:" << node_name_ << " parse ext shape type failed as need:" << unknown_type_
                  << " but got:" << *type;
  }
  MS_LOG(INFO) << "Node:" << node_name_ << "parse ext shape type success infoLen=" << aicpu_ext_info->infoLen;
  return true;
}

bool AicpuExtInfoHandler::ParseExtInputShape(AicpuExtInfo *aicpu_ext_info) {
  auto need_len = input_num_ * sizeof(AicpuShapeAndType);

  if (aicpu_ext_info->infoLen != need_len) {
    MS_LOG(ERROR) << "Node:" << node_name_
                  << " parse ext input shape failed as aicpu_ext_info->infoLen:" << aicpu_ext_info->infoLen
                  << " and need_len:" << need_len;
  }
  auto input = reinterpret_cast<AicpuShapeAndType *>(aicpu_ext_info->infoMsg);

  for (uint32_t index = 0; index < input_num_; ++index) {
    input_shape_and_type_.emplace_back(&input[index]);
  }
  MS_LOG(INFO) << "Node:" << node_name_.c_str() << " parse ext input shape success infoLen=" << aicpu_ext_info->infoLen;
  return true;
}

bool AicpuExtInfoHandler::ParseExtOutputShape(AicpuExtInfo *aicpu_ext_info) {
  auto need_len = output_num_ * sizeof(AicpuShapeAndType);
  if (aicpu_ext_info->infoLen != need_len) {
    MS_LOG(INFO) << "Node:" << node_name_
                 << " parse ext output shape failed, aicpu_ext_info->infoLen:" << aicpu_ext_info->infoLen
                 << " need_len:" << need_len;
    return false;
  }

  auto output = reinterpret_cast<AicpuShapeAndType *>(aicpu_ext_info->infoMsg);
  for (uint32_t index = 0; index < output_num_; ++index) {
    output_shape_and_type_.emplace_back(&output[index]);
  }
  MS_LOG(INFO) << "Node:" << node_name_ << " parse ext output shape success infoLen=" << aicpu_ext_info->infoLen;
  return true;
}

bool AicpuExtInfoHandler::UpdateInputShapeAndType(uint32_t input_index, const NotNull<AnfNodePtr> &anf_node) {
  if (input_index >= input_num_) {
    MS_LOG(ERROR) << "input_index=" << input_index << " >= input_num_:" << input_num_;
    return false;
  }

  auto input_shape = AnfAlgo::GetInputDeviceShape(anf_node, input_index);
  auto data_type = AnfAlgo::GetInputDeviceDataType(anf_node, input_index);
  std::vector<int64_t> tmp_shape;
  std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(tmp_shape), SizeToLong);
  return UpdateShapeAndType(tmp_shape, data_type, NOT_NULL(input_shape_and_type_[input_index]));
}

bool AicpuExtInfoHandler::UpdateOutputShapeAndType(uint32_t output_index, const NotNull<AnfNodePtr> &anf_node) {
  if (output_index >= output_num_) {
    MS_LOG(ERROR) << "output_index:" << output_index << " >= output_num_:" << output_num_;
    return false;
  }

  auto shape = AnfAlgo::GetOutputDeviceShape(anf_node, output_index);
  auto max_shape = AnfAlgo::GetOutputMaxShape(anf_node, output_index);
  if (shape.size() != max_shape.size()) {
    MS_LOG(ERROR) << "shape size != max_shape size";
    return true;
  }

  for (size_t i = 0; i < shape.size(); ++i) {
    if (i < max_shape.size() && shape[i] == SIZE_MAX) {
      MS_LOG(INFO) << "Node:" << node_name_ << " update shape from SIZE_MAX to " << max_shape[i];
      shape[i] = max_shape[i];
    }
  }

  std::vector<int64_t> tmp_shape;
  std::transform(shape.begin(), shape.end(), std::back_inserter(tmp_shape), SizeToLong);
  return UpdateShapeAndType(tmp_shape, AnfAlgo::GetOutputDeviceDataType(anf_node, output_index),
                            NOT_NULL(output_shape_and_type_[output_index]));
}

bool AicpuExtInfoHandler::GetOutputShapeAndType(uint32_t output_index, NotNull<std::vector<int64_t> *> shape,
                                                NotNull<TypeId *> data_type) {
  MS_LOG(INFO) << "Get " << node_name_ << " Output:" << output_index << " Shape And Type";
  GetShapeAndType(NOT_NULL(output_shape_and_type_[output_index]), shape, data_type);
  return true;
}

bool AicpuExtInfoHandler::UpdateShapeAndType(const std::vector<int64_t> &shape, TypeId data_type,
                                             NotNull<AicpuShapeAndType *> shape_and_type) {
  if (shape.empty() || shape.size() > kernel::kMaxShapeDims) {
    MS_LOG(ERROR) << "Invalid shape:" << shape.size();
    return false;
  }

  size_t index = 0;
  for (; index < shape.size(); ++index) {
    shape_and_type->dims[index] = shape[index];
  }
  if (index < kernel::kMaxShapeDims) {
    shape_and_type->dims[index] = kDimEndFlag;
  }

  // now only support update shape, type is not support
  return true;
}

void AicpuExtInfoHandler::GetShapeAndType(NotNull<const AicpuShapeAndType *> shape_and_type,
                                          NotNull<std::vector<int64_t> *> shape, NotNull<TypeId *> data_type) {
  for (int64_t tmpDim : shape_and_type->dims) {
    if (tmpDim == kDimEndFlag) {
      break;
    }
    shape->emplace_back(tmpDim);
    MS_LOG(INFO) << "Debug tmpDim:" << tmpDim;
  }

  auto ms_type = kernel::AicpuOpUtil::ProtoTypeToMsType(shape_and_type->type);
  if (ms_type == -1) {
    MS_LOG(EXCEPTION) << "Unspport Proto Type:" << shape_and_type->type;
  }
  MS_LOG(INFO) << "Debug ms_type:" << ms_type;
  *data_type = static_cast<TypeId>(ms_type);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
