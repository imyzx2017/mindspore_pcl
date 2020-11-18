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

#include <vector>
#include "src/runtime/kernel/arm/base/resize_base.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/arm/fp32/resize_fp32.h"
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INVALID_OP_ATTR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
constexpr int kMaxInputNum = 2;
constexpr int kOutputNum = 1;
constexpr int kRank = 4;
}  // namespace

int ResizeBaseCPUKernel::CheckParameters() {
  auto parameter = reinterpret_cast<ResizeParameter *>(op_parameter_);
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "cast ResizeParameter failed.";
    return RET_NULL_PTR;
  }
  method_ = parameter->method_;
  if (method_ != static_cast<int>(schema::ResizeMethod_LINEAR) &&
      method_ != static_cast<int>(schema::ResizeMethod_NEAREST)) {
    MS_LOG(ERROR) << "Resize method should be bilinear or nearest_neighbor, but got " << method_;
    return RET_INVALID_OP_ATTR;
  }
  if (this->in_tensors_.size() == lite::kSingleNum) {
    new_height_ = parameter->new_height_;
    if (new_height_ < 1) {
      MS_LOG(ERROR) << "Resize new_height should >= 1, but got " << new_height_;
      return RET_INVALID_OP_ATTR;
    }
    new_width_ = parameter->new_width_;
    if (new_width_ < 1) {
      MS_LOG(ERROR) << "Resize new_width should >= 1, but got " << new_width_;
      return RET_INVALID_OP_ATTR;
    }
  } else if (this->in_tensors_.size() == lite::kDoubleNum) {
    auto out_shape = this->in_tensors_[1]->data_c();
    if (out_shape == nullptr) {
      MS_LOG(INFO) << "Out shape is not assigned";
      const_shape_ = false;
    } else {
      new_height_ = reinterpret_cast<int32_t *>(out_shape)[0];
      if (new_height_ < 1) {
        MS_LOG(ERROR) << "Resize new_height should >= 1, but got " << new_height_;
        return RET_INVALID_OP_ATTR;
      }
      new_width_ = reinterpret_cast<int32_t *>(out_shape)[1];
      if (new_width_ < 1) {
        MS_LOG(ERROR) << "Resize new_width should >= 1, but got " << new_width_;
        return RET_INVALID_OP_ATTR;
      }
      const_shape_ = true;
    }
  }
  align_corners_ = parameter->align_corners_;
  preserve_aspect_ratio = parameter->preserve_aspect_ratio_;
  if (preserve_aspect_ratio) {
    MS_LOG(ERROR) << "Resize currently not support preserve_aspect_ratio true";
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeBaseCPUKernel::CheckInputsOuputs() {
  if (in_tensors_.size() <= lite::kDoubleNum) {
    for (size_t i = 0; i < in_tensors_.size(); i++) {
      auto input = in_tensors_.at(i);
      if (input == nullptr) {
        return RET_NULL_PTR;
      }
    }
  } else {
    MS_LOG(ERROR) << "Resize input num should be no more than" << kMaxInputNum << ", but got " << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != kOutputNum) {
    MS_LOG(ERROR) << "Resize output num should be " << kOutputNum << ", but got " << out_tensors_.size();
    return RET_ERROR;
  }
  auto output = out_tensors_.at(0);
  if (output == nullptr) {
    return RET_NULL_PTR;
  }
  return RET_OK;
}

int ResizeBaseCPUKernel::Init() {
  auto ret = CheckParameters();
  if (ret != RET_OK) {
    return ret;
  }
  ret = CheckInputsOuputs();
  if (ret != RET_OK) {
    return ret;
  }

  auto input = in_tensors_.at(0);
  auto input_shape = input->shape();
  if (!input_shape.empty() && input_shape.size() != kRank) {
    MS_LOG(ERROR) << "Resize op support input rank 4, got " << input_shape.size();
    return RET_ERROR;
  }

  return RET_OK;
}

kernel::LiteKernel *CpuResizeFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                               const mindspore::lite::PrimitiveC *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Resize);
  auto *kernel = new (std::nothrow) ResizeCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ResizeCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }

  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Resize, CpuResizeFp32KernelCreator)
}  // namespace mindspore::kernel
