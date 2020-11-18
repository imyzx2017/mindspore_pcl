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

#include "src/runtime/kernel/arm/base/reduce_base.h"
#include "src/kernel_registry.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "src/runtime/kernel/arm/fp32/reduce_fp32.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Mean;
using mindspore::schema::PrimitiveType_Reduce;

namespace mindspore::kernel {
namespace {
constexpr size_t kInputNum = 1;
constexpr size_t kOutputNum = 1;
}  // namespace

int ReduceBaseCPUKernel::CheckInputsOutputs() {
  if (in_tensors_.size() < kInputNum) {
    MS_LOG(ERROR) << "Reduce inputs size should be at least " << kInputNum << " but got " << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != kOutputNum) {
    MS_LOG(ERROR) << "Reduce outputs size should be " << kOutputNum << " but got " << out_tensors_.size();
    return RET_ERROR;
  }
  auto input = in_tensors_.at(0);
  if (input == nullptr) {
    MS_LOG(ERROR) << "Reduce input is nullptr";
    return RET_NULL_PTR;
  }
  auto output = out_tensors_.at(0);
  if (output == nullptr) {
    MS_LOG(ERROR) << "Reduce output is nullptr";
    return RET_NULL_PTR;
  }
  return RET_OK;
}

int ReduceBaseCPUKernel::CheckParameters() {
  size_t input_rank = in_tensors_.at(0)->shape().size();
  if (static_cast<size_t>(num_axes_) > input_rank) {
    MS_LOG(ERROR) << "Reduce op invalid num of reduce axes " << num_axes_ << " larger than input rank " << input_rank;
    return RET_ERROR;
  }

  for (auto i = 0; i < num_axes_; i++) {
    if (axes_[i] < -static_cast<int>(input_rank) || axes_[i] >= static_cast<int>(input_rank)) {
      MS_LOG(ERROR) << "Reduce got invalid axis " << axes_[i] << ", axis should be in ["
                    << -static_cast<int>(input_rank) << ", " << input_rank - 1 << "].";
      return RET_ERROR;
    }
    if (axes_[i] < 0) {
      axes_[i] += static_cast<int>(input_rank);
    }
  }

  if (reduce_to_end_) {
    // actual num of axes to reduce
    num_axes_ = static_cast<int>(input_rank) - axes_[0];
    for (auto i = 1; i < num_axes_; ++i) {
      axes_[i] = axes_[0] + i;
    }
  }

  if (num_axes_ == 0) {
    for (size_t i = 0; i < input_rank; i++) {
      axes_[i] = i;
    }
    num_axes_ = static_cast<int>(input_rank);
  }

  return RET_OK;
}

int ReduceBaseCPUKernel::Init() {
  auto reduce_param = reinterpret_cast<ReduceParameter *>(op_parameter_);
  if (reduce_param == nullptr) {
    return RET_NULL_PTR;
  }
  if (in_tensors_.size() > 1) {
    auto axes_ptr = in_tensors_.at(1);
    num_axes_ = axes_ptr->ElementsNum();
    memcpy(axes_, axes_ptr->MutableData(), axes_ptr->Size());
  } else {
    num_axes_ = reduce_param->num_axes_;
    memcpy(axes_, reduce_param->axes_, sizeof(reduce_param->axes_));
  }

  mode_ = reduce_param->mode_;
  reduce_to_end_ = reduce_param->reduce_to_end_;

  auto ret = CheckInputsOutputs();
  if (ret != RET_OK) {
    return ret;
  }

  return RET_OK;
}

void ReduceBaseCPUKernel::CalculateInnerOuterSize() {
  outer_sizes_.clear();
  inner_sizes_.clear();
  axis_sizes_.clear();
  auto tmp_shape = in_tensors_.at(0)->shape();
  for (auto i = 0; i < num_axes_; ++i) {
    int axis = axes_[i];
    auto outer_size = 1;
    for (int j = 0; j < axis; j++) {
      outer_size *= tmp_shape[j];
    }
    outer_sizes_.emplace_back(outer_size);
    auto inner_size = 1;
    for (int k = axis + 1; k < static_cast<int>(tmp_shape.size()); k++) {
      inner_size *= tmp_shape[k];
    }
    inner_sizes_.emplace_back(inner_size);
    axis_sizes_.emplace_back(tmp_shape[axis]);
    tmp_shape[axis] = 1;
  }
}

void ReduceBaseCPUKernel::CalculateTmpBufferSize() {
  buffer_sizes_.clear();
  auto input_shape = in_tensors_.at(0)->shape();
  for (auto i = 0; i < num_axes_; i++) {
    int axis = axes_[i];
    size_t size = 1;
    for (size_t j = 0; j < input_shape.size(); j++) {
      if (axis != static_cast<int>(j)) {
        size *= input_shape[j];
      }
    }
    MS_ASSERT(context_->allocator != nullptr);
    buffer_sizes_.emplace_back(size);
    input_shape[axis] = 1;
  }
}

int ReduceBaseCPUKernel::ReSize() {
  auto ret = CheckParameters();
  if (ret != RET_OK) {
    return ret;
  }
  CalculateTmpBufferSize();
  CalculateInnerOuterSize();
  return RET_OK;
}

kernel::LiteKernel *CpuReduceFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                               const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Reduce);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Reduce opParameter nullptr";
    return nullptr;
  }
  if (desc.type != schema::PrimitiveType_Reduce) {
    MS_LOG(ERROR) << "Reduce op desc.type should be PrimitiveType_Reduce, got " << desc.type;
    free(opParameter);
    return nullptr;
  }
  auto *kernel = new (std::nothrow) ReduceCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Reduce new ReduceCPUKernel failed.";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

kernel::LiteKernel *CpuMeanFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                             const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                             const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Mean);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Reduce opParameter nullptr";
    return nullptr;
  }
  if (desc.type != schema::PrimitiveType_Mean) {
    MS_LOG(ERROR) << "Reduce op desc.type should be PrimitiveType_Mean, got " << desc.type;
    free(opParameter);
    return nullptr;
  }
  auto *kernel = new (std::nothrow) ReduceCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Reduce new ReduceCPUKernel failed.";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Reduce, CpuReduceFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt, PrimitiveType_Reduce, CpuReduceFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Reduce, CpuReduceFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Mean, CpuMeanFp32KernelCreator)

}  // namespace mindspore::kernel
