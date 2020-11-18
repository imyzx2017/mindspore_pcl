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

#include "src/runtime/kernel/arm/fp32/reduce_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "nnacl/fp32/reduce.h"
#include "src/runtime/kernel/arm/base/reduce_base.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Mean;
using mindspore::schema::PrimitiveType_Reduce;
using mindspore::schema::ReduceMode;
using mindspore::schema::ReduceMode_ReduceASum;
using mindspore::schema::ReduceMode_ReduceMax;
using mindspore::schema::ReduceMode_ReduceMean;
using mindspore::schema::ReduceMode_ReduceMin;
using mindspore::schema::ReduceMode_ReduceProd;
using mindspore::schema::ReduceMode_ReduceSum;
using mindspore::schema::ReduceMode_ReduceSumSquare;

namespace mindspore::kernel {

int ReduceCPUKernel::Init() {
  auto ret = ReduceBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }

  switch (mode_) {
    case static_cast<int>(ReduceMode_ReduceSum): {
      reducer_ = ReduceSum;
      break;
    }
    case static_cast<int>(ReduceMode_ReduceMean): {
      reducer_ = ReduceMean;
      break;
    }
    case static_cast<int>(ReduceMode_ReduceMax): {
      reducer_ = ReduceMax;
      break;
    }
    case static_cast<int>(ReduceMode_ReduceMin): {
      reducer_ = ReduceMin;
      int_reducer_ = IntReduceMin;
      break;
    }
    case static_cast<int>(ReduceMode_ReduceProd): {
      reducer_ = ReduceProd;
      int_reducer_ = IntReduceProd;
      break;
    }
    case static_cast<int>(ReduceMode_ReduceSumSquare): {
      reducer_ = ReduceSum;
      break;
    }
    case static_cast<int>(ReduceMode_ReduceASum): {
      reducer_ = ReduceSum;
      break;
    }
    default:
      MS_LOG(ERROR) << "Reduce unsupported reduce mode: " << mode_;
      return RET_ERROR;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ReduceCPUKernel::ReSize() { return ReduceBaseCPUKernel::ReSize(); }

int ReduceCPUKernel::CallReduceUnit(int task_id) {
  int ret;
  if (data_type_ == kDataTypeFloat) {
    ret = reducer_(outer_size_, inner_size_, axis_size_, static_cast<const float *>(src_data_),
                   static_cast<float *>(dst_data_), task_id, context_->thread_num_);
  } else {
    ret = int_reducer_(outer_size_, inner_size_, axis_size_, static_cast<const int *>(src_data_),
                       static_cast<int *>(dst_data_), task_id, context_->thread_num_);
  }

  return ret;
}

int ReduceImpl(void *cdata, int task_id) {
  auto reduce = reinterpret_cast<ReduceCPUKernel *>(cdata);
  auto error_code = reduce->CallReduceUnit(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Reduce Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ReduceCPUKernel::Run() {
  if (in_tensors().at(0)->data_type() == kNumberTypeFloat32) {
    data_type_ = kDataTypeFloat;
  } else {
    data_type_ = kDataTypeInt;
  }
  auto ret = MallocTmpBuffer();
  if (ret != RET_OK) {
    FreeTmpBuffer();
    return ret;
  }

  src_data_ = in_tensors_.at(0)->data_c();
  HandleASumAndSumSquare();
  for (size_t i = 0; i < static_cast<size_t>(num_axes_); ++i) {
    if (i != static_cast<size_t>(num_axes_ - 1)) {
      dst_data_ = data_buffers_[i];
    } else {
      dst_data_ = out_tensors_.at(0)->MutableData();
    }
    outer_size_ = outer_sizes_[i];
    inner_size_ = inner_sizes_[i];
    axis_size_ = axis_sizes_[i];
    auto error_code = ParallelLaunch(this->context_->thread_pool_, ReduceImpl, this, context_->thread_num_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "Reduce run error, error_code[" << error_code << "]";
      FreeTmpBuffer();
      return RET_ERROR;
    }
    src_data_ = dst_data_;
  }
  if (reduce_param_->reduce_to_end_ && reduce_param_->coeff - 1.0f > 1e-5) {
    ret = CalculateCoeffOutput();
    if (ret != RET_OK) {
      return ret;
    }
  }

  FreeTmpBuffer();
  return RET_OK;
}

void ReduceCPUKernel::HandleASumAndSumSquare() {
  if (data_type_ == kDataTypeInt) {
    return;
  }
  int num = in_tensors_.at(0)->ElementsNum();
  float *data = reinterpret_cast<float *>(in_tensors_.at(0)->data_c());
  if (data == nullptr) {
    return;
  }
  if (reduce_param_->mode_ == static_cast<int>(ReduceMode_ReduceASum)) {
    for (int i = 0; i < num; ++i) {
      if (data[i] < 0.0f) {
        data[i] = 0.0f - data[i];
      }
    }
  }
  if (reduce_param_->mode_ == static_cast<int>(ReduceMode_ReduceSumSquare)) {
    for (int i = 0; i < num; ++i) {
      data[i] = data[i] * data[i];
    }
  }
}

int ReduceCPUKernel::CalculateCoeffOutput() {
  auto out_tensor = out_tensors_.at(0);
  int num = out_tensor->ElementsNum();
  if (data_type_ != kDataTypeFloat) {
    return RET_ERROR;
  }
  float *out_data = reinterpret_cast<float *>(out_tensor->MutableData());
  if (out_data == nullptr) {
    return RET_NULL_PTR;
  }
  for (int i = 0; i < num; ++i) {
    out_data[i] *= reduce_param_->coeff;
  }
  return RET_OK;
}

int ReduceCPUKernel::MallocTmpBuffer() {
  data_buffers_.clear();
  for (auto size : buffer_sizes_) {
    void *buffer = nullptr;
    if (data_type_ == kDataTypeFloat) {
      buffer = context_->allocator->Malloc(size * sizeof(float));
    } else {
      buffer = context_->allocator->Malloc(size * sizeof(int));
    }
    if (buffer == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed.";
      return RET_ERROR;
    }
    data_buffers_.emplace_back(buffer);
  }
  return RET_OK;
}

void ReduceCPUKernel::FreeTmpBuffer() {
  for (auto buffer : data_buffers_) {
    if (buffer != nullptr) {
      context_->allocator->Free(buffer);
      buffer = nullptr;
    }
  }
  data_buffers_.clear();
}
}  // namespace mindspore::kernel
