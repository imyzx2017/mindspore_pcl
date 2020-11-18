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

#include "src/runtime/kernel/arm/fp16/reduce_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "nnacl/fp16/reduce_fp16.h"
#include "src/runtime/kernel/arm/base/reduce_base.h"
#include "nnacl/fp16/cast_fp16.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Mean;
using mindspore::schema::PrimitiveType_Reduce;
using mindspore::schema::ReduceMode;
using mindspore::schema::ReduceMode_ReduceMax;
using mindspore::schema::ReduceMode_ReduceMean;
using mindspore::schema::ReduceMode_ReduceMin;
using mindspore::schema::ReduceMode_ReduceProd;
using mindspore::schema::ReduceMode_ReduceSum;
using mindspore::schema::ReduceMode_ReduceSumSquare;

namespace mindspore::kernel {

int ReduceFp16CPUKernel::Init() {
  auto ret = ReduceBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  if (mode_ != static_cast<int>(ReduceMode_ReduceMean)) {
    MS_LOG(ERROR) << "Reduce fp16 only support ReduceMode_ReduceMean";
    return RET_ERROR;
  }
  reducer_ = ReduceMeanFp16;

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ReduceFp16CPUKernel::ReSize() { return ReduceBaseCPUKernel::ReSize(); }

int ReduceFp16CPUKernel::CallReduceUnit(int task_id) {
  auto ret =
    reducer_(outer_size_, inner_size_, axis_size_, fp16_src_data_, fp16_dst_data_, task_id, context_->thread_num_);
  return ret;
}

static int ReduceFp16Impl(void *cdata, int task_id) {
  auto reduce = reinterpret_cast<ReduceFp16CPUKernel *>(cdata);
  auto error_code = reduce->CallReduceUnit(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Reduce Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ReduceFp16CPUKernel::Run() {
  auto ret = MallocTmpBuffer();
  if (ret != RET_OK) {
    FreeTmpBuffer();
    return ret;
  }

  auto in_tensor = in_tensors_.at(0);
  if (in_tensor->data_type() == kNumberTypeFloat32 || in_tensor->data_type() == kNumberTypeFloat) {
    auto input_data = reinterpret_cast<float *>(in_tensor->MutableData());
    Float32ToFloat16(input_data, fp16_input_, in_tensor->ElementsNum());
  } else {
    fp16_input_ = reinterpret_cast<float16_t *>(in_tensor->MutableData());
  }

  fp16_src_data_ = fp16_input_;
  for (size_t i = 0; i < data_buffers_.size(); ++i) {
    fp16_dst_data_ = data_buffers_[i];
    outer_size_ = outer_sizes_[i];
    inner_size_ = inner_sizes_[i];
    axis_size_ = axis_sizes_[i];
    auto error_code = ParallelLaunch(this->context_->thread_pool_, ReduceFp16Impl, this, context_->thread_num_);
    if (error_code != RET_OK) {
      FreeTmpBuffer();
      MS_LOG(ERROR) << "Reduce run error, error_code[" << error_code << "]";
      return RET_ERROR;
    }
    fp16_src_data_ = fp16_dst_data_;
  }

  auto out_tensor = out_tensors_.at(0);
  if (out_tensor->data_type() == kNumberTypeFloat32 || out_tensor->data_type() == kNumberTypeFloat) {
    dst_data_ = reinterpret_cast<float *>(out_tensor->MutableData());
    Float16ToFloat32(fp16_dst_data_, dst_data_, out_tensor->ElementsNum());
  } else {
    memcpy(out_tensor->MutableData(), fp16_dst_data_, out_tensor->ElementsNum() * sizeof(float16_t));
  }

  FreeTmpBuffer();
  return RET_OK;
}

void ReduceFp16CPUKernel::FreeTmpBuffer() {
  for (auto buffer : data_buffers_) {
    if (buffer != nullptr) {
      context_->allocator->Free(buffer);
      buffer = nullptr;
    }
  }
  data_buffers_.clear();

  auto in_tensor = in_tensors_.at(0);
  if (in_tensor->data_type() == kNumberTypeFloat32 || in_tensor->data_type() == kNumberTypeFloat) {
    if (fp16_input_ != nullptr) {
      context_->allocator->Free(fp16_input_);
      fp16_input_ = nullptr;
    }
  }
}

int ReduceFp16CPUKernel::MallocTmpBuffer() {
  data_buffers_.clear();
  for (auto size : buffer_sizes_) {
    float16_t *buffer = reinterpret_cast<float16_t *>(context_->allocator->Malloc(size * sizeof(float16_t)));
    if (buffer == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed";
      return RET_ERROR;
    }
    data_buffers_.emplace_back(buffer);
  }

  auto in_tensor = in_tensors_.front();
  if (in_tensor->data_type() == kNumberTypeFloat32 || in_tensor->data_type() == kNumberTypeFloat) {
    fp16_input_ =
      reinterpret_cast<float16_t *>(context_->allocator->Malloc(in_tensor->ElementsNum() * sizeof(float16_t)));
    if (fp16_input_ == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

kernel::LiteKernel *CpuReduceFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
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
    return nullptr;
  }
  auto *kernel = new (std::nothrow) ReduceFp16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Reduce new ReduceCPUKernel failed.";
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

kernel::LiteKernel *CpuMeanFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
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
  auto *kernel = new (std::nothrow) ReduceFp16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
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

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Reduce, CpuReduceFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Mean, CpuMeanFp16KernelCreator)
}  // namespace mindspore::kernel
