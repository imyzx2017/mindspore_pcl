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

#include "src/runtime/kernel/arm/fp16/transpose_fp16.h"
#include <vector>
#include "nnacl/fp16/transpose_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "nnacl/fp16/cast_fp16.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_OP_EXECUTE_FAILURE;
using mindspore::schema::PrimitiveType_Transpose;

namespace mindspore::kernel {
int TransposeFp16CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TransposeFp16CPUKernel::ReSize() {
  TransposeParameter *param = reinterpret_cast<TransposeParameter *>(this->op_parameter_);
  num_unit_ = static_cast<int>(in_tensors_[kInputIndex]->shape().at(param->perm_[kNHWC_H]));
  thread_h_num_ = MSMIN(thread_num_, num_unit_);
  thread_h_stride_ = UP_DIV(num_unit_, thread_h_num_);
  auto &in_tensor = in_tensors_.front();
  auto &out_tensor = out_tensors_.front();
  auto in_shape = in_tensor->shape();
  auto out_shape = out_tensor->shape();
  param->strides_[param->num_axes_ - 1] = 1;
  param->out_strides_[param->num_axes_ - 1] = 1;
  param->data_size_ = in_tensor->Size();
  for (int i = param->num_axes_ - 2; i >= 0; i--) {
    param->strides_[i] = in_shape[i + 1] * param->strides_[i + 1];
    param->out_strides_[i] = out_shape[i + 1] * param->out_strides_[i + 1];
  }

  return RET_OK;
}

int TransposeFp16CPUKernel::MallocFp16Buffer() {
  auto &in_tensor = in_tensors_.front();
  auto &out_tensor = out_tensors_.front();

  if (in_tensor->data_type() == kNumberTypeFloat || in_tensor->data_type() == kNumberTypeFloat32) {
    fp16_in_data_ =
      reinterpret_cast<float16_t *>(context_->allocator->Malloc(sizeof(float16_t) * in_tensor->ElementsNum()));
    if (fp16_in_data_ == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed";
      return RET_ERROR;
    }
  }
  if (out_tensor->data_type() == kNumberTypeFloat || out_tensor->data_type() == kNumberTypeFloat32) {
    fp16_out_data_ =
      reinterpret_cast<float16_t *>(context_->allocator->Malloc(sizeof(float16_t) * out_tensor->ElementsNum()));
    if (fp16_out_data_ == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void TransposeFp16CPUKernel::FreeFp16Buffer() {
  auto &in_tensor = in_tensors_.front();
  auto &out_tensor = out_tensors_.front();

  if (in_tensor->data_type() == kNumberTypeFloat || in_tensor->data_type() == kNumberTypeFloat32) {
    if (fp16_in_data_ != nullptr) {
      context_->allocator->Free(fp16_in_data_);
      fp16_in_data_ = nullptr;
    }
  }
  if (out_tensor->data_type() == kNumberTypeFloat || out_tensor->data_type() == kNumberTypeFloat32) {
    if (fp16_out_data_ != nullptr) {
      context_->allocator->Free(fp16_out_data_);
      fp16_out_data_ = nullptr;
    }
  }
}

int TransposeFp16CPUKernel::TransposeParallel(int task_id) {
  int num_unit_thread = MSMIN(thread_h_stride_, num_unit_ - task_id * thread_h_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_h_stride_;
  TransposeParameter *param = reinterpret_cast<TransposeParameter *>(this->op_parameter_);

  auto ret = Fp16DoTranspose(fp16_in_data_, fp16_out_data_, in_shape_, out_shape_, param, thread_offset,
                             thread_offset + num_unit_thread);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Transpose error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }

  return RET_OK;
}

static int TransposeFp16Run(void *cdata, int task_id) {
  auto g_kernel = reinterpret_cast<TransposeFp16CPUKernel *>(cdata);
  auto ret = g_kernel->TransposeParallel(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "TransposeRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_OP_EXECUTE_FAILURE;
  }
  return RET_OK;
}

int TransposeFp16CPUKernel::Run() {
  MS_ASSERT(in_tensors_.size() == 1);
  MS_ASSERT(out_tensors_.size() == 1);
  auto &in_tensor = in_tensors_.front();
  auto &out_tensor = out_tensors_.front();
  if (in_tensor == nullptr || out_tensor == nullptr) {
    MS_LOG(ERROR) << "null pointer referencing.";
    return RET_ERROR;
  }

  // malloc when Run
  auto ret = MallocFp16Buffer();
  if (ret != RET_OK) {
    FreeFp16Buffer();
    return ret;
  }

  if (in_tensor->data_type() == kNumberTypeFloat || in_tensor->data_type() == kNumberTypeFloat32) {
    in_data_ = reinterpret_cast<float *>(in_tensor->MutableData());
    Float32ToFloat16(in_data_, fp16_in_data_, in_tensor->ElementsNum());
  } else {
    fp16_in_data_ = reinterpret_cast<float16_t *>(in_tensor->MutableData());
  }
  if (out_tensor->data_type() == kNumberTypeFloat16) {
    fp16_out_data_ = reinterpret_cast<float16_t *>(out_tensor->MutableData());
  }

  in_shape_ = const_cast<int *>(in_tensor->shape().data());
  out_shape_ = const_cast<int *>(out_tensor->shape().data());

  ret = ParallelLaunch(this->context_->thread_pool_, TransposeFp16Run, this, thread_h_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Tranpose error error_code[" << ret << "]";
    FreeFp16Buffer();
    return ret;
  }

  if (out_tensor->data_type() == kNumberTypeFloat || out_tensor->data_type() == kNumberTypeFloat32) {
    out_data_ = reinterpret_cast<float *>(out_tensor->MutableData());
    Float16ToFloat32(fp16_out_data_, out_data_, out_tensor->ElementsNum());
  }
  FreeFp16Buffer();

  return ret;
}

kernel::LiteKernel *CpuTransposeFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                  const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                  const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_Transpose);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "desc type is not Transpose";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) TransposeFp16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New kernel fails.";
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

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Transpose, CpuTransposeFp16KernelCreator)
}  // namespace mindspore::kernel
