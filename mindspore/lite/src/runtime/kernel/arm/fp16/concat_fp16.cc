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
#include "src/runtime/kernel/arm/fp16/concat_fp16.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "src/runtime/kernel/arm/fp32/concat_fp32.h"
#include "nnacl/fp16/concat_fp16.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp16/cast_fp16.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Concat;

namespace mindspore::kernel {
int ConcatFp16CPUKernel::Init() {
  auto ret = ConcatBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int ConcatFp16CPUKernel::ReSize() { return ConcatBaseCPUKernel::ReSize(); }

int ConcatFp16CPUKernel::MallocTmpBuffer() {
  for (const auto &in_tensor : in_tensors_) {
    float16_t *ptr = nullptr;
    if (in_tensor->data_type() == kNumberTypeFloat32 || in_tensor->data_type() == kNumberTypeFloat) {
      ptr = reinterpret_cast<float16_t *>(context_->allocator->Malloc(sizeof(float16_t) * in_tensor->ElementsNum()));
      if (ptr == nullptr) {
        MS_LOG(ERROR) << "malloc failed";
        return RET_ERROR;
      }
    }
    fp16_inputs_.push_back(ptr);
  }

  auto &out_tensor = out_tensors_.at(0);
  if (out_tensor->data_type() == kNumberTypeFloat32 || out_tensor->data_type() == kNumberTypeFloat) {
    fp16_output_ =
      reinterpret_cast<float16_t *>(context_->allocator->Malloc(sizeof(float16_t) * out_tensors_[0]->ElementsNum()));
    if (fp16_output_ == nullptr) {
      MS_LOG(ERROR) << "malloc failed";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

void ConcatFp16CPUKernel::FreeTmpBuffer() {
  for (size_t i = 0; i < fp16_inputs_.size(); i++) {
    auto &in_tensor = in_tensors_.at(i);
    auto in_ptr = fp16_inputs_.at(i);
    if (in_tensor->data_type() == kNumberTypeFloat32 || in_tensor->data_type() == kNumberTypeFloat) {
      if (in_ptr != nullptr) {
        context_->allocator->Free(in_ptr);
        in_ptr = nullptr;
      }
    }
  }
  fp16_inputs_.clear();

  auto &out_tensor = out_tensors_.at(0);
  if (out_tensor->data_type() == kNumberTypeFloat32 || out_tensor->data_type() == kNumberTypeFloat) {
    if (fp16_output_ != nullptr) {
      context_->allocator->Free(fp16_output_);
      fp16_output_ = nullptr;
    }
  }
}

int ConcatFp16CPUKernel::Run() {
  auto ret = MallocTmpBuffer();
  if (ret != RET_OK) {
    FreeTmpBuffer();
    return ret;
  }

  auto input_num = in_tensors_.size();
  std::vector<int *> inputs_output_shape(input_num + 1, nullptr);

  std::vector<std::vector<int>> shapes;
  for (size_t i = 0; i < input_num; ++i) {
    const auto in_tensor = in_tensors_[i];
    if (in_tensor->data_type() == kNumberTypeFloat || in_tensor->data_type() == kNumberTypeFloat32) {
      auto in_tensor_data = reinterpret_cast<float *>(in_tensor->MutableData());
      if (in_tensor_data == nullptr) {
        MS_LOG(ERROR) << "got nullptr when cast in_tensor to float ptr";
        return RET_ERROR;
      }
      Float32ToFloat16(in_tensor_data, fp16_inputs_[i], in_tensor->ElementsNum());
    } else {
      fp16_inputs_[i] = reinterpret_cast<float16_t *>(in_tensor->MutableData());
    }

    shapes.push_back(in_tensors_[i]->shape());
    inputs_output_shape[i] = shapes[i].data();
  }
  auto output_shape = out_tensors_.at(0)->shape();
  inputs_output_shape[input_num] = output_shape.data();
  auto output_addr = out_tensors_.at(0)->MutableData();
  if (out_tensors_.at(0)->data_type() == kNumberTypeFloat16) {
    fp16_output_ = reinterpret_cast<float16_t *>(out_tensors_.at(0)->MutableData());
  }
  int dtype_len = in_tensors_.at(0)->data_type() == kNumberTypeInt32 ? sizeof(int32_t) : sizeof(float16_t);
  ConcatFp16(reinterpret_cast<void **>(fp16_inputs_.data()), input_num, axis_, inputs_output_shape.data(),
             output_shape.size(), reinterpret_cast<void *>(fp16_output_), dtype_len);

  if (out_tensors_.at(0)->data_type() == kNumberTypeFloat32 || out_tensors_.at(0)->data_type() == kNumberTypeFloat) {
    Float16ToFloat32(fp16_output_, reinterpret_cast<float *>(output_addr), out_tensors_.at(0)->ElementsNum());
  }
  FreeTmpBuffer();
  return RET_OK;
}

kernel::LiteKernel *CpuConcatFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                               const InnerContext *ctx, const kernel::KernelKey &desc,
                                               const mindspore::lite::PrimitiveC *primitive) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "Input parameter is nullptr!";
    return nullptr;
  }
  kernel::LiteKernel *kernel = new (std::nothrow) ConcatFp16CPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ConcatCPUKernel fail!";
    free(parameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << parameter->name_
                  << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Concat, CpuConcatFp16KernelCreator)
}  // namespace mindspore::kernel
