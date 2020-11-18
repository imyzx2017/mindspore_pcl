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

#include "src/runtime/kernel/arm/fp16/pad_fp16.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Pad;

namespace mindspore::kernel {
int PadFp16CPUKernel::RunImpl(int task_id) {
  PadFp16(input_, output_, in_, out_, pad_param_->paddings_, task_id, context_->thread_num_);
  return RET_OK;
}

int PadFp16CPUKernel::RunMirrorPadImpl(int task_id) {
  int unit = UP_DIV(out_tensors_.at(0)->ElementsNum(), context_->thread_num_);
  int begin = unit * task_id;
  int end = MSMIN(begin + unit, out_tensors_.at(0)->ElementsNum());
  MirrorPadFp16(input_, output_, in_, pad_param_, begin, end);
  return RET_OK;
}

int PadFp16CPUKernel::Run() {
  auto input_tensor = in_tensors_.at(0);
  auto output_tensor = out_tensors_.at(0);
  is_input_fp32_ = input_tensor->data_type() == kNumberTypeFloat32;
  is_output_fp32_ = output_tensor->data_type() == kNumberTypeFloat32;

  input_ = ConvertInputFp32toFp16(input_tensor, context_);
  output_ = MallocOutputFp16(output_tensor, context_);
  if (input_ == nullptr || output_ == nullptr) {
    FreeInputAndOutput();
    MS_LOG(ERROR) << "input or output is nullptr";
    return RET_ERROR;
  }
  int ret = 0;
  if (pad_param_->pad_mode_ == static_cast<int>(schema::PaddingMode_CONSTANT)) {
    if (pad_param_->constant_value_ - 0.0f < 1e-5) {
      memset(output_, 0, output_tensor->ElementsNum() * sizeof(float16_t));
    } else {
      for (int i = 0; i < output_tensor->ElementsNum(); ++i) {
        output_[i] = pad_param_->constant_value_;
      }
    }
    ret = ParallelLaunch(this->context_->thread_pool_, PadImpl, this, op_parameter_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "BatchnormRun error error_code[" << ret << "]";
    }
  } else {
    HandleMirrorPad();
    ret = ParallelLaunch(this->context_->thread_pool_, MirrorPadImpl, this, context_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Pad Reflect or Symmetric mode run error, error_code[" << ret << "]";
    }
  }
  if (is_output_fp32_) {
    Float16ToFloat32(output_, reinterpret_cast<float *>(output_tensor->MutableData()), output_tensor->ElementsNum());
  }
  FreeInputAndOutput();
  return ret;
}

void PadFp16CPUKernel::FreeInputAndOutput() {
  if (is_input_fp32_) {
    context_->allocator->Free(input_);
    input_ = nullptr;
  }
  if (is_output_fp32_) {
    context_->allocator->Free(output_);
    output_ = nullptr;
  }
}

kernel::LiteKernel *CpuPadFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                            const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                            const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) PadFp16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PadFp16CPUKernel fail!";
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
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Pad, CpuPadFp16KernelCreator)
}  // namespace mindspore::kernel
