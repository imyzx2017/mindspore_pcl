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

#include "src/runtime/kernel/arm/int8/power_int8.h"
#include <limits>
#include "nnacl/int8/power_int8.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Power;

namespace mindspore::kernel {

int PowerInt8CPUKernel::Init() {
  auto ret = PowerBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }

  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  MS_ASSERT(input);
  MS_ASSERT(output);

  auto in_quant_args = input->GetQuantParams();
  param_->quant_arg_.in_args_.scale_ = in_quant_args.front().scale;
  param_->quant_arg_.in_args_.zp_ = in_quant_args.front().zeroPoint;

  auto out_quant_args = output->GetQuantParams();
  param_->quant_arg_.out_args_.scale_ = out_quant_args.front().scale;
  param_->quant_arg_.out_args_.zp_ = out_quant_args.front().zeroPoint;

  param_->quant_arg_.output_activation_max_ = std::numeric_limits<int8_t>::max();
  param_->quant_arg_.output_activation_min_ = std::numeric_limits<int8_t>::min();

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int PowerInt8CPUKernel::ReSize() { return PowerBaseCPUKernel::ReSize(); }

int PowerInt8CPUKernel::DoPower(int task_id) {
  const int8_t *input_data = reinterpret_cast<const int8_t *>(in_tensors_[0]->MutableData());
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensors_[0]->MutableData());

  auto size = in_tensors_[0]->ElementsNum();
  int stride = UP_DIV(size, op_parameter_->thread_num_);
  int count = MSMIN(stride, size - stride * task_id);
  int8_t *exp_ptr = nullptr;
  param_->broadcast_ = true;
  if (in_tensors_.size() == 2) {
    auto exp_tensor = in_tensors_.at(1);
    auto exp_quant_args = exp_tensor->GetQuantParams();
    param_->quant_arg_.exp_args_.scale_ = exp_quant_args.front().scale;
    param_->quant_arg_.exp_args_.zp_ = exp_quant_args.front().zeroPoint;
    exp_ptr = reinterpret_cast<int8_t *>(exp_tensor->MutableData());
    param_->broadcast_ = false;
    if (in_tensors_[0]->Size() != in_tensors_[1]->Size()) {
      MS_LOG(ERROR) << "Power input size  " << in_tensors_[0]->Size() << " is not equal to exponent size  "
                    << in_tensors_[1]->Size();
      return RET_ERROR;
    }
  }
  if (!param_->broadcast_) {
    exp_ptr = exp_ptr + stride * task_id;
  }
  auto ret = PowerInt8(input_data + stride * task_id, exp_ptr, output_data + stride * task_id, count, param_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PowerInt8 error ,task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int PowerInt8Run(void *cdata, int task_id) {
  auto power_kernel = reinterpret_cast<PowerInt8CPUKernel *>(cdata);
  auto ret = power_kernel->DoPower(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoPower error, task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int PowerInt8CPUKernel::Run() {
  auto ret = ParallelLaunch(this->context_->thread_pool_, PowerInt8Run, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PowerInt8Run error, error_code[" << ret << "]";
  }
  return ret;
}
kernel::LiteKernel *CpuPowerInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                              const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                              const mindspore::lite::PrimitiveC *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Power);
  auto *kernel = new (std::nothrow) PowerInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PowerInt8CPUKernel fail!";
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
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Power, CpuPowerInt8KernelCreator)

}  // namespace mindspore::kernel
