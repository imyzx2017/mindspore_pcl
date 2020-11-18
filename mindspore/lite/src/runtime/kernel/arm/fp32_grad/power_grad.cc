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

#include "src/runtime/kernel/arm/fp32_grad/power_grad.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/arithmetic.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PowerGrad;

namespace mindspore::kernel {
int PowerGradCPUKernel::Init() {
  if (2 != in_tensors_.size()) {
    MS_LOG(ERROR) << "Power Grad Filter should have 2 inputs";
    return RET_ERROR;
  }
  if (1 != out_tensors_.size()) {
    MS_LOG(ERROR) << "Power Grad Filter should have one output";
    return RET_ERROR;
  }
  return RET_OK;
}

int PowerGradCPUKernel::ReSize() { return RET_OK; }

int PowerGradCPUKernel::Execute(int task_id) {
  auto dy_addr = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto x_addr = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  auto dx_addr = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  auto size = in_tensors_.at(0)->ElementsNum();

  float exp = power_ - 1;
  Power(x_addr, &exp, dx_addr, size, scale_, shift_, true);
  ElementMul(dx_addr, dy_addr, dx_addr, size);
  float scale = scale_ * power_;
  for (int i = 0; i < size; i++) {
    dx_addr[i] *= scale;
  }

  return RET_OK;
}

int PowerGradRun(void *cdata, int task_id) {
  auto power_kernel = reinterpret_cast<PowerGradCPUKernel *>(cdata);
  auto error_code = power_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "power grad error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PowerGradCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, PowerGradRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "power grad function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuPowerGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                  const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                  const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_PowerGrad);
  auto *kernel = new (std::nothrow) PowerGradCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PowerGradCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_PowerGrad, CpuPowerGradFp32KernelCreator)
}  // namespace mindspore::kernel
