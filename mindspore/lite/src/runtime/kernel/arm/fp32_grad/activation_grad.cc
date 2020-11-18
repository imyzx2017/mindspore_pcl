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

#include "src/runtime/kernel/arm/fp32_grad/activation_grad.h"
#include "nnacl/fp32_grad/activation_grad.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_HSWISH;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::PrimitiveType_ActivationGrad;

namespace mindspore::kernel {
int ActivationGradCPUKernel::Init() {
  if (2 != in_tensors_.size()) {
    MS_LOG(ERROR) << "ActivationGrad should have 2 input tensors";
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationGradCPUKernel::ReSize() { return RET_OK; }

int ActivationGradCPUKernel::DoActivation(int task_id) {
  auto yt_addr = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto input_addr = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  int length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, 1);
  int count = MSMIN(stride, length - stride * task_id);

  auto error_code = RET_OK;

  if (param_act_grad_->type_ == schema::ActivationType_RELU) {
    error_code =
      ReluGrad(yt_addr + stride * task_id, input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (param_act_grad_->type_ == schema::ActivationType_RELU6) {
    error_code =
      Relu6Grad(yt_addr + stride * task_id, input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (param_act_grad_->type_ == schema::ActivationType_LEAKY_RELU) {
    error_code = LReluGrad(yt_addr + stride * task_id, input_addr + stride * task_id, count,
                           output_addr + stride * task_id, param_act_grad_->alpha_);
  } else if (param_act_grad_->type_ == schema::ActivationType_SIGMOID) {
    // Sigmoid gets the input tensors in reverse order!
    error_code =
      SigmoidGrad(input_addr + stride * task_id, yt_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (param_act_grad_->type_ == schema::ActivationType_TANH) {
    error_code =
      TanhGrad(yt_addr + stride * task_id, input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (param_act_grad_->type_ == schema::ActivationType_HSWISH) {
    error_code =
      HSwishGrad(yt_addr + stride * task_id, input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (param_act_grad_->type_ == schema::ActivationType_HSIGMOID) {
    error_code =
      HSigmoidGrad(yt_addr + stride * task_id, input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else {
    MS_LOG(ERROR) << "Activation type error";
    return RET_ERROR;
  }
  if (error_code != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationGradRun(void *cdata, int task_id) {
  auto activationGrad_kernel = reinterpret_cast<ActivationGradCPUKernel *>(cdata);
  auto error_code = activationGrad_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ActivationGradRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationGradCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, ActivationGradRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Activation Grad function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuActivationGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                       const std::vector<lite::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::InnerContext *ctx,
                                                       const kernel::KernelKey &desc,
                                                       const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_ActivationGrad);
  auto *kernel = new (std::nothrow) ActivationGradCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ActivationGradCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ActivationGrad, CpuActivationGradFp32KernelCreator)
}  // namespace mindspore::kernel
