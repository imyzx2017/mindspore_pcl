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

#include "src/runtime/kernel/arm/fp32/exp_fp32.h"
#include <math.h>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Exp;

namespace mindspore::kernel {
int ExpCPUKernel::Init() {
  exp_parameter_ = reinterpret_cast<ExpParameter *>(op_parameter_);
  exp_parameter_->thread_num_ = thread_count_;

  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int ExpCPUKernel::ReSize() {
  exp_parameter_->thread_num_ = thread_count_;
  float log_ = (exp_parameter_->base_ == -1) ? 1 : logf(exp_parameter_->base_);
  exp_parameter_->in_scale_ = exp_parameter_->scale_ * log_;
  if (exp_parameter_->shift_ == 0) {
    exp_parameter_->out_scale_ = 1;
  } else {
    if (log_ == 1) {
      exp_parameter_->out_scale_ = expf(exp_parameter_->shift_);
    } else {
      exp_parameter_->out_scale_ = powf(exp_parameter_->base_, exp_parameter_->shift_);
    }
  }
  return RET_OK;
}

int ExpCPUKernel::DoExcute(int task_id) {
  Exp(input_addr_, output_addr_, exp_parameter_, task_id);
  return RET_OK;
}

int ExpRun(void *cdata, int task_id) {
  auto ExpData = reinterpret_cast<ExpCPUKernel *>(cdata);
  auto ret = ExpData->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ExpRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ExpCPUKernel::Run() {
  input_addr_ = reinterpret_cast<float *>(in_tensors_.front()->MutableData());
  output_addr_ = reinterpret_cast<float *>(out_tensors_.front()->MutableData());
  exp_parameter_->element_num_ = in_tensors_.front()->ElementsNum();

  auto ret = ParallelLaunch(this->context_->thread_pool_, ExpRun, this, exp_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Exp error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuExpFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                            const lite::InnerContext *ctx, const KernelKey &desc,
                                            const mindspore::lite::PrimitiveC *primitive) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is nullptr";
    return nullptr;
  }
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "ctx is nullptr";
    free(parameter);
    return nullptr;
  }
  MS_ASSERT(desc.type == PrimitiveType_Exp);
  auto *kernel = new (std::nothrow) ExpCPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create Kernel failed, name: " << parameter->name_;
    free(parameter);
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init Kernel failed, name: " << parameter->name_
                  << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Exp, CpuExpFp32KernelCreator)
}  // namespace mindspore::kernel
