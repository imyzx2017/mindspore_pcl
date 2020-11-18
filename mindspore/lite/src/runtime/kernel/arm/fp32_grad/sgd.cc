
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

#include "src/runtime/kernel/arm/fp32_grad/sgd.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "src/runtime/kernel/arm/fp32/nchw2nhwc_fp32.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Sgd;

namespace mindspore::kernel {

int SgdCPUKernel::ReSize() { return RET_OK; }

int SgdCPUKernel::Execute(int task_id) {
  auto weight = reinterpret_cast<float *>(in_tensors_[0]->MutableData());
  auto accumulate = reinterpret_cast<float *>(in_tensors_[3]->MutableData());
  float learning_rate = reinterpret_cast<float *>(in_tensors_[2]->MutableData())[0];
  auto gradient = reinterpret_cast<float *>(in_tensors_[1]->MutableData());
  float moment = reinterpret_cast<float *>(in_tensors_[4]->MutableData())[0];
  size_t elem_num = in_tensors_[0]->ElementsNum();

  if (sgd_param_->use_nesterov_) {
    for (size_t i = 0; i < elem_num; ++i) {
      accumulate[i] = accumulate[i] * moment + gradient[i];
      weight[i] -= (accumulate[i] * moment + gradient[i]) * learning_rate;
    }
  } else {
    for (size_t i = 0; i < elem_num; ++i) {
      accumulate[i] = accumulate[i] * moment + gradient[i] * (1.f - sgd_param_->dampening_);
      weight[i] -= accumulate[i] * learning_rate;
    }
  }
  return RET_OK;
}

int SgdRun(void *cdata, int task_id) {
  auto Sgd_kernel = reinterpret_cast<SgdCPUKernel *>(cdata);
  auto error_code = Sgd_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SGD run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SgdCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, SgdRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SGD function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SgdCPUKernel::Init() {
  // Only for test with uninitialized Data
  size_t elem_num = in_tensors_[0]->ElementsNum();
  auto accumulate = reinterpret_cast<float *>(in_tensors_[3]->MutableData());
  for (size_t i = 0; i < elem_num; i++) accumulate[i] = 0.0;

  if (sgd_param_->dampening_ < 0.0f) {
    MS_LOG(ERROR) << "dampening should be at least 0.0";
    return RET_ERROR;
  }

  if (sgd_param_->use_nesterov_ && sgd_param_->dampening_ > 0.0f) {
    MS_LOG(ERROR) << "If use nesterov, dampening must equal to 0.0";
    return RET_ERROR;
  }

  return RET_OK;
}

kernel::LiteKernel *CpuSgdFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                            const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                            const lite::PrimitiveC *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_Sgd);
  auto *kernel = new (std::nothrow) SgdCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SgdCPUKernel failed!";
    free(opParameter);
    return nullptr;
  }

  auto ret = kernel->Init();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }

  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Sgd, CpuSgdFp32KernelCreator)
}  // namespace mindspore::kernel
