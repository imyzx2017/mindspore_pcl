
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

#include "src/runtime/kernel/arm/fp32_grad/apply_momentum.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "src/runtime/kernel/arm/fp32/nchw2nhwc_fp32.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ApplyMomentum;

namespace mindspore::kernel {

int ApplyMomentumCPUKernel::ReSize() { return RET_OK; }

int ApplyMomentumCPUKernel::Execute(int task_id) {
  auto weight = reinterpret_cast<float *>(in_tensors_[0]->MutableData());
  auto accumulate = reinterpret_cast<float *>(in_tensors_[1]->MutableData());
  float learning_rate = reinterpret_cast<float *>(in_tensors_[2]->MutableData())[0];
  auto gradient = reinterpret_cast<float *>(in_tensors_[3]->MutableData());
  float moment = reinterpret_cast<float *>(in_tensors_[4]->MutableData())[0];
  size_t elem_num = in_tensors_[0]->ElementsNum();

  if (apply_momentum_param_->use_nesterov_) {
    for (size_t i = 0; i < elem_num; ++i) {
      accumulate[i] = accumulate[i] * moment + gradient[i];
      weight[i] -= (accumulate[i] * moment + gradient[i]) * learning_rate;
    }
  } else {
    for (size_t i = 0; i < elem_num; ++i) {
      accumulate[i] = accumulate[i] * moment + gradient[i];
      weight[i] -= accumulate[i] * learning_rate;
    }
  }
  return RET_OK;
}

int ApplyMomentumRun(void *cdata, int task_id) {
  auto applyMomentum_kernel = reinterpret_cast<ApplyMomentumCPUKernel *>(cdata);
  auto error_code = applyMomentum_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "apply Momentum run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ApplyMomentumCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, ApplyMomentumRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Apply Momentum function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ApplyMomentumCPUKernel::Init() { return RET_OK; }

kernel::LiteKernel *CpuApplyMomentumFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                      const std::vector<lite::Tensor *> &outputs,
                                                      OpParameter *opParameter, const lite::InnerContext *ctx,
                                                      const kernel::KernelKey &desc,
                                                      const lite::PrimitiveC *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_ApplyMomentum);
  auto *kernel = new (std::nothrow) ApplyMomentumCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ApplyMomentumCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }

  auto ret = kernel->Init();
  if (0 != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }

  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ApplyMomentum, CpuApplyMomentumFp32KernelCreator)
}  // namespace mindspore::kernel
