
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

#include "src/runtime/kernel/arm/fp32_grad/adam.h"
#include <cmath>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "src/runtime/kernel/arm/fp32/nchw2nhwc_fp32.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Adam;

namespace mindspore::kernel {

int AdamCPUKernel::ReSize() { return RET_OK; }

int AdamCPUKernel::Execute(int task_id) {
  auto weight = reinterpret_cast<float *>(in_tensors_[0]->MutableData());
  auto m = reinterpret_cast<float *>(in_tensors_[1]->MutableData());
  auto v = reinterpret_cast<float *>(in_tensors_[2]->MutableData());
  auto beta1_power = reinterpret_cast<float *>(in_tensors_[3]->MutableData())[0];
  auto beta2_power = reinterpret_cast<float *>(in_tensors_[4]->MutableData())[0];
  auto learning_rate = reinterpret_cast<float *>(in_tensors_[5]->MutableData())[0];
  auto beta1 = reinterpret_cast<float *>(in_tensors_[6]->MutableData())[0];
  auto beta2 = reinterpret_cast<float *>(in_tensors_[7]->MutableData())[0];
  auto eps = reinterpret_cast<float *>(in_tensors_[8]->MutableData())[0];
  auto gradient = reinterpret_cast<float *>(in_tensors_[9]->MutableData());
  size_t elem_num = in_tensors_[0]->ElementsNum();
  auto update_lr = learning_rate * std::sqrt(1 - beta2_power) / (1 - beta1_power);

  if (adam_param_->use_nesterov_) {  // Nadam
    for (size_t i = 0; i < elem_num; ++i) {
      m[i] += (gradient[i] - m[i]) * (1 - beta1);
      v[i] += (gradient[i] * gradient[i] - v[i]) * (1 - beta2);
      weight[i] -= update_lr * (m[i] * beta1 + (1 - beta1) * gradient[i]) / (std::sqrt(v[i]) + eps);
    }
  } else {
    for (size_t i = 0; i < elem_num; ++i) {
      m[i] += (gradient[i] - m[i]) * (1 - beta1);
      v[i] += (gradient[i] * gradient[i] - v[i]) * (1 - beta2);
      weight[i] -= update_lr * m[i] / (std::sqrt(v[i]) + eps);
    }
  }
  return RET_OK;
}

int AdamRun(void *cdata, int task_id) {
  auto Adam_kernel = reinterpret_cast<AdamCPUKernel *>(cdata);
  auto error_code = Adam_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Adam run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int AdamCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, AdamRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Adam function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int AdamCPUKernel::Init() { return RET_OK; }

kernel::LiteKernel *CpuAdamFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                             const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                             const lite::PrimitiveC *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_Adam);
  auto *kernel = new (std::nothrow) AdamCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  MS_ASSERT(kernel != nullptr);

  auto ret = kernel->Init();
  if (0 != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }

  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Adam, CpuAdamFp32KernelCreator)
}  // namespace mindspore::kernel
