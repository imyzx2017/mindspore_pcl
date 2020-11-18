
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

#include "src/runtime/kernel/arm/fp32_grad/assign.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "src/runtime/kernel/arm/fp32/nchw2nhwc_fp32.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Assign;

namespace mindspore::kernel {

int AssignCPUKernel::ReSize() { return RET_OK; }

int AssignCPUKernel::Execute(int task_id) {
  auto x = reinterpret_cast<float *>(in_tensors_[0]->MutableData());
  auto y = reinterpret_cast<float *>(in_tensors_[1]->MutableData());
  size_t size = in_tensors_[0]->Size();

  memcpy(x, y, size);
  return RET_OK;
}

int AssignRun(void *cdata, int task_id) {
  auto Assign_kernel = reinterpret_cast<AssignCPUKernel *>(cdata);
  auto error_code = Assign_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "assign run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int AssignCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, AssignRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Assign function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int AssignCPUKernel::Init() { return RET_OK; }

kernel::LiteKernel *CpuAssignFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                               const lite::PrimitiveC *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_Assign);
  auto *kernel = new (std::nothrow) AssignCPUKernel(opParameter, inputs, outputs, ctx, primitive);
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Assign, CpuAssignFp32KernelCreator)
}  // namespace mindspore::kernel
