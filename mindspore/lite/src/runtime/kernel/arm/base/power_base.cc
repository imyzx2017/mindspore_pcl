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
#include "src/runtime/kernel/arm/base/power_base.h"
#include <vector>
#include "src/runtime/kernel/arm/fp32/power_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Power;

namespace mindspore::kernel {
int PowerBaseCPUKernel::Init() { return RET_OK; }

int PowerBaseCPUKernel::ReSize() { return RET_OK; }

kernel::LiteKernel *CpuPowerFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                              const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                              const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Power);
  PowerCPUKernel *kernel = new (std::nothrow) PowerCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PowerCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Power, CpuPowerFp32KernelCreator)
}  // namespace mindspore::kernel
