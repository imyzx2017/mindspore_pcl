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

#include "src/runtime/kernel/arm/fp32/squeeze_fp32.h"
#include <vector>
#include "nnacl/squeeze.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Squeeze;

namespace mindspore::kernel {
int SqueezeCPUKernel::Init() { return RET_OK; }

int SqueezeCPUKernel::ReSize() { return RET_OK; }

int SqueezeCPUKernel::Run() {
  mindspore::lite::STATUS ret = RET_ERROR;
  size_t data_size = in_tensors_.front()->Size();
  if (in_tensors_.front()->data_type() == kNumberTypeInt32) {
    auto input_ptr = reinterpret_cast<int32_t *>(in_tensors_.front()->MutableData());
    auto output_ptr = reinterpret_cast<int32_t *>(out_tensors_.front()->MutableData());
    ret = DoSqueezeInt32(input_ptr, output_ptr, data_size);
  } else {
    auto input_ptr = reinterpret_cast<float *>(in_tensors_.front()->MutableData());
    auto output_ptr = reinterpret_cast<float *>(out_tensors_.front()->MutableData());
    ret = DoSqueeze(input_ptr, output_ptr, data_size);
  }

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Do squeeze fail!ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuSqueezeFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                                const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_Squeeze);
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "desc type is not Squeeze";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) SqueezeCPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New kernel fails.";
    free(parameter);
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << parameter->name_
                  << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(parameter->type_));
    delete kernel;
    return nullptr;
  }

  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Squeeze, CpuSqueezeFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Squeeze, CpuSqueezeFp32KernelCreator)
}  // namespace mindspore::kernel
