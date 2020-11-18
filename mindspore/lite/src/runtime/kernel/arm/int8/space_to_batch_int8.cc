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
#include "src/runtime/kernel/arm/int8/space_to_batch_int8.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32/space_to_batch.h"
#include "nnacl/int8/space_to_batch_int8.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SpaceToBatch;
using mindspore::schema::PrimitiveType_SpaceToBatchND;

namespace mindspore::kernel {
int SpaceToBatchInt8CPUKernel::Run() {
  auto input_tensor = in_tensors_.at(0);
  auto output_tensor = out_tensors_.at(0);
  auto input_ptr = reinterpret_cast<const int8_t *>(input_tensor->MutableData());
  auto output_ptr = reinterpret_cast<int8_t *>(output_tensor->MutableData());
  SpaceToBatchParameter *param = reinterpret_cast<SpaceToBatchParameter *>(this->op_parameter_);
  if (output_tensor->GetQuantParams().empty()) {
    MS_LOG(ERROR) << "SpaceToBatchInt8 need quantization parameters which is not found.";
    return RET_ERROR;
  }
  auto quant_arg = output_tensor->GetQuantParams().front();

  if (param->need_paddings_) {
    DoSpaceToBatchPaddingNHWCInt8(input_ptr, output_ptr, param, quant_arg.zeroPoint);
  } else {
    DoSpaceToBatchNHWCInt8(input_ptr, output_ptr, param->block_sizes_, param->input_shape_, param->output_shape_);
  }
  return RET_OK;
}

kernel::LiteKernel *CpuSpaceToBatchInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                     const std::vector<lite::Tensor *> &outputs, OpParameter *param,
                                                     const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                     const mindspore::lite::PrimitiveC *primitive) {
  if (param == nullptr) {
    MS_LOG(ERROR) << "Input param is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) SpaceToBatchInt8CPUKernel(param, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SpaceToBatchInt8CPUKernel fail!";
    free(param);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << param->name_
                  << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(param->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_SpaceToBatch, CpuSpaceToBatchInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_SpaceToBatchND, CpuSpaceToBatchInt8KernelCreator)
}  // namespace mindspore::kernel
