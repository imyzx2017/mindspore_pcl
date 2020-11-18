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

#include "src/runtime/kernel/arm/int8/crop_int8.h"
#include <limits>
#include "nnacl/int8/crop_int8.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Crop;

namespace mindspore::kernel {

int CropInt8CPUKernel::Init() {
  auto ret = CropBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto in_quant_args = input_tensor->GetQuantParams();
  crop_para_->quant_arg.in_args_.scale_ = in_quant_args.front().scale;
  crop_para_->quant_arg.in_args_.zp_ = in_quant_args.front().zeroPoint;

  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->GetQuantParams();
  crop_para_->quant_arg.out_args_.scale_ = out_quant_args.front().scale;
  crop_para_->quant_arg.out_args_.zp_ = out_quant_args.front().zeroPoint;

  crop_para_->quant_arg.output_activation_max_ = std::numeric_limits<int8_t>::max();
  crop_para_->quant_arg.output_activation_min_ = std::numeric_limits<int8_t>::min();
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

CropInt8CPUKernel::~CropInt8CPUKernel() {
  if (crop_para_->in_shape_ != nullptr) {
    free(const_cast<int *>(crop_para_->in_shape_));
    crop_para_->in_shape_ = nullptr;
  }

  if (crop_para_->out_shape_ != nullptr) {
    free(const_cast<int *>(crop_para_->out_shape_));
    crop_para_->out_shape_ = nullptr;
  }
}

int CropInt8CPUKernel::ReSize() { return CropBaseCPUKernel::ReSize(); }

int CropInt8CPUKernel::Run() {
  auto ret = ParallelLaunch(this->context_->thread_pool_, CropInt8Run, this, thread_count_);
  return ret;
}

int CropInt8Run(void *cdata, int task_id) {
  auto crop = reinterpret_cast<CropInt8CPUKernel *>(cdata);
  crop->DoExecute(task_id);
  return RET_OK;
}

int CropInt8CPUKernel::DoExecute(int task_id) {
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto out_tensor = out_tensors_.at(kOutputIndex);
  int8_t *input_data = reinterpret_cast<int8_t *>(input_tensor->MutableData());
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensor->MutableData());
  Int8Crop(input_data, output_data, task_id, crop_para_);
  return RET_OK;
}

kernel::LiteKernel *CpuCropInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                             const InnerContext *ctx, const kernel::KernelKey &desc,
                                             const mindspore::lite::PrimitiveC *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Crop);
  auto *kernel = new (std::nothrow) CropInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new CropCPUKernel fail!";
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
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Crop, CpuCropInt8KernelCreator)

}  // namespace mindspore::kernel
