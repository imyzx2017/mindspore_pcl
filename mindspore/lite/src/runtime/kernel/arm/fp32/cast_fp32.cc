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
#include "src/runtime/kernel/arm/fp32/cast_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/tensor.h"
#include "nnacl/fp32/cast.h"
#include "nnacl/op_base.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Cast;

namespace mindspore::kernel {
namespace {
int CastRun(void *cdata, int task_id) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "input cdata is nullptr!";
    return RET_ERROR;
  }

  return reinterpret_cast<CastCPUKernel *>(cdata)->DoCast(task_id);
}
}  // namespace

int CastCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int CastCPUKernel::ReSize() {
  data_num_ = in_tensors_[0]->ElementsNum();
  if (data_num_ == 0) {
    return RET_OK;
  }
  op_parameter_->thread_num_ = MSMIN(op_parameter_->thread_num_, static_cast<int>(data_num_));
  stride_ = UP_DIV(data_num_, op_parameter_->thread_num_);
  return RET_OK;
}

int CastCPUKernel::DoCast(int thread_id) {
  auto input = in_tensors_.at(0);
  int data_num = MSMIN(stride_, data_num_ - thread_id * stride_);
  if (data_num <= 0) {
    return RET_OK;
  }

  auto offset = thread_id * stride_;
  auto output = out_tensors_.at(0);
  auto output_data = output->data_c();
  MS_ASSERT(output_data != nullptr);
  auto input_data_type = input->data_type();
  auto output_data_type = output->data_type();
  if (input_data_type == output_data_type) {
    auto datalen = lite::DataTypeSize(input_data_type);
    memcpy(reinterpret_cast<char *>(output_data) + offset * datalen,
           reinterpret_cast<char *>(input->data_c()) + offset * datalen, data_num * datalen);
    return RET_OK;
  }
  if (output_data_type != kNumberTypeFloat32) {
    if (input_data_type == kNumberTypeFloat32 && output_data_type == kNumberTypeInt64) {
      Float32ToInt64(reinterpret_cast<float *>(input->data_c()) + offset,
                     reinterpret_cast<int64_t *>(output_data) + offset, data_num);
    } else if (input_data_type == kNumberTypeFloat32 && output_data_type == kNumberTypeInt32) {
      Float32ToInt32(reinterpret_cast<float *>(input->data_c()) + offset,
                     reinterpret_cast<int32_t *>(output_data) + offset, data_num);
    } else if (input_data_type == kNumberTypeFloat32 && output_data_type == kNumberTypeFloat16) {
      Float32ToFp16(reinterpret_cast<float *>(input->data_c()) + offset,
                    reinterpret_cast<uint16_t *>(output_data) + offset, data_num);
    } else if (input_data_type == kNumberTypeInt32 && output_data_type == kNumberTypeInt64) {
      Int32ToInt64(reinterpret_cast<int32_t *>(input->data_c()) + offset,
                   reinterpret_cast<int64_t *>(output_data) + offset, data_num);
    } else {
      MS_LOG(ERROR) << "Unsupported datatype from " << input_data_type << " to " << output_data_type;
      return RET_ERROR;
    }
  } else {
    switch (input_data_type) {
      case kNumberTypeBool:
        BoolToFloat32(reinterpret_cast<bool *>(input->MutableData()) + offset,
                      reinterpret_cast<float *>(output_data) + offset, data_num);
        break;
      case kNumberTypeUInt8:
        Uint8ToFloat32(reinterpret_cast<uint8_t *>(input->MutableData()) + offset,
                       reinterpret_cast<float *>(output_data) + offset, data_num);
        break;
      case kNumberTypeInt32:
        Int32ToFloat32(reinterpret_cast<int32_t *>(input->MutableData()) + offset,
                       reinterpret_cast<float *>(output_data) + offset, data_num);
        break;
      case kNumberTypeFloat16:
        Fp16ToFloat32(reinterpret_cast<uint16_t *>(input->MutableData()) + offset,
                      reinterpret_cast<float *>(output_data) + offset, data_num);
        break;
      default:
        MS_LOG(ERROR) << "Unsupported input data type " << input_data_type;
        return RET_ERROR;
    }
  }
  return RET_OK;
}

int CastCPUKernel::Run() {
  if (data_num_ == 0) {
    return RET_OK;
  }
  return ParallelLaunch(this->context_->thread_pool_, CastRun, this, op_parameter_->thread_num_);
}

kernel::LiteKernel *CpuCastFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                             const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                             const mindspore::lite::PrimitiveC *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "Input context is nullptr!";
    free(opParameter);
    return nullptr;
  }
  if (ctx->thread_num_ == 0) {
    MS_LOG(ERROR) << "context thread num is 0!";
    free(opParameter);
    return nullptr;
  }
  auto *kernel = new (std::nothrow) CastCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new CastCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Cast, CpuCastFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeUInt8, PrimitiveType_Cast, CpuCastFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Cast, CpuCastFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Cast, CpuCastFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Cast, CpuCastFp32KernelCreator)
#ifndef ENABLE_ARM
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Cast, CpuCastFp32KernelCreator)
#endif
}  // namespace mindspore::kernel
