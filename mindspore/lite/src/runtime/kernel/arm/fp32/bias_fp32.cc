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

#include "src/runtime/kernel/arm/fp32/bias_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/int8/bias_add_int8.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BiasAdd;

namespace mindspore::kernel {
int BiasCPUKernel::ReSize() {
  auto dims = in_tensors_[0]->shape();
  MS_ASSERT(dims.size() <= 5);
  bias_param_->ndim_ = dims.size();
  for (size_t i = 0; i < bias_param_->ndim_; i++) {
    bias_param_->in_shape0_[i] = dims[i];
    bias_param_->in_shape1_[i] = 1;
    bias_param_->out_shape_[i] = dims[i];
  }
  bias_param_->in_shape1_[bias_param_->ndim_ - 1] = dims[bias_param_->ndim_ - 1];
  return RET_OK;
}

int BiasCPUKernel::Run() {
  auto in = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto bias = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  auto out = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  size_t data_size = in_tensors_.at(0)->ElementsNum();
  MS_ASSERT(context_->allocator != nullptr);
  float *tile_in = reinterpret_cast<float *>(context_->allocator->Malloc(data_size * sizeof(float)));
  float *tile_bias = reinterpret_cast<float *>(context_->allocator->Malloc(data_size * sizeof(float)));
  if (tile_in == nullptr || tile_bias == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    context_->allocator->Free(tile_in);
    context_->allocator->Free(tile_bias);
    return RET_ERROR;
  }
  BroadcastAdd(in, bias, tile_in, tile_bias, out, data_size, bias_param_);
  context_->allocator->Free(tile_in);
  context_->allocator->Free(tile_bias);
  return RET_OK;
}

int BiasCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

kernel::LiteKernel *CpuBiasFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                             const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                             const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(parameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_BiasAdd);
  auto kernel = new (std::nothrow) BiasCPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed, name: " << parameter->name_;
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BiasAdd, CpuBiasFp32KernelCreator)
}  // namespace mindspore::kernel
