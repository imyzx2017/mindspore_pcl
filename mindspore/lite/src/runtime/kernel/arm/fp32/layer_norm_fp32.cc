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
#include "src/runtime/kernel/arm/fp32/layer_norm_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LayerNorm;

namespace mindspore::kernel {
int LayerNormCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int LayerNormCPUKernel::ReSize() {
  auto shape = in_tensors_.front()->shape();
  outer_size_ = 1;
  inner_size_ = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i + param_->normalized_dims_ < shape.size()) {
      outer_size_ *= shape[i];
    } else {
      inner_size_ *= shape[i];
    }
  }
  return RET_OK;
}

int LayerNormCPUKernel::DoLayerNorm(int thread_id) {
  int ret = LayerNorm(outer_size_, inner_size_, src_data_, gamma_data_, beta_data_, param_->elementwise_affine_,
                      param_->epsilon_, dst_data_, thread_id, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoLayerNorm error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int LayerNormRun(void *cdata, int task_id) {
  auto LayerNormData = reinterpret_cast<LayerNormCPUKernel *>(cdata);
  auto ret = LayerNormData->DoLayerNorm(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LayerNormRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int LayerNormCPUKernel::Run() {
  src_data_ = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  if (param_->elementwise_affine_) {
    gamma_data_ = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
    beta_data_ = reinterpret_cast<float *>(in_tensors_.at(2)->MutableData());
  }
  dst_data_ = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  auto ret = ParallelLaunch(this->context_->thread_pool_, LayerNormRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LayerNormRun error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuLayerNormFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                  const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                  const mindspore::lite::PrimitiveC *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed, opParameter is nullptr, type: PrimitiveType_LayerNorm. ";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_LayerNorm);
  auto *kernel = new (std::nothrow) LayerNormCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new LayerNormCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LayerNorm, CpuLayerNormFp32KernelCreator)
}  // namespace mindspore::kernel
