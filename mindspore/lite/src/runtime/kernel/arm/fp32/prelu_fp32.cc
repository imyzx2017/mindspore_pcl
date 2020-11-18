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
#include "src/runtime/kernel/arm/fp32/prelu_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PReLU;

namespace mindspore::kernel {
namespace {
int PReluRun(void *cdata, int task_id) {
  auto PRelu = reinterpret_cast<PReluCPUKernel *>(cdata);
  auto ret = PRelu->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PReluRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace

int PReluCPUKernel::Init() { return RET_OK; }

int PReluCPUKernel::DoExcute(int task_id) {
  if (prelu_param_->channelShared) {
    PReluShareChannel(input_data_, output_data_, prelu_param_, task_id);
  } else {
    PRelu(input_data_, output_data_, prelu_param_, task_id);
  }
  return RET_OK;
}

int PReluCPUKernel::ProcessInput() {
  // input tensor
  auto input_tensor = in_tensors_[0];
  auto in_shape = input_tensor->shape();
  auto n_dim = in_shape.size();
  auto channel_num = in_shape.at(n_dim - 1);
  int input_plane = 1;
  for (size_t i = 0; i < n_dim - 1; ++i) {
    input_plane *= in_shape[i];
  }
  int tile_block = UP_DIV(input_plane, TILE_NUM);
  prelu_param_->input_num_ = input_tensor->ElementsNum();
  prelu_param_->tile_block_ = tile_block;
  prelu_param_->channel_num_ = channel_num;
  input_data_ =
    reinterpret_cast<float *>(context_->allocator->Malloc(tile_block * TILE_NUM * channel_num * sizeof(float)));
  if (input_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc input_data_ failed.";
    return RET_ERROR;
  }
  memcpy(input_data_, ori_input_, tile_block * TILE_NUM * channel_num * sizeof(float));
  return RET_OK;
}

int PReluCPUKernel::ProcessShareChannelInput() {
  // input tensor
  auto input_tensor = in_tensors_[0];
  prelu_param_->input_num_ = input_tensor->ElementsNum();
#ifdef ENABLE_ARM64
  prelu_param_->tile_block_ = UP_DIV(prelu_param_->input_num_, 64);
  input_data_ = reinterpret_cast<float *>(context_->allocator->Malloc(prelu_param_->tile_block_ * 64 * sizeof(float)));
  if (input_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc input_data_ failed.";
    return RET_ERROR;
  }
  memcpy(input_data_, ori_input_, prelu_param_->tile_block_ * 64 * sizeof(float));
#elif ENABLE_ARM32
  prelu_param_->tile_block_ = UP_DIV(prelu_param_->input_num_, 32);
  input_data_ = reinterpret_cast<float *>(context_->allocator->Malloc(prelu_param_->tile_block_ * 32 * sizeof(float)));
  if (input_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc input_data_ failed.";
    return RET_ERROR;
  }
  memcpy(input_data_, ori_input_, prelu_param_->tile_block_ * 32 * sizeof(float));
#else
  prelu_param_->tile_block_ = UP_DIV(prelu_param_->input_num_, 32);
  input_data_ = reinterpret_cast<float *>(context_->allocator->Malloc(prelu_param_->tile_block_ * 32 * sizeof(float)));
  if (input_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc input_data_ failed.";
    return RET_ERROR;
  }
  memcpy(input_data_, ori_input_, prelu_param_->tile_block_ * 32 * sizeof(float));
#endif
  return RET_OK;
}

int PReluCPUKernel::Run() {
  MS_ASSERT(in_tensors_.size() >= 2);
  auto input_tensor = in_tensors_[0];
  ori_input_ = reinterpret_cast<float *>(input_tensor->MutableData());
  output_data_ = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->MutableData());

  if (prelu_param_->channelShared) {
    auto ret = ProcessShareChannelInput();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ProcessShareChannel failed.";
      return ret;
    }
  } else {
    auto ret = ProcessInput();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Process failed.";
      return ret;
    }
  }

  // negative slope tensor
  auto negative_slope_tensor = in_tensors_.at(1);
  prelu_param_->slope_ = reinterpret_cast<float *>(negative_slope_tensor->MutableData());

  auto ret = ParallelLaunch(this->context_->thread_pool_, PReluRun, this, prelu_param_->op_parameter_.thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PRelu Run error: error_code[" << ret << "]";
    context_->allocator->Free(input_data_);
    return RET_ERROR;
  }

  memcpy(output_data_, input_data_, prelu_param_->input_num_ * sizeof(float));
  context_->allocator->Free(input_data_);
  return RET_OK;
}

kernel::LiteKernel *CpuPReluFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *param,
                                              const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                              const mindspore::lite::PrimitiveC *primitive) {
  if (param == nullptr) {
    MS_LOG(ERROR) << "input param is nullptr!";
    return nullptr;
  }

  auto *kernel = new (std::nothrow) PReluCPUKernel(param, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PReluCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_PReLU, CpuPReluFp32KernelCreator)
}  // namespace mindspore::kernel
