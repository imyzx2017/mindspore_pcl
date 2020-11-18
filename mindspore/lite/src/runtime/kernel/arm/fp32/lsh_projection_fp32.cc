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
#include "src/runtime/kernel/arm/fp32/lsh_projection_fp32.h"

#include "include/errorcode.h"
#include "src/common/string_util.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LshProjection;

namespace mindspore::kernel {
int LshProjectionCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int LshProjectionCPUKernel::ReSize() { return RET_OK; }

int LshProjectionRun(void *cdata, int task_id) {
  auto kernel = reinterpret_cast<LshProjectionCPUKernel *>(cdata);
  return kernel->DoExecute(task_id);
}

int LshProjectionCPUKernel::Run() {
  auto input0_tensor = in_tensors_.at(0);
  auto input1_tensor = in_tensors_.at(1);
  auto out_tensor = out_tensors_.at(0);

  hash_seed_ = reinterpret_cast<float *>(input0_tensor->MutableData());
  feature_ = reinterpret_cast<int32_t *>(input1_tensor->MutableData());
  weight_ = in_tensors_.size() == 2 ? nullptr : reinterpret_cast<float *>(in_tensors_.at(2)->MutableData());
  output_ = reinterpret_cast<int32_t *>(out_tensor->MutableData());

  param_->hash_buff_size_ = sizeof(float) + sizeof(int32_t);
  param_->feature_num_ = input1_tensor->ElementsNum();
  param_->hash_shape_[0] = input0_tensor->DimensionSize(0);
  param_->hash_shape_[1] = input0_tensor->DimensionSize(1);
  param_->thread_stride_ = op_parameter_->thread_num_ > 1 ? UP_DIV(param_->hash_shape_[0], op_parameter_->thread_num_)
                                                          : param_->hash_shape_[0];
  auto ret = MallocKeys();
  if (ret != RET_OK) {
    return ret;
  }
  ret = ParallelLaunch(this->context_->thread_pool_, LshProjectionRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LshProjection kernel parallel launch failed";
  }
  FreeKeys();
  return ret;
}

int LshProjectionCPUKernel::MallocKeys() {
  param_->hash_buffs_ = static_cast<char **>(context_->allocator->Malloc(op_parameter_->thread_num_ * sizeof(char *)));
  if (param_->hash_buffs_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  for (int i = 0; i < op_parameter_->thread_num_; i++) {
    param_->hash_buffs_[i] = static_cast<char *>(context_->allocator->Malloc(param_->hash_buff_size_));
    if (param_->hash_buffs_[i] == nullptr) {
      FreeKeys();
      MS_LOG(ERROR) << "Memory allocation failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void LshProjectionCPUKernel::FreeKeys() {
  if (param_->hash_buffs_ != nullptr) {
    for (int i = 0; i < op_parameter_->thread_num_; i++) {
      context_->allocator->Free(param_->hash_buffs_[i]);
    }
    context_->allocator->Free(param_->hash_buffs_);
  }
}

int LshProjectionCPUKernel::DoExecute(int task_id) {
  int cur_group_num = MSMIN(param_->hash_shape_[0] - task_id * param_->thread_stride_, param_->thread_stride_);
  int start = task_id * param_->thread_stride_;
  int end = start + cur_group_num;
  char *hash_buff = param_->hash_buffs_[task_id];

  switch (param_->lsh_type_) {
    case schema::LshProjectionType_SPARSE:
      LshProjectionSparse(hash_seed_, feature_, weight_, output_, param_, start, end, hash_buff);
      break;
    case schema::LshProjectionType_DENSE:
      LshProjectionDense(hash_seed_, feature_, weight_, output_, param_, start, end, hash_buff);
      break;
    default:
      return RET_ERROR;
  }
  return RET_OK;
}

int LshProjectionCPUKernel::GetSignBit(int32_t *feature_, float *weight_, float seed, LshProjectionParameter *para,
                                       char *hash_buff) {
  double score = 0.0;
  for (int i = 0; i < para->feature_num_; i++) {
    memcpy(hash_buff, &seed, sizeof(float));
    memcpy(hash_buff + sizeof(float), &(feature_[i]), sizeof(int32_t));
    int64_t hash_i = static_cast<int64_t>(lite::StringHash64(hash_buff, para->hash_buff_size_));
    double hash_d = static_cast<double>(hash_i);
    if (weight_ == nullptr) {
      score += hash_d;
    } else {
      score += weight_[i] * hash_d;
    }
  }
  return (score > 0) ? 1 : 0;
}

void LshProjectionCPUKernel::LshProjectionSparse(float *hash_seed_, int32_t *feature_, float *weight_, int32_t *output_,
                                                 LshProjectionParameter *para, int32_t start, int32_t end,
                                                 char *hash_buff) {
  for (int i = start; i < end; i++) {
    int32_t hash_sign = 0;
    for (int j = 0; j < para->hash_shape_[1]; j++) {
      int bit = GetSignBit(feature_, weight_, hash_seed_[i * para->hash_shape_[1] + j], para, hash_buff);
      hash_sign = (hash_sign << 1) | bit;
    }
    output_[i] = hash_sign + i * (1 << para->hash_shape_[1]);
  }
}

void LshProjectionCPUKernel::LshProjectionDense(float *hash_seed_, int32_t *feature_, float *weight_, int32_t *output_,
                                                LshProjectionParameter *para, int32_t start, int32_t end,
                                                char *hash_buff) {
  for (int i = start; i < end; i++) {
    for (int j = 0; j < para->hash_shape_[1]; j++) {
      output_[i * para->hash_shape_[1] + j] =
        GetSignBit(feature_, weight_, hash_seed_[i * para->hash_shape_[1] + j], para, hash_buff);
    }
  }
}

kernel::LiteKernel *CpuLshProjectionFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                      const std::vector<lite::Tensor *> &outputs,
                                                      OpParameter *op_parameter, const lite::InnerContext *ctx,
                                                      const kernel::KernelKey &desc,
                                                      const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) LshProjectionCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new LshProjectionCPUKernel fail!";
    free(op_parameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed! name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LshProjection, CpuLshProjectionFp32KernelCreator)
}  // namespace mindspore::kernel
