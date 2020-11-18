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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_MATMUL_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_MATMUL_INT8_H_

#include <vector>
#include "src/runtime/kernel/arm/base/matmul_base.h"
#include "include/context.h"
#include "nnacl/quantization/quantize.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class MatmulInt8CPUKernel : public MatmulBaseCPUKernel {
 public:
  MatmulInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx,
                      const mindspore::lite::PrimitiveC *primitive)
      : MatmulBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~MatmulInt8CPUKernel() override;
  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);

 private:
  void FreeTmpBuffer() {
    if (a_r4x16_ptr_ != nullptr) {
      ctx_->allocator->Free(a_r4x16_ptr_);
      a_r4x16_ptr_ = nullptr;
    }
    if (input_sums_ != nullptr) {
      ctx_->allocator->Free(input_sums_);
      input_sums_ = nullptr;
    }
    if (b_c16x4_batch_ != nullptr) {
      ctx_->allocator->Free(b_c16x4_batch_);
      b_c16x4_batch_ = nullptr;
    }
    if (weight_bias_sums_batch_ != nullptr) {
      ctx_->allocator->Free(weight_bias_sums_batch_);
      weight_bias_sums_batch_ = nullptr;
    }
    if (bias_ptr_ != nullptr) {
      ctx_->allocator->Free(bias_ptr_);
      bias_ptr_ = nullptr;
    }
  }
  MatmulQuantArg quant_params_;
  int8_t *a_r4x16_ptr_ = nullptr;
  int8_t *b_c16x4_ptr_ = nullptr;
  int8_t *c_ptr_ = nullptr;
  int *bias_ptr_ = nullptr;
  int *input_sums_ = nullptr;
  int *weight_bias_sums_ = nullptr;
  int8_t *b_c16x4_batch_ = nullptr;
  int *weight_bias_sums_batch_ = nullptr;
};  // namespace mindspore::kernel
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_MATMUL_INT8_H_
