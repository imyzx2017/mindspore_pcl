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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_FUSED_BATCHNORM_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_FUSED_BATCHNORM_FP16_H_

#include <vector>
#include "src/runtime/kernel/arm/fp32/fused_batchnorm_fp32.h"

namespace mindspore::kernel {
class FusedBatchnormFp16CPUKernel : public FusedBatchnormCPUKernel {
 public:
  FusedBatchnormFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                              const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx,
                              const mindspore::lite::PrimitiveC *primitive)
      : FusedBatchnormCPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  virtual ~FusedBatchnormFp16CPUKernel() {}

  virtual int DoExecute(int task_id);
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_FUSED_BATCHNORM_FP16_H_
