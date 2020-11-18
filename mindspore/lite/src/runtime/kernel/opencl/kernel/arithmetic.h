/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_ARITHMETIC_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_ARITHMETIC_H_

#include <vector>
#include <string>
#include "src/runtime/kernel/arm/fp32/arithmetic_fp32.h"
#include "src/runtime/kernel/opencl/opencl_kernel.h"

namespace mindspore::kernel {

class ArithmeticOpenCLKernel : public OpenCLKernel {
 public:
  ArithmeticOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs)
      : OpenCLKernel(parameter, inputs, outputs) {}
  ~ArithmeticOpenCLKernel() override = default;

  int Run() override;
  int Prepare() override;
  int CheckSpecs() override;
  int InitWeights() override;
  void SetConstArgs() override;
  void SetGlobalLocal() override;

 private:
  cl::Kernel kernel_;
  bool element_flag_{true};
  float activation_min_{-FLT_MAX};
  float activation_max_{FLT_MAX};
  std::vector<std::vector<int>> inputs_nhwc_shapes_;
  std::vector<void *> inputs_weight_ptrs_;
  std::string kernel_name_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_ARITHMETIC_H_
