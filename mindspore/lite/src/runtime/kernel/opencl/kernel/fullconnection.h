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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_FULLCONNECTION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_FULLCONNECTION_H_

#include <vector>

#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "nnacl/matmul_parameter.h"

namespace mindspore::kernel {

class FullConnectionOpenCLKernel : public OpenCLKernel {
 public:
  FullConnectionOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                             const std::vector<lite::Tensor *> &outputs)
      : OpenCLKernel(parameter, inputs, outputs) {}
  ~FullConnectionOpenCLKernel() override = default;

  int Run() override;
  int Prepare() override;
  int CheckSpecs() override;
  int InitWeights() override;
  void SetConstArgs() override;
  void SetGlobalLocal() override;
  int Init() override;

 private:
  cl::Kernel kernel_;
  void *padWeight_{nullptr};
  void *bias_{nullptr};
  bool enable_fp16_{false};
  bool transposeA{false};
  bool transposeB{true};
  float activation_min_{-FLT_MAX};
  float activation_max_{FLT_MAX};
  Image2DInfo inShape = Image2DInfo(nullptr);
  Image2DInfo outShape = Image2DInfo(nullptr);
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_FULLCONNECTION_H_
