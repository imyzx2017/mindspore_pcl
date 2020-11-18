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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_FILL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_FILL_H_

#include <vector>
#include "mindspore/lite/nnacl/fp32/fill.h"
#include "mindspore/lite/nnacl/shape.h"
#include "src/runtime/kernel/opencl/opencl_kernel.h"

namespace mindspore::kernel {

class FillOpenCLKernel : public OpenCLKernel {
 public:
  FillOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                   const std::vector<lite::Tensor *> &outputs)
      : OpenCLKernel(parameter, inputs, outputs) {}

  ~FillOpenCLKernel() override = default;

  int Init() override;

  int Run() override;

 private:
  int RunFill();
  int RunShape();
  cl::Kernel kernel_;

 private:
  float default_{0.0f};
};

}  // namespace mindspore::kernel
#endif
