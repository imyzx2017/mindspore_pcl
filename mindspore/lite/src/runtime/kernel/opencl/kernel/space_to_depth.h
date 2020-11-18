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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SAPCE_TO_DEPTH_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SAPCE_TO_DEPTH_H_

#include <vector>
#include <string>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "nnacl/fp32/space_to_depth.h"

namespace mindspore::kernel {
class SpaceToDepthOpenCLKernel : public OpenCLKernel {
 public:
  SpaceToDepthOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                           const std::vector<lite::Tensor *> &outputs)
      : OpenCLKernel(parameter, inputs, outputs) {}
  ~SpaceToDepthOpenCLKernel() override = default;

  int Run() override;
  int Prepare() override;
  int CheckSpecs() override;
  void SetConstArgs() override;
  void SetGlobalLocal() override;

 private:
  cl::Kernel kernel_;
  Image2DInfo in_shape_ = Image2DInfo(nullptr);
  Image2DInfo out_shape_ = Image2DInfo(nullptr);
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SAPCE_TO_DEPTH_H_
