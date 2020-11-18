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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_STRIDED_SLICE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_STRIDED_SLICE_H_

#include <vector>
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "nnacl/fp32/slice.h"

namespace mindspore::kernel {

class SliceOpenCLKernel : public OpenCLKernel {
 public:
  SliceOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                    const std::vector<lite::Tensor *> &outputs)
      : OpenCLKernel(parameter, inputs, outputs) {}

  ~SliceOpenCLKernel() override = default;

  int Prepare() override;
  int Run() override;

  int CheckSpecs() override;
  void SetConstArgs() override;
  void SetGlobalLocal() override;

 private:
  int InitConstArgs();

  cl::Kernel kernel_;
  cl_int4 input_shape_{};
  cl_int4 output_shape_{};
  cl_int2 io_slices_{};
  cl_int4 begin_{};
  cl_int4 stride_{{1, 1, 1, 1}};
  cl_int4 size_{};
};

}  // namespace mindspore::kernel
#endif
