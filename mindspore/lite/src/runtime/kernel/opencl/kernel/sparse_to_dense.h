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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SPARSE_TO_DENSE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SPARSE_TO_DENSE_H_

#include <vector>
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "mindspore/lite/nnacl/fp32/sparse_to_dense.h"

namespace mindspore::kernel {

class SparseToDenseOpenCLKernel : public OpenCLKernel {
 public:
  SparseToDenseOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                            const std::vector<lite::Tensor *> &outputs)
      : OpenCLKernel(parameter, inputs, outputs) {}

  ~SparseToDenseOpenCLKernel() override = default;

  int Init() override;
  int Run() override;
  int InitWeights() override;

 private:
  int InferShapeTo4D();
  int InitOutputToDefault();

 private:
  cl::Kernel kernel_;
  //  bool IndicesIsScalar{false};
  bool enable_fp16_{false};
  float default_{0.0f};
  float weight_scalar_{0.f};
  void *weight_vector_{nullptr};
  int input_dim_{1};
  std::vector<int32_t> output_shape_;

  size_t N_{1};
  size_t H_{1};
  size_t W_{1};
  size_t C_{1};
};
}  // namespace mindspore::kernel
#endif
