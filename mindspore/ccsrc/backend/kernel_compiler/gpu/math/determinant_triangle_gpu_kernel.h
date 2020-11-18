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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DETRMINANT_TRIANGLE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DETRMINANT_TRIANGLE_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/determinant_triangle_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class DetTriangleGpuKernel : public GpuKernel {
 public:
  DetTriangleGpuKernel() : input_size_(sizeof(T)), output_size_(sizeof(T)) {}
  ~DetTriangleGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    if (!CheckTriangle(input_addr, fill_mode_, matrix_n_, outputs[0]->size / sizeof(T),
                       reinterpret_cast<cudaStream_t>(stream_ptr))) {
      if (fill_mode_ == 0) {
        MS_LOG(ERROR) << "The elements in the upper half of the maxtices should be all 0.";
      } else if (fill_mode_ == 1) {
        MS_LOG(ERROR) << "The elements in the lower half of the maxtices should be all 0.";
      } else {
        MS_LOG(ERROR) << "The input matrix should be either upper filled or lower filled.";
      }
      return false;
    }
    DetTriangle(input_addr, output_addr, matrix_n_, outputs[0]->size / sizeof(T),
                reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but DetTriangle needs 1 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but DetTriangle needs 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }
    matrix_n_ = input_shape[input_shape.size() - 1];
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < output_shape.size(); i++) {
      output_size_ *= output_shape[i];
    }
    if (output_size_ != input_size_ / matrix_n_ / matrix_n_) {
      MS_LOG(ERROR) << "The output shape is wrong.";
      return false;
    }
    if (input_shape[input_shape.size() - 2] != input_shape[input_shape.size() - 1]) {
      MS_LOG(ERROR) << "The maxtices should be in shape of square.";
      return false;
    }
    fill_mode_ = static_cast<int>(GetValue<int64_t>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("fill_mode")));
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  size_t input_size_;
  size_t output_size_;
  size_t matrix_n_;
  int fill_mode_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DETRMINANT_TRIANGLE_GPU_KERNEL_H_
