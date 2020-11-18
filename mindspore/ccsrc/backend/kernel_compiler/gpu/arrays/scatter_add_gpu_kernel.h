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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SCATTER_ADD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SCATTER_ADD_GPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/scatter_add_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class ScatterAddKernel : public GpuKernel {
 public:
  ScatterAddKernel() { ResetResource(); }
  ~ScatterAddKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);
    int *indices = GetDeviceAddress<int>(inputs, 1);
    T *updates = GetDeviceAddress<T>(inputs, 2);
    T *output = GetDeviceAddress<T>(outputs, 0);
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(&output[0], &input[0], input_size_ * sizeof(T), cudaMemcpyDeviceToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync output failed");
    CalScatterAdd(inner_size_, indices_size_, indices, updates, output, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 3) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but ScatterAdd needs 3 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but ScatterAdd has 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    input_size_ = 1;
    inner_size_ = 1;
    for (size_t i = 1; i < input_shape.size(); i++) {
      inner_size_ *= input_shape[i];
    }
    input_size_ = input_shape[0] * inner_size_;
    auto indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    indices_size_ = 1;
    for (size_t i = 0; i < indices_shape.size(); i++) {
      indices_size_ *= indices_shape[i];
    }
    updates_size_ = indices_size_ * inner_size_;
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 0;
    inner_size_ = 0;
    indices_size_ = 0;
    updates_size_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    input_size_list_.push_back(indices_size_ * sizeof(int));
    input_size_list_.push_back(updates_size_ * sizeof(T));
    output_size_list_.push_back(input_size_ * sizeof(T));
  }

 private:
  int input_size_;
  int inner_size_;
  int indices_size_;
  int updates_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SCATTER_ADD_GPU_KERNEL_H_
