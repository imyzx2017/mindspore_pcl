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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DYNAMIC_SHAPE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DYNAMIC_SHAPE_GPU_KERNEL_H_

#include <cuda_runtime.h>

#include <vector>

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class DynamicShapeGpuKernel : public GpuKernel {
 public:
  DynamicShapeGpuKernel() { ResetResource(); }
  ~DynamicShapeGpuKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    int *output_device_address = GetDeviceAddress<int>(outputs, 0);
    size_t prev_node_output_shape_size = prev_node_output_shape_.size() * sizeof(int);
    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemcpyAsync(output_device_address, prev_node_output_shape_.data(), prev_node_output_shape_size,
                      cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync prev_node_output_shape failed");

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_count = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_count != 1) {
      MS_LOG(EXCEPTION) << input_count << " arguments were provided, but DynamicShapeGpuKernel expects 1.";
    }

    std::vector<size_t> prev_node_output_shape_tmp = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    input_size_ = 1;
    for (const size_t &e : prev_node_output_shape_tmp) {
      input_size_ *= e;
      // shapes are Tensors with elements of type int32, but GetPrevNodeOutputInferShape returns vector of size_t,
      // so we use an int* for allocated output memory and cast to an int here, otherwise the memcpy will fail with a
      // silently.
      prev_node_output_shape_.push_back(e);
    }

    output_size_ = prev_node_output_shape_.size();

    InitSizeLists();

    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = -1;
    output_size_ = -1;
    prev_node_output_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(output_size_ * sizeof(int));
  }

 private:
  size_t input_size_;
  size_t output_size_;
  std::vector<int> prev_node_output_shape_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DYNAMIC_SHAPE_GPU_KERNEL_H_
