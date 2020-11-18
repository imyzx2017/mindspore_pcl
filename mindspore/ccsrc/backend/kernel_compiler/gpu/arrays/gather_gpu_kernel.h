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

#ifndef MINDSPORE_GATHER_GPU_KERNEL_H
#define MINDSPORE_GATHER_GPU_KERNEL_H

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/gather.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class GatherGpuFwdKernel : public GpuKernel {
 public:
  GatherGpuFwdKernel() : axis_(0), handle_(nullptr) {}
  ~GatherGpuFwdKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    S *index_addr = GetDeviceAddress<S>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    Gather(input_addr, index_addr, output_addr, dims_[0], dims_[1], dims_[2],
           reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "Argument number is " << input_num << ", but GatherGpuFwdKernel needs 2.";
    }
    input_shapes_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    index_shapes_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    output_shapes_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);

    axis_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "dim"));
    if (axis_ < 0) {
      axis_ = axis_ + SizeToInt(input_shapes_.size());
    }
    Reshape();
    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override { handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle(); }
  void InitSizeLists() override {
    size_t size = GetSize(input_shapes_, true);
    input_size_list_.push_back(size);

    size = GetSize(index_shapes_, false);
    input_size_list_.push_back(size);

    size = GetSize(output_shapes_, true);
    output_size_list_.push_back(size);
  }

 private:
  void Reshape() {
    size_t dim_before_axis = 1;
    for (size_t i = 0; i < IntToSize(axis_); i++) {
      dim_before_axis *= output_shapes_[i];
    }
    size_t dim_of_index = output_shapes_[IntToSize(axis_)];
    size_t dim_after_index = 1;
    for (size_t i = IntToSize(axis_) + 1; i < output_shapes_.size(); i++) {
      dim_after_index *= output_shapes_[i];
    }

    dims_[0] = dim_before_axis;
    dims_[1] = dim_of_index;
    dims_[2] = dim_after_index;
    return;
  }
  size_t GetSize(const std::vector<size_t> &shape, const bool flag = true) const {
    if (shape.size() == 0) {
      return 0;
    }
    size_t result = flag ? sizeof(T) : sizeof(S);
    for (size_t i = 0; i < shape.size(); i++) {
      result *= shape[i];
    }
    return result;
  }

  std::vector<size_t> input_shapes_;
  std::vector<size_t> index_shapes_;
  std::vector<size_t> output_shapes_;

  size_t dims_[3] = {};
  int axis_;
  cudnnHandle_t handle_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_GATHER_GPU_KERNEL_H
