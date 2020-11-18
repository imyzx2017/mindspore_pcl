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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARGMAXWITHVALUEGPUKERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARGMAXWITHVALUEGPUKERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/argmaxwithvalue_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T, typename S>
class ArgmaxWithValueGpuKernel : public GpuKernel {
 public:
  ArgmaxWithValueGpuKernel() : input_size_(0), output_size_(0), bound_(0), outerSize_(0), innerSize_(0) {}
  ~ArgmaxWithValueGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 1);
    S *index = GetDeviceAddress<S>(outputs, 0);
    CalArgmaxWithValue(input, bound_, outerSize_, innerSize_, index, output,
                       reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    std::vector<size_t> shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 1);
    int dims = shape.size();
    int axis = static_cast<int>(GetAttr<int64_t>(kernel_node, "axis"));
    if (axis < 0) {
      axis += dims;
    }
    input_size_ = sizeof(T);
    for (auto x : shape) {
      input_size_ *= x;
    }
    output_size_ = sizeof(S);
    for (auto x : output_shape) {
      output_size_ *= x;
    }
    bound_ = shape[axis];
    outerSize_ = 1;
    for (int i = axis - 1; i >= 0; i--) {
      outerSize_ *= shape[i];
    }

    innerSize_ = 1;
    for (int i = axis + 1; i < dims; i++) {
      innerSize_ *= shape[i];
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    output_size_list_.push_back(output_size_ / sizeof(S) * sizeof(T));
  }

 private:
  size_t input_size_;
  size_t output_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  size_t bound_;
  size_t outerSize_;
  size_t innerSize_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARGMAXWITHVALUEGPUKERNEL_H_
