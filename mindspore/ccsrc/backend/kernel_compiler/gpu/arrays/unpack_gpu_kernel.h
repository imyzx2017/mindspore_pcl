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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_UNPACK_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_UNPACK_GPU_KERNEL_H

#include <vector>
#include <memory>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/unpack.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class UnpackGpuFwdKernel : public GpuKernel {
 public:
  UnpackGpuFwdKernel() : axis_(0), output_num_(0), input_size_(1), dims_after_axis_(1), outputs_host_(nullptr) {}
  ~UnpackGpuFwdKernel() override = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);
    T **outputs_array = GetDeviceAddress<T *>(workspace, 0);
    for (size_t i = 0; i < outputs.size(); i++) {
      outputs_host_[i] = GetDeviceAddress<T>(outputs, i);
    }
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(outputs_array, outputs_host_.get(), sizeof(T *) * output_num_,
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "Unpack opt cudaMemcpyAsync outputs failed");
    UnpackKernel(SizeToInt(input_size_), output_num_, dims_after_axis_, outputs_array, input,
                 reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    if (!CheckParam(kernel_node)) {
      return false;
    }
    axis_ = static_cast<int32_t>(GetAttr<int64_t>(kernel_node, "axis"));
    if (axis_ < 0) {
      auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
      axis_ += SizeToInt(input_shape.size());
    }
    auto origin_data_format = AnfAlgo::GetOriginDataFormat(kernel_node);
    auto input_format = AnfAlgo::GetInputFormat(kernel_node, 0);
    axis_ = AxisTransform(origin_data_format, input_format, axis_);

    output_num_ = static_cast<int32_t>(GetAttr<int64_t>(kernel_node, "num"));
    outputs_host_ = std::make_unique<T *[]>(output_num_);
    for (int i = 0; i < output_num_; i++) {
      size_t _size = 1;
      auto _shape = AnfAlgo::GetOutputDeviceShape(kernel_node, i);
      for (size_t j = 0; j < _shape.size(); j++) {
        _size *= _shape[j];
      }
      output_size_list_.push_back(_size * sizeof(T));
    }
    workspace_size_list_.push_back(sizeof(T *) * output_num_);

    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    for (int i = 0; i < SizeToInt(input_shape.size()); i++) {
      input_size_ *= input_shape[i];
      if (i > axis_) {
        dims_after_axis_ *= input_shape[i];
      }
    }
    input_size_list_.push_back(input_size_ * sizeof(T));
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {}

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "input number is " << input_num << ", but UnpackGpuFwdKernel needs 1 input.";
      return false;
    }
    return true;
  }
  int axis_;
  int output_num_;
  size_t input_size_;
  int dims_after_axis_;
  std::unique_ptr<T *[]> outputs_host_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_UNPACK_GPU_KERNEL_H
