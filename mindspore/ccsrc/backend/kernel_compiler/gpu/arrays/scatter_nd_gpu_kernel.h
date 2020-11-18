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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_SCATTER_ND_GPU_KERNEL_H
#define MINDSPORE_CCSRC_KERNEL_GPU_SCATTER_ND_GPU_KERNEL_H

#include <vector>
#include <algorithm>
#include "backend/kernel_compiler/gpu/cuda_impl/scatter_nd.cuh"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class ScatterNdGpuFwdKernel : public GpuKernel {
 public:
  ScatterNdGpuFwdKernel()
      : input_size_(1),
        indices_size_(1),
        output_size_(1),
        block_size_(1),
        indices_stride_(nullptr),
        work_shape_(nullptr),
        indices_dim_0_(0),
        indices_dim_1_(0),
        memcpy_flag_(false) {}
  ~ScatterNdGpuFwdKernel() {
    if (indices_stride_ != nullptr) {
      device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(static_cast<void *>(indices_stride_));
    }
    if (work_shape_ != nullptr) {
      device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(static_cast<void *>(work_shape_));
    }
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    S *indices = GetDeviceAddress<S>(inputs, 0);
    T *update = GetDeviceAddress<T>(inputs, 1);
    T *output = GetDeviceAddress<T>(outputs, 0);

    if (!memcpy_flag_) {
      const size_t indices_len = sizeof(S) * vec_indices_stride_.size();
      const size_t vec_work_len = sizeof(S) * vec_work_shape_.size();
      CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(indices_stride_, &vec_indices_stride_[0], indices_len,
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpy failed in ScatterNdGpuFwdKernel::Launch.");
      CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(work_shape_, &vec_work_shape_[0], vec_work_len, cudaMemcpyHostToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpy failed in ScatterNdGpuFwdKernel::Launch.");
      memcpy_flag_ = true;
    }

    CHECK_CUDA_RET_WITH_EXCEPT(
      cudaMemsetAsync(output, static_cast<T>(0.0), output_size_, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemSet failed in ScatterNdGpuFwdKernel::Launch.");

    const size_t input_size = input_size_ / sizeof(T);
    const size_t output_size = output_size_ / sizeof(T);

    ScatterNd(indices, update, output, block_size_, input_size, output_size, indices_dim_0_, indices_dim_1_,
              indices_stride_, work_shape_, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    memcpy_flag_ = false;
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but transpose needs 2 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but transpose needs 1 output.";
      return false;
    }

    input_shapes_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    indices_shapes_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    output_shapes_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);

    std::vector<int64_t> shape_me = GetAttr<std::vector<int64_t>>(kernel_node, "shape");
    (void)std::transform(shape_me.begin(), shape_me.end(), std::back_inserter(vec_work_shape_),
                         [](const int64_t &value) { return static_cast<S>(value); });

    GetSize();

    const size_t indices_len = sizeof(S) * vec_indices_stride_.size();
    void *indices_stride_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(indices_len);
    if (indices_stride_work == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to alloc indices_stride_work, size: " << indices_len;
    }
    indices_stride_ = static_cast<S *>(indices_stride_work);

    const size_t vec_work_len = sizeof(S) * vec_work_shape_.size();
    void *work_shape_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(vec_work_len);
    if (work_shape_work == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to alloc work_shape_work, size: " << vec_work_len;
    }
    work_shape_ = static_cast<S *>(work_shape_work);

    InitSizeLists();

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(indices_size_);
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    return;
  }

  void GetSize() {
    indices_size_ = sizeof(S);
    for (size_t i = 0; i < indices_shapes_.size(); i++) {
      indices_size_ *= indices_shapes_[i];
    }
    input_size_ = sizeof(T);
    for (size_t i = 0; i < input_shapes_.size(); i++) {
      input_size_ *= input_shapes_[i];
    }
    output_size_ = sizeof(T);
    for (size_t i = 0; i < output_shapes_.size(); i++) {
      output_size_ *= output_shapes_[i];
    }

    // calculate indices dim 0/1
    indices_dim_0_ = indices_shapes_[0];
    indices_dim_1_ = indices_shapes_[indices_shapes_.size() - 1];

    // calculate block_size
    for (size_t i = indices_dim_1_; i < output_shapes_.size(); i++) {
      block_size_ *= output_shapes_[i];
    }

    // calculate indices_stride
    vec_indices_stride_.resize(indices_dim_1_, 0);
    vec_indices_stride_[indices_dim_1_ - 1] = block_size_;

    for (size_t i = indices_dim_1_ - 1; i > 0; --i) {
      vec_indices_stride_[i - 1] = vec_indices_stride_[i] * output_shapes_[i];
    }
  }

 private:
  std::vector<size_t> input_shapes_;
  std::vector<size_t> indices_shapes_;
  std::vector<size_t> output_shapes_;
  std::vector<S> vec_indices_stride_;
  std::vector<S> vec_work_shape_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  size_t input_size_;
  size_t indices_size_;
  size_t output_size_;
  size_t block_size_;

  S *indices_stride_;
  S *work_shape_;
  size_t indices_dim_0_;
  size_t indices_dim_1_;
  bool memcpy_flag_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_SCATTER_ND_GPU_KERNEL_H
