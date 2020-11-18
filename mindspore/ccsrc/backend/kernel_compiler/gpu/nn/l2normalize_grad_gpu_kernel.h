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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_L2NORMALIZE_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_L2NORMALIZE_GRAD_GPU_KERNEL_H_

#include <map>
#include <string>
#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/broadcast_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/l2normalize_impl.cuh"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
namespace mindspore {
namespace kernel {
constexpr int MAX_DIMS = 7;
constexpr size_t INPUT_SIZE = 3;
template <typename T>
class L2NormalizeGradGpuKernel : public GpuKernel {
 public:
  L2NormalizeGradGpuKernel()
      : cudnn_handle_(nullptr),
        data_type_(CUDNN_DATA_FLOAT),
        nan_prop_(CUDNN_NOT_PROPAGATE_NAN),
        reduce_indices_(CUDNN_REDUCE_TENSOR_NO_INDICES),
        reduce_tensor_descriptor_(nullptr),
        reduce_sum_tensor_descriptor_(nullptr),
        inputA_descriptor_(nullptr),
        outputC_descriptor_(nullptr),
        all_match_(false),
        is_null_input_(false),
        output_size_(0),
        workspace_size_(0),
        epsilon_(0.0),
        axis_(0) {}
  ~L2NormalizeGradGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *x_addr = GetDeviceAddress<T>(inputs, 0);
    T *y_addr = GetDeviceAddress<T>(inputs, 1);
    T *dy_addr = GetDeviceAddress<T>(inputs, 2);
    T *dx_addr = GetDeviceAddress<T>(outputs, 0);
    T *reduce_workspace_addr = GetDeviceAddress<T>(workspace, 0);
    T *reduce_y_dy_workspace_addr = GetDeviceAddress<T>(workspace, 1);
    T *workspace_addr = GetDeviceAddress<T>(workspace, 2);
    T *workspace_y_dy_addr = GetDeviceAddress<T>(workspace, 3);

    const float alpha = 1;
    const float beta = 0;

    if (all_match_) {
      CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(reduce_workspace_addr, x_addr, input_size_list_[0],
                                                 cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync failed in L2Normalize::Launch.");
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, workspace_addr, workspace_size_list_[2],
                          &alpha, inputA_descriptor_, x_addr, &beta, outputC_descriptor_, reduce_workspace_addr),
        "cudnnReduceTensor failed.");
    }
    GetMaxWithEpsAndValue(workspace_size_list_[0] / sizeof(T), epsilon_, reduce_workspace_addr,
                          reinterpret_cast<cudaStream_t>(stream_ptr));
    BroadcastArith(output_shape_, output_shape_, output_shape_, BROADCAST_TYPE_MUL, y_addr, dy_addr, dx_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    if (all_match_) {
      CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(reduce_y_dy_workspace_addr, dx_addr, output_size_list_[0],
                                                 cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync failed in L2Normalize::Launch.");
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnReduceTensor(cudnn_handle_, reduce_sum_tensor_descriptor_, nullptr, 0, workspace_y_dy_addr,
                          workspace_size_list_[3], &alpha, inputA_descriptor_, dx_addr, &beta, outputC_descriptor_,
                          reduce_y_dy_workspace_addr),
        "cudnnReduceTensor failed.");
    }
    BroadcastArith(rhs_shape_, lhs_shape_, output_shape_, BROADCAST_TYPE_MUL, reduce_y_dy_workspace_addr, y_addr,
                   dx_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    BroadcastArith(output_shape_, output_shape_, output_shape_, BROADCAST_TYPE_SUB, dy_addr, dx_addr, dx_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    BroadcastArith(output_shape_, rhs_shape_, output_shape_, BROADCAST_TYPE_REALDIV, dx_addr, reduce_workspace_addr,
                   dx_addr, reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != INPUT_SIZE) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but l2normalize op needs 3 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but l2normalize op needs 1 output.";
      return false;
    }

    int input_dim_length = SizeToInt(AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0).size());
    int axis = static_cast<int>(GetAttr<int64_t>(kernel_node, "axis"));
    axis_ = axis < 0 ? (axis + input_dim_length) : axis;

    epsilon_ = GetAttr<float>(kernel_node, "epsilon");

    for (size_t i = 0; i < INPUT_SIZE; i++) {
      input_shape_list_.emplace_back(AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, i));
    }
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    for (auto &shape : input_shape_list_) {
      if (output_shape != shape) {
        MS_LOG(EXCEPTION) << "Input shape and output shape should be same!";
      }
    }

    output_size_ = sizeof(T);
    for (auto dim : output_shape) {
      output_size_ *= dim;
    }

    is_null_input_ = CHECK_NULL_INPUT(input_shape_list_[0]);
    if (is_null_input_) {
      MS_LOG(WARNING) << "L2NormalizeGPUKernel input is null";
      InitSizeLists();
      return true;
    }
    if (input_shape_list_[0].size() > MAX_DIMS) {
      MS_LOG(EXCEPTION) << "Broadcast operation not support dim greater than 7";
    }

    std::vector<size_t> output_reduce_shape = output_shape;
    output_reduce_shape[axis_] = 1;

    lhs_shape_.resize(MAX_DIMS, 1);
    rhs_shape_.resize(MAX_DIMS, 1);
    output_shape_.resize(MAX_DIMS, 1);
    all_match_ = true;
    for (size_t i = 0; i < output_shape.size(); i++) {
      output_shape_[i] = output_shape[i];
      lhs_shape_[i] = output_shape[i];
      rhs_shape_[i] = output_reduce_shape[i];
      if (lhs_shape_[i] != rhs_shape_[i]) {
        all_match_ = false;
      }
    }

    InferInAndOutDesc(input_shape_list_[0], output_reduce_shape);
    InferArrayReduceType(kernel_node);

    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateReduceTensorDescriptor(&reduce_tensor_descriptor_),
                                "cudnnCreateReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateReduceTensorDescriptor(&reduce_sum_tensor_descriptor_),
                                "cudnnCreateReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&inputA_descriptor_),
                                "cudnnCreateTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&outputC_descriptor_),
                                "cudnnCreateTensorDescriptor failed.");
  }
  void InitSizeLists() override {
    for (auto &shape : input_shape_list_) {
      size_t input_size = sizeof(T);
      for (auto &size : shape) {
        input_size *= size;
      }
      input_size_list_.emplace_back(input_size);
    }

    output_size_list_.push_back(output_size_);

    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(outputC_descriptor_, &workspace_size_),
                                "cudnnGetTensorSizeInBytes failed.");
    workspace_size_list_.push_back(workspace_size_);
    workspace_size_list_.push_back(workspace_size_);

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnGetReductionWorkspaceSize(cudnn_handle_, reduce_tensor_descriptor_, inputA_descriptor_, outputC_descriptor_,
                                     &workspace_size_),
      "cudnnGetReductionWorkspaceSize failed.");
    workspace_size_list_.push_back(workspace_size_);

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnGetReductionWorkspaceSize(cudnn_handle_, reduce_sum_tensor_descriptor_, inputA_descriptor_,
                                     outputC_descriptor_, &workspace_size_),
      "cudnnGetReductionWorkspaceSize failed.");
    workspace_size_list_.push_back(workspace_size_);

    return;
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyReduceTensorDescriptor(reduce_tensor_descriptor_),
                               "cudnnDestroyReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyReduceTensorDescriptor(reduce_sum_tensor_descriptor_),
                               "cudnnDestroyReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(inputA_descriptor_),
                               "cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(outputC_descriptor_),
                               "cudnnDestroyTensorDescriptor failed.");
  }
  void InferArrayReduceType(const CNodePtr &kernel_node) {
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetReduceTensorDescriptor(reduce_tensor_descriptor_, CUDNN_REDUCE_TENSOR_NORM2, CUDNN_DATA_FLOAT, nan_prop_,
                                     reduce_indices_, CUDNN_32BIT_INDICES),
      "cudnnSetReduceTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetReduceTensorDescriptor(reduce_sum_tensor_descriptor_, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT,
                                     nan_prop_, reduce_indices_, CUDNN_32BIT_INDICES),
      "cudnnSetReduceTensorDescriptor failed");
    return;
  }
  void InferInAndOutDesc(const std::vector<size_t> &input_shape, const std::vector<size_t> &output_shape) {
    std::vector<size_t> inputA;
    std::vector<size_t> outputC_shape = output_shape;
    constexpr int split_dim = 4;

    if (input_shape.size() <= split_dim) {
      ShapeNdTo4d(input_shape, &inputA);
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensor4dDescriptor(inputA_descriptor_, CUDNN_TENSOR_NCHW, data_type_,
                                                             inputA[0], inputA[1], inputA[2], inputA[3]),
                                  "cudnnSetTensor4dDescriptor failed");
    } else {
      CudnnSetTensorNdDescriptor(input_shape, inputA_descriptor_, data_type_);
      for (auto dim : input_shape) {
        inputA.emplace_back(dim);
      }
    }

    std::vector<size_t> outputC;

    if (outputC_shape.size() <= split_dim) {
      ShapeNdTo4d(outputC_shape, &outputC);
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensor4dDescriptor(outputC_descriptor_, CUDNN_TENSOR_NCHW, data_type_,
                                                             outputC[0], outputC[1], outputC[2], outputC[3]),
                                  "cudnnSetTensor4dDescriptor failed");
    } else {
      CudnnSetTensorNdDescriptor(outputC_shape, outputC_descriptor_, data_type_);
      for (auto dim : outputC_shape) {
        outputC.emplace_back(dim);
      }
    }

    return;
  }

  cudnnHandle_t cudnn_handle_;
  cudnnDataType_t data_type_;
  cudnnNanPropagation_t nan_prop_;
  cudnnReduceTensorIndices_t reduce_indices_;
  cudnnReduceTensorDescriptor_t reduce_tensor_descriptor_;
  cudnnReduceTensorDescriptor_t reduce_sum_tensor_descriptor_;
  cudnnTensorDescriptor_t inputA_descriptor_;
  cudnnTensorDescriptor_t outputC_descriptor_;

  bool all_match_;
  bool is_null_input_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  std::vector<std::vector<size_t> > input_shape_list_;
  size_t output_size_;
  size_t workspace_size_;
  float epsilon_;
  int axis_;
  std::vector<size_t> lhs_shape_;
  std::vector<size_t> rhs_shape_;
  std::vector<size_t> output_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_L2NORMALIZE_GRAD_GPU_KERNEL_H_
