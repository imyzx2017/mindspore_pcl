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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_BATCH_NORM_EX_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_BATCH_NORM_EX_GPU_KERNEL_H_

#include <string>
#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "utils/utils.h"

namespace mindspore {
namespace kernel {
template <typename T>
class FusedBatchNormExGpuKernel : public GpuKernel {
 public:
  FusedBatchNormExGpuKernel()
      : input_x_size_(0),
        input_z_size_(0),
        para_size_(0),
        output_size_(0),
        workspace_size_(0),
        reserve_size_(0),
        mode_(CUDNN_BATCHNORM_SPATIAL),
        bn_ops_(CUDNN_BATCHNORM_OPS_BN),
        epsilon_(10e-5),
        exp_avg_factor_(0.1),
        is_null_input_(false),
        x_desc_(nullptr),
        y_desc_(nullptr),
        z_desc_(nullptr),
        scale_bias_mean_var_desc_(nullptr),
        activation_desc_(nullptr),
        handle_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT) {}
  ~FusedBatchNormExGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    VARIABLE_NOT_USED(stream_ptr);
    if (is_null_input_) {
      return true;
    }
    auto x = GetDeviceAddress<T>(inputs, 0);
    auto scale = GetDeviceAddress<float>(inputs, 1);
    auto bias = GetDeviceAddress<float>(inputs, 2);
    auto runing_mean = GetDeviceAddress<float>(inputs, 3);
    auto runnig_variance = GetDeviceAddress<float>(inputs, 4);
    T *z = nullptr;
    if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
      z = GetDeviceAddress<T>(inputs, 5);
    }

    auto y = GetDeviceAddress<T>(outputs, 0);
    auto save_mean = GetDeviceAddress<float>(outputs, 3);
    auto save_variance = GetDeviceAddress<float>(outputs, 4);
    auto reserve_addr = GetDeviceAddress<float>(outputs, 5);
    T *workspace_addr = nullptr;
    if (workspace_size_ != 0) {
      workspace_addr = GetDeviceAddress<T>(workspace, 0);
    }
    const float alpha = 1;
    const float beta = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnBatchNormalizationForwardTrainingEx(handle_, mode_, bn_ops_, &alpha, &beta, x_desc_, x, z_desc_, z, y_desc_,
                                               y, scale_bias_mean_var_desc_, scale, bias, exp_avg_factor_, runing_mean,
                                               runnig_variance, epsilon_, save_mean, save_variance, activation_desc_,
                                               workspace_addr, workspace_size_, reserve_addr, reserve_size_),
      "Kernel launch failed");
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    MS_EXCEPTION_IF_NULL(kernel_node);
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    if (kernel_name == kFusedBatchNormEx) {
      bn_ops_ = CUDNN_BATCHNORM_OPS_BN;
    } else if (kernel_name == kFusedBatchNormExWithActivation) {
      bn_ops_ = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
    } else if (kernel_name == kFusedBatchNormExWithAddAndActivation) {
      bn_ops_ = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    } else {
      MS_LOG(EXCEPTION) << "Invalid kernel name: " << kernel_name;
    }

    InitResource();
    mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    epsilon_ = GetAttr<float>(kernel_node, "epsilon");
    exp_avg_factor_ = GetAttr<float>(kernel_node, "momentum");

    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
      if (input_num != 6) {
        MS_LOG(EXCEPTION) << "input tensor size is " << input_num << ", " << kernel_name << " should be 6";
      }
    } else {
      if (input_num != 5) {
        MS_LOG(EXCEPTION) << "input tensor size is " << input_num << ", " << kernel_name << "  should be 5";
      }
    }

    auto shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    if (shape.size() != 4) {
      MS_LOG(EXCEPTION) << "tensor shape is " << shape.size() << ", FusedBatchNormExGpuKernel should be 4";
    }
    is_null_input_ = CHECK_NULL_INPUT(shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "FusedBatchNormExGpuKernel input is null";
      InitSizeLists();
      return true;
    }
    auto format = AnfAlgo::GetInputFormat(kernel_node, 0);
    auto format_attr = GetAttr<std::string>(kernel_node, "data_format");
    if (format_attr == kOpFormat_NHWC) {
      format = kOpFormat_NHWC;
    }
    SetTensorDescriptor(format, shape);
    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(x_desc_), "Destroy x desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(y_desc_), "Destroy y desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(scale_bias_mean_var_desc_), "Destroy para desc failed");
    if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
      CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(z_desc_), "Destroy z desc failed");
    }

    if (bn_ops_ != CUDNN_BATCHNORM_OPS_BN) {
      CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyActivationDescriptor(activation_desc_),
                                 "Destroy activation descriptor failed");
    }
  }

 protected:
  void InitResource() override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&x_desc_), "Create x desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&y_desc_), "Create y desc failed");
    if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&z_desc_), "Create z desc failed");
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&scale_bias_mean_var_desc_), "Create para desc failed");

    if (bn_ops_ != CUDNN_BATCHNORM_OPS_BN) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateActivationDescriptor(&activation_desc_),
                                  "Create activation descriptor failed");
    }
  }

  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(x_desc_, &input_x_size_), "Get input x size failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(scale_bias_mean_var_desc_, &para_size_),
                                  "Get para size failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(y_desc_, &output_size_), "Get output size failed");
      if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
        CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(z_desc_, &input_z_size_), "Get input z size failed");
      }

      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
                                    handle_, mode_, bn_ops_, x_desc_, z_desc_, y_desc_, scale_bias_mean_var_desc_,
                                    activation_desc_, &workspace_size_),
                                  "cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize failed");

      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
                                    handle_, mode_, bn_ops_, activation_desc_, x_desc_, &reserve_size_),
                                  "Get reserve size failed");
    }

    input_size_list_.push_back(input_x_size_);  // input x
    input_size_list_.push_back(para_size_);     // scale
    input_size_list_.push_back(para_size_);     // bias
    input_size_list_.push_back(para_size_);     // mean
    input_size_list_.push_back(para_size_);     // variance
    if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
      input_size_list_.push_back(input_z_size_);  // input z
    }

    output_size_list_.push_back(output_size_);   // output
    output_size_list_.push_back(para_size_);     // save scale
    output_size_list_.push_back(para_size_);     // save bias
    output_size_list_.push_back(para_size_);     // save mean
    output_size_list_.push_back(para_size_);     // save variance
    output_size_list_.push_back(reserve_size_);  // reserve space

    workspace_size_list_.push_back(workspace_size_);
  }

 private:
  void SetTensorDescriptor(const std::string &format, const std::vector<size_t> &shape) {
    cudnnTensorFormat_t cudnn_format;
    int batch, channel, height, width;
    if (format == kOpFormat_NHWC) {
      batch = SizeToInt(shape[0]);
      height = SizeToInt(shape[1]);
      width = SizeToInt(shape[2]);
      channel = SizeToInt(shape[3]);
      cudnn_format = CUDNN_TENSOR_NHWC;
    } else {
      batch = SizeToInt(shape[0]);
      channel = SizeToInt(shape[1]);
      height = SizeToInt(shape[2]);
      width = SizeToInt(shape[3]);
      cudnn_format = CUDNN_TENSOR_NCHW;
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(x_desc_, cudnn_format, cudnn_data_type_, batch, channel, height, width),
      "Set x desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(y_desc_, cudnn_format, cudnn_data_type_, batch, channel, height, width),
      "Set y desc failed");

    if (bn_ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnSetTensor4dDescriptor(z_desc_, cudnn_format, cudnn_data_type_, batch, channel, height, width),
        "Set z desc failed");
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(scale_bias_mean_var_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, channel, 1, 1),
      "Set para desc failed");

    if (bn_ops_ != CUDNN_BATCHNORM_OPS_BN) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnSetActivationDescriptor(activation_desc_, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0),
        "cudnnSetActivationDescriptor failed");
    }
  }

  size_t input_x_size_;
  size_t input_z_size_;
  size_t para_size_;
  size_t output_size_;
  size_t workspace_size_;
  size_t reserve_size_;
  cudnnBatchNormMode_t mode_;
  cudnnBatchNormOps_t bn_ops_;
  double epsilon_;
  double exp_avg_factor_;
  bool is_null_input_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  cudnnTensorDescriptor_t z_desc_;
  cudnnTensorDescriptor_t scale_bias_mean_var_desc_;
  cudnnActivationDescriptor_t activation_desc_;

  cudnnHandle_t handle_;
  cudnnDataType_t cudnn_data_type_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_FUSED_BATCH_NORM_EX_GPU_KERNEL_H_
