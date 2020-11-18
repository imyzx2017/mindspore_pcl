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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_IM2COLGPUKERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_IM2COLGPUKERNEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/pad_impl.cuh"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class Im2ColGpuFwdKernel : public GpuKernel {
 public:
  Im2ColGpuFwdKernel()
      : cudnn_handle_(nullptr),
        input_desc_(nullptr),
        output_desc_(nullptr),
        filter_desc_(nullptr),
        conv_desc_(nullptr),
        padded_desc_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        old_height_(0),
        old_width_(0),
        pad_height_(0),
        pad_width_(0),
        pad_top_(0),
        pad_left_(0),
        n_(0),
        c_(0),
        is_null_input_(false),
        input_size_(0),
        output_size_(0),
        padded_size_(0),
        workspace_size_(0),
        use_pad_(true) {}
  ~Im2ColGpuFwdKernel() override { DestroyResource(); }
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    if ((pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) && use_pad_) {
      T *padded_addr = GetDeviceAddress<T>(workspace, 0);
      CalPad(padded_size_ / sizeof(T), input_addr, n_, c_, old_height_, old_width_, old_height_ + pad_height_,
             old_width_ + pad_width_, pad_top_, pad_left_, pad_value_, padded_addr,
             reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnIm2Col(cudnn_handle_, padded_desc_, padded_addr, filter_desc_, conv_desc_, output_addr),
        "cudnnIm2ColForward failed");
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnIm2Col(cudnn_handle_, input_desc_, input_addr, filter_desc_, conv_desc_, output_addr),
        "cudnnIm2ColForward failed");
    }

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    if (!CheckParam(kernel_node)) {
      return false;
    }
    auto in_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    std::vector<int> filter_shape;
    std::vector<int64_t> filter_shape_me = GetAttr<std::vector<int64_t>>(kernel_node, "kernel_size");
    (void)std::transform(filter_shape_me.begin(), filter_shape_me.end(), std::back_inserter(filter_shape),
                         [](const int64_t &value) { return static_cast<int>(value); });
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(in_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "cudnnIm2ColForward input is null.";
      InitSizeLists();
      return true;
    }
    Set4DDesc(in_shape, filter_shape, output_shape);
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetConvolutionGroupCount(conv_desc_, 1), "cudnnSetConvGroupCount failed");
    pad_height_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "pad"));
    pad_width_ = pad_height_;
    pad_mode_ = GetAttr<std::string>(kernel_node, "pad_mode");
    SetStrideAndDilation(kernel_node);
    if (pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) {
      SetPad(in_shape, kernel_node);
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_height_ = 0;
        pad_width_ = 0;
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnSetConvolution2dDescriptor(conv_desc_, pad_height_, pad_width_, stride_[2], stride_[3], dilation_[2],
                                        dilation_[3], CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
        "cudnnSetConvolution2dDescriptor failed");
    }
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH),
                                  "cudnnSetConvolutionMathType failed.")
    }
    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc_),
                               "cudnnDestroyConvolutionDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyFilterDescriptor(filter_desc_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(padded_desc_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(output_desc_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(input_desc_), "cudnnDestroyTensorDescriptor failed");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&input_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&output_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&padded_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateFilterDescriptor(&filter_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateConvolutionDescriptor(&conv_desc_),
                                "cudnnCreateConvolutionDescriptor failed");
  }

  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(input_desc_, reinterpret_cast<size_t *>(&input_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(output_desc_, reinterpret_cast<size_t *>(&output_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(padded_desc_, reinterpret_cast<size_t *>(&padded_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    if ((pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) && use_pad_ && !is_null_input_) {
      workspace_size_list_.push_back(padded_size_);
    }
    return;
  }

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but Im2Col needs 1 inputs.";
      return false;
    }

    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but Im2Col needs 1 output.";
      return false;
    }
    return true;
  }
  void SetPad(const std::vector<size_t> &in_shape, const CNodePtr &kernel_node) {
    std::vector<int> pad_list;
    std::vector<int64_t> pad_list_me = GetAttr<std::vector<int64_t>>(kernel_node, "pad_list");
    (void)std::transform(pad_list_me.begin(), pad_list_me.end(), std::back_inserter(pad_list),
                         [](const int64_t &value) { return static_cast<int>(value); });

    n_ = SizeToInt(in_shape[0]);
    c_ = SizeToInt(in_shape[1]);
    old_height_ = SizeToInt(in_shape[2]);
    old_width_ = SizeToInt(in_shape[3]);
    pad_height_ = pad_list[0] + pad_list[1];
    pad_width_ = pad_list[2] + pad_list[3];
    pad_top_ = pad_list[0];
    pad_left_ = pad_list[2];

    // if use_pad_ == true, using zero padding in advance, else using the default cudnn pad.
    if (pad_height_ % 2 == 0 && pad_width_ % 2 == 0) {
      use_pad_ = false;
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensor4dDescriptor(padded_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, n_, c_,
                                                           old_height_ + pad_height_, old_width_ + pad_width_),
                                "cudnnSetTensor4dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetConvolution2dDescriptor(
                                  conv_desc_, use_pad_ ? 0 : pad_top_, use_pad_ ? 0 : pad_left_, stride_[2], stride_[3],
                                  dilation_[2], dilation_[3], CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                                "cudnnSetConvolution2dDescriptor failed");
  }

  void Set4DDesc(const std::vector<size_t> &in_shape, const std::vector<int> &filter_shape,
                 const std::vector<size_t> &output_shape) {
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(in_shape[0]),
                                 SizeToInt(in_shape[1]), SizeToInt(in_shape[2]), SizeToInt(in_shape[3])),
      "cudnnSetTensor4dDescriptor failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetFilter4dDescriptor(filter_desc_, cudnn_data_type_, CUDNN_TENSOR_NCHW, 1,
                                                           SizeToInt(in_shape[1]), filter_shape[0], filter_shape[1]),
                                "cudnnSetFilter4dDescriptor failed");

    auto out_H = output_shape[0] * output_shape[1] * output_shape[2];
    auto out_W = output_shape[3] * output_shape[4] * output_shape[5];
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_,
                                                           SizeToInt(out_H), SizeToInt(out_W), 1, 1),
                                "cudnnSetTensor4dDescriptor failed");
  }

  void SetStrideAndDilation(const CNodePtr &kernel_node) {
    std::vector<int64_t> stride_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "stride");
    std::vector<int64_t> dilation_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "dilation");
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    (void)std::transform(dilation_me.begin(), dilation_me.end(), std::back_inserter(dilation_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (stride_.size() != 4) {
      MS_LOG(EXCEPTION) << "Im2Col's stride must be 4d!";
    }
    if (stride_[0] != 1 || stride_[1] != 1) {
      MS_LOG(EXCEPTION) << "Im2Col's stride only support 1 in N axis and C axis!";
    }
    if (dilation_.size() != 4) {
      MS_LOG(EXCEPTION) << "Im2Col's dilation must be 4d!";
    }
    if (dilation_[0] != 1 || dilation_[1] != 1) {
      MS_LOG(EXCEPTION) << "Im2Col's dilation only support 1 in N axis and C axis!";
    }
  }

  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionFwdAlgo_t conv_algorithm_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorDescriptor_t padded_desc_;
  std::string pad_mode_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  const float pad_value_ = 0.0;
  cudnnDataType_t cudnn_data_type_;
  int old_height_;
  int old_width_;
  int pad_height_;
  int pad_width_;
  int pad_top_;
  int pad_left_;
  int n_;
  int c_;
  std::vector<int> stride_;
  std::vector<int> dilation_;
  bool is_null_input_;
  size_t input_size_;
  size_t output_size_;
  size_t padded_size_;
  size_t workspace_size_;
  bool use_pad_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_IM2COLGPUKERNEL_H_
