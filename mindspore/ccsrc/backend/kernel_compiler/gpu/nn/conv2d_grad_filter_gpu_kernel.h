/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV2D_GRAD_FILTER_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV2D_GRAD_FILTER_GPU_KERNEL_H_

#include <algorithm>
#include <string>
#include <vector>

#include "backend/kernel_compiler/gpu/cuda_impl/pad_impl.cuh"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class ConvGradFilterGpuBkwKernel : public GpuKernel {
 public:
  ConvGradFilterGpuBkwKernel()
      : cudnn_handle_(nullptr),
        dw_desc_(nullptr),
        conv_desc_(nullptr),
        dy_desc_(nullptr),
        x_desc_(nullptr),
        padded_descriptor_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        compute_format_(CUDNN_TENSOR_NCHW),
        old_height_(0),
        old_width_(0),
        pad_height_(0),
        pad_width_(0),
        pad_top_(0),
        pad_left_(0),
        n_(0),
        c_(0),
        group_(1),
        is_null_input_(false),
        input_size_(0),
        dy_size_(0),
        output_size_(0),
        padded_size_(0),
        workspace_size_(0),
        use_pad_(true) {}
  ~ConvGradFilterGpuBkwKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *dy = GetDeviceAddress<T>(inputs, 0);
    T *x = GetDeviceAddress<T>(inputs, 1);
    T *dw = GetDeviceAddress<T>(outputs, 0);
    T *work_space = nullptr;
    if (workspace_size_ != 0) {
      work_space = GetDeviceAddress<T>(workspace, 0);
    }

    const float alpha = 1;
    const float beta = 0;

    if ((pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) && use_pad_) {
      T *padded = GetDeviceAddress<T>(workspace, 1);
      if (data_format_ == kOpFormat_NHWC) {
        CalPadNHWC(padded_size_ / sizeof(T), x, n_, old_height_, old_width_, c_, old_height_ + pad_height_,
                   old_width_ + pad_width_, pad_top_, pad_left_, pad_value_, padded,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
      } else {
        CalPad(padded_size_ / sizeof(T), x, n_, c_, old_height_, old_width_, old_height_ + pad_height_,
               old_width_ + pad_width_, pad_top_, pad_left_, pad_value_, padded,
               reinterpret_cast<cudaStream_t>(stream_ptr));
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnConvolutionBackwardFilter(cudnn_handle_, &alpha, padded_descriptor_, padded, dy_desc_, dy, conv_desc_,
                                       algo_, work_space, workspace_size_, &beta, dw_desc_, dw),
        "ConvolutionBackwardFilter failed");
      return true;
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnConvolutionBackwardFilter(cudnn_handle_, &alpha, x_desc_, x, dy_desc_, dy, conv_desc_, algo_, work_space,
                                     workspace_size_, &beta, dw_desc_, dw),
      "ConvolutionBackwardFilter failed");
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    if (!CheckParam(kernel_node)) {
      return false;
    }
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    auto dy_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    auto in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
    is_null_input_ = CHECK_NULL_INPUT(dy_shape) || CHECK_NULL_INPUT(in_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "ConvGradFilterGpuBkwKernel input is null.";
      InitSizeLists();
      return true;
    }
    data_format_ = AnfAlgo::GetInputFormat(kernel_node, 0);
    format_attr_ = GetAttr<std::string>(kernel_node, "data_format");
    if (format_attr_ == kOpFormat_NHWC) {
      data_format_ = kOpFormat_NHWC;
    }
    std::vector<size_t> filter_shape;
    GetFilterShape(kernel_node, &filter_shape);
    if (data_format_ == kOpFormat_NHWC) {
      compute_format_ = CUDNN_TENSOR_NHWC;
    }
    SetNCHW(in_shape, &n_, &c_, &old_height_, &old_width_, data_format_);
    Set4DDesc(dy_shape, filter_shape, in_shape);
    group_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "group"));
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetConvolutionGroupCount(conv_desc_, group_), "cudnnSetConvGroupCount failed");

    std::vector<int> pad_list;
    std::vector<int64_t> pad_list_me = GetAttr<std::vector<int64_t>>(kernel_node, "pad_list");
    (void)std::transform(pad_list_me.begin(), pad_list_me.end(), std::back_inserter(pad_list),
                         [](const int64_t &value) { return static_cast<int>(value); });
    pad_height_ = pad_list[0];
    pad_width_ = pad_list[2];
    auto symmetry_pad = (pad_height_ == pad_list[1]) && (pad_width_ == pad_list[3]);
    pad_mode_ = GetAttr<std::string>(kernel_node, "pad_mode");
    SetStrideAndDilation(kernel_node);
    cudnnTensorDescriptor_t x_desc_real = nullptr;
    int padA[2];
    int strideA[2] = {stride_[0], stride_[1]};
    int dilaA[2] = {dilation_[2], dilation_[3]};
    if (pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase || !symmetry_pad) {
      pad_height_ = pad_list[0] + pad_list[1];
      pad_width_ = pad_list[2] + pad_list[3];
      pad_top_ = pad_list[0];
      pad_left_ = pad_list[2];
      if (pad_height_ % 2 == 0 && pad_width_ % 2 == 0) {
        use_pad_ = false;
      }
      int dimA[4];
      int strideApadded[4];
      if (data_format_ == kOpFormat_NCHW || data_format_ == kOpFormat_DEFAULT) {
        auto padded_shape = {IntToSize(n_), IntToSize(c_), IntToSize(old_height_ + pad_height_),
                             IntToSize(old_width_ + pad_width_)};
        SetDimA(padded_shape, dimA, 4, data_format_);
        SetStrideA(padded_shape, strideApadded, 4, data_format_);
      } else if (data_format_ == kOpFormat_NHWC) {
        auto padded_shape = {IntToSize(n_), IntToSize(old_height_ + pad_height_), IntToSize(old_width_ + pad_width_),
                             IntToSize(c_)};
        SetDimA(padded_shape, dimA, 4, data_format_);
        SetStrideA(padded_shape, strideApadded, 4, data_format_);
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnSetTensorNdDescriptor(padded_descriptor_, cudnn_data_type_, 4, dimA, strideApadded),
        "cudnnSetTensor4dDescriptor failed");
      if (use_pad_) {
        padA[0] = 0;
        padA[1] = 0;
      } else {
        padA[0] = pad_top_;
        padA[1] = pad_left_;
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnSetConvolutionNdDescriptor(conv_desc_, 2, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
        "cudnnSetConvolutionNdDescriptor failed");
      x_desc_real = use_pad_ ? padded_descriptor_ : x_desc_;
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_height_ = 0;
        pad_width_ = 0;
      }
      padA[0] = pad_height_;
      padA[1] = pad_width_;
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnSetConvolutionNdDescriptor(conv_desc_, 2, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
        "cudnnSetConvolution2dDescriptor failed");
      x_desc_real = x_desc_;
    }
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH),
                                  "cudnnSetConvolutionMathType failed.")
    }
    SelectAlgorithm(x_desc_real);
    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc_),
                               "cudnnDestroyConvolutionDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyFilterDescriptor(dw_desc_), "cudnnDestroyFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(padded_descriptor_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(dy_desc_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(x_desc_), "cudnnDestroyTensorDescriptor failed");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&x_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&dy_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&padded_descriptor_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateFilterDescriptor(&dw_desc_), "cudnnCreateFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateConvolutionDescriptor(&conv_desc_),
                                "cudnnCreateConvolutionDescriptor failed");
  }
  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(dy_desc_, reinterpret_cast<size_t *>(&dy_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(x_desc_, reinterpret_cast<size_t *>(&input_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetFilterSizeInBytes(dw_desc_, reinterpret_cast<size_t *>(&output_size_)),
                                  "cudnnGetFilterSizeInBytes failed");
    }
    input_size_list_.push_back(dy_size_);
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);

    if ((pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) && use_pad_ && !is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnGetTensorSizeInBytes(padded_descriptor_, reinterpret_cast<size_t *>(&padded_size_)),
        "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle_, padded_descriptor_, dy_desc_, conv_desc_,
                                                       dw_desc_, algo_, reinterpret_cast<size_t *>(&workspace_size_)),
        "cudnnGetConvolutionBackwardFilterWorkspaceSize failed");
      workspace_size_list_.push_back(padded_size_);
    } else {
      if (!is_null_input_) {
        CHECK_CUDNN_RET_WITH_EXCEPT(
          cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle_, x_desc_, dy_desc_, conv_desc_, dw_desc_, algo_,
                                                         reinterpret_cast<size_t *>(&workspace_size_)),
          "cudnnGetConvolutionBackwardFilterWorkspaceSize failed");
      }
    }
    (void)workspace_size_list_.insert(workspace_size_list_.begin(), workspace_size_);
  }

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but ConvGradFilter needs 2 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but ConvGradFilter needs 1 output.";
      return false;
    }
    return true;
  }
  void SelectAlgorithm(cudnnTensorDescriptor_t x_desc_real) {
    if (group_ > 1 || CUDNN_MAJOR < 7) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle_, x_desc_real, dy_desc_, conv_desc_, dw_desc_,
                                                   CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, 0, &algo_),
        "GetConvolutionBackwardFilterAlgorithm failed");
    } else {
      constexpr int requested_algo_count = 1;
      int returned_algo_count;
      cudnnConvolutionBwdFilterAlgoPerf_t perf_results;
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnn_handle_, x_desc_real, dy_desc_, conv_desc_, dw_desc_,
                                                      requested_algo_count, &returned_algo_count, &perf_results),
        "GetConvolutionBackwardFilterAlgorithm failed");
      algo_ = perf_results.algo;
    }
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    }
  }
  void GetFilterShape(const CNodePtr &kernel_node, std::vector<size_t> *filter_shape) {
    auto shp_tuple_x = AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("filter_sizes")->cast<ValueTuplePtr>()->value();
    (void)std::transform(std::begin(shp_tuple_x), std::end(shp_tuple_x), std::back_inserter(*filter_shape),
                         [](const ValuePtr &e) -> size_t { return static_cast<int>(e->cast<Int64ImmPtr>()->value()); });
  }
  void Set4DDesc(const std::vector<size_t> &dy_shape, const std::vector<size_t> &filter_shape,
                 const std::vector<size_t> &in_shape) {
    const int nbDims = 4;
    int dimA[4];
    int strideAin[4];
    int dimAdy[4];
    int strideAdy[4];
    SetDimA(in_shape, dimA, 4, data_format_);
    SetStrideA(in_shape, strideAin, 4, data_format_);
    SetDimA(dy_shape, dimAdy, 4, data_format_);
    SetStrideA(dy_shape, strideAdy, 4, data_format_);
    // filter shape relued by format_attr_. In native mode it's OHWI. In transpose mode it's OIHW.
    int filterDimA[4];
    SetDimA(filter_shape, filterDimA, 4, format_attr_);
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensorNdDescriptor(dy_desc_, cudnn_data_type_, nbDims, dimAdy, strideAdy),
                                "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetFilterNdDescriptor(dw_desc_, cudnn_data_type_, compute_format_, nbDims, filterDimA),
      "cudnnSetFilterNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensorNdDescriptor(x_desc_, cudnn_data_type_, nbDims, dimA, strideAin),
                                "cudnnSetTensorNdDescriptor failed");
  }
  void SetStrideAndDilation(const CNodePtr &kernel_node) {
    std::vector<int64_t> stride_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "stride");
    std::vector<int64_t> dilation_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "dilation");
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    (void)std::transform(dilation_me.begin(), dilation_me.end(), std::back_inserter(dilation_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (stride_.size() != 2) {
      MS_LOG(EXCEPTION) << "ConvGradFilterGpuBkwKernel's stride must be 2d!";
    }
    if (dilation_.size() != 4) {
      MS_LOG(EXCEPTION) << "ConvGradFilterGpuBkwKernel's dilation must be 4d!";
    }
    if (dilation_[0] != 1 || dilation_[1] != 1) {
      MS_LOG(EXCEPTION) << "ConvGradFilterGpuBkwKernel dilation only support 1 in N axis and C axis!";
    }
  }
  cudnnHandle_t cudnn_handle_;
  cudnnFilterDescriptor_t dw_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorDescriptor_t dy_desc_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t padded_descriptor_;
  cudnnConvolutionBwdFilterAlgo_t algo_;
  std::string pad_mode_;
  std::string data_format_ = kOpFormat_NCHW;
  std::string format_attr_ = kOpFormat_NCHW;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  const float pad_value_ = 0.0;
  cudnnDataType_t cudnn_data_type_;
  cudnnTensorFormat_t compute_format_;
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
  int group_;
  bool is_null_input_;
  size_t input_size_;
  size_t dy_size_;
  size_t output_size_;
  size_t padded_size_;
  size_t workspace_size_;
  bool use_pad_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDePORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV2D_GRAD_FILTER_GPU_KERNEL_H_
