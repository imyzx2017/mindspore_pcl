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

#include "src/runtime/kernel/arm/int8/convolution_int8.h"
#include "include/errorcode.h"
#include "nnacl/int8/conv_int8.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"
#include "src/runtime/kernel/arm/int8/convolution_1x1_int8.h"
#include "src/runtime/kernel/arm/int8/convolution_3x3_int8.h"
#include "src/runtime/runtime_api.h"
#ifdef ENABLE_ARM64
#include "src/runtime/kernel/arm/int8/opt_op_handler.h"
#endif

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2D;

namespace mindspore::kernel {
void ConvolutionInt8CPUKernel::CheckSupportOptimize() {
  tile_num_ = 8;
#ifdef ENABLE_ARM32
  tile_num_ = 4;
  support_optimize_ = false;
#endif

#ifdef ENABLE_ARM64
  if (mindspore::lite::IsSupportSDot()) {
    matmul_func_ = MatMulRInt8_optimize_handler;
    support_optimize_ = true;
  } else {
    tile_num_ = 4;
    support_optimize_ = false;
  }
#endif
  conv_param_->tile_num_ = tile_num_;
}

int ConvolutionInt8CPUKernel::InitWeightBias() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = filter_tensor->Channel();
  auto output_channel = filter_tensor->Batch();
  int kernel_h = filter_tensor->Height();
  int kernel_w = filter_tensor->Width();
  int kernel_plane = kernel_h * kernel_w;
  conv_param_->input_channel_ = input_channel;
  conv_param_->output_channel_ = output_channel;
  int up_round_deep;
  int up_round_oc;
#ifdef ENABLE_ARM32
  up_round_oc = UP_ROUND(output_channel, C2NUM);
  up_round_deep = UP_ROUND(kernel_plane * input_channel, C16NUM);
#else
  if (support_optimize_) {
    up_round_oc = UP_ROUND(output_channel, C8NUM);
    up_round_deep = UP_ROUND(kernel_plane * input_channel, C4NUM);
  } else {
    up_round_oc = UP_ROUND(output_channel, C4NUM);
    up_round_deep = UP_ROUND(kernel_plane * input_channel, C16NUM);
  }
#endif
  int pack_weight_size = up_round_oc * up_round_deep;
  size_t bias_size = up_round_oc * sizeof(int32_t);
  int32_t input_zp = conv_param_->conv_quant_arg_.input_quant_args_[0].zp_;

  // init weight
  auto origin_weight = reinterpret_cast<int8_t *>(in_tensors_.at(kWeightIndex)->MutableData());
  packed_weight_ = reinterpret_cast<int8_t *>(malloc(pack_weight_size));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed_weight_ failed.";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, pack_weight_size);
#ifdef ENABLE_ARM32
  RowMajor2Row2x16MajorInt8(origin_weight, packed_weight_, output_channel, input_channel * kernel_plane);
#else
  if (support_optimize_) {
    RowMajor2Row8x4MajorInt8(origin_weight, packed_weight_, output_channel, input_channel * kernel_plane);
  } else {
    RowMajor2Row16x4MajorInt8(origin_weight, packed_weight_, output_channel, input_channel * kernel_plane);
  }
#endif

  // init bias
  bias_data_ = reinterpret_cast<int32_t *>(malloc(bias_size));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias_data_ failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, bias_size);
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<int32_t *>(in_tensors_.at(kBiasIndex)->MutableData());
    memcpy(bias_data_, ori_bias, output_channel * sizeof(int32_t));
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  auto *bias_data = reinterpret_cast<int32_t *>(bias_data_);
  bool filter_peroc = conv_quant_arg_->per_channel_ & FILTER_PER_CHANNEL;
  if (filter_peroc) {
    filter_zp_ptr_ = reinterpret_cast<int32_t *>(malloc(output_channel * sizeof(int32_t)));
  }
  for (int oc = 0; oc < output_channel; oc++) {
    int32_t filter_zp = conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_;
    if (filter_peroc) {
      filter_zp = conv_param_->conv_quant_arg_.filter_quant_args_[oc].zp_;
      filter_zp_ptr_[oc] = filter_zp;
    }
    int32_t weight_sum_value = up_round_deep * filter_zp;
    for (int i = 0; i < kernel_plane * input_channel; i++) {
      weight_sum_value += origin_weight[oc * kernel_plane * input_channel + i] - filter_zp;
    }
    bias_data[oc] += filter_zp * input_zp * up_round_deep - weight_sum_value * input_zp;
  }

  size_t input_sum_size;
  if (conv_quant_arg_->per_channel_ & FILTER_PER_CHANNEL) {
    input_sum_size = up_round_oc * tile_num_ * thread_count_ * sizeof(int32_t);
  } else {
    input_sum_size = tile_num_ * thread_count_ * sizeof(int32_t);
  }
  input_sum_ = reinterpret_cast<int32_t *>(malloc(input_sum_size));
  if (input_sum_ == nullptr) {
    MS_LOG(ERROR) << "malloc input_sum_ failed.";
    return RET_ERROR;
  }
  memset(input_sum_, 0, input_sum_size);
  return RET_OK;
}

int ConvolutionInt8CPUKernel::InitTmpBuffer() {
  MS_ASSERT(ctx_->allocator != nullptr);
  int kernel_plane = conv_param_->kernel_h_ * conv_param_->kernel_w_;
  int tmp_size;
  if (support_optimize_) {
    tmp_size = UP_ROUND(kernel_plane * conv_param_->input_channel_, C4NUM);
  } else {
    tmp_size = UP_ROUND(kernel_plane * conv_param_->input_channel_, C16NUM);
  }
  matmul_packed_input_ = reinterpret_cast<int8_t *>(
    ctx_->allocator->Malloc(thread_count_ * tile_num_ * kernel_plane * conv_param_->input_channel_));
  if (matmul_packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc matmul_packed_input_ failed.";
    return RET_ERROR;
  }
  packed_input_ = reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(tmp_size * thread_count_ * tile_num_));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed_input_ failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionInt8CPUKernel::Init() {
  CheckSupportOptimize();
  auto ret = SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set quant param failed.";
    return ret;
  }

  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Initialization for optimized int8 conv failed.";
    return RET_ERROR;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionInt8CPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::CheckResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }

  ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionInt8CPUKernel::RunImpl(int task_id) {
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto ori_input_data = reinterpret_cast<int8_t *>(input_tensor->MutableData());
  auto output_addr = reinterpret_cast<int8_t *>(out_tensors_.at(kOutputIndex)->MutableData());
  ConvInt8(ori_input_data, packed_input_, matmul_packed_input_, packed_weight_, reinterpret_cast<int32_t *>(bias_data_),
           output_addr, filter_zp_ptr_, input_sum_, task_id, conv_param_, matmul_func_, support_optimize_);
  return RET_OK;
}

int ConvolutionInt8Impl(void *cdata, int task_id) {
  auto conv = reinterpret_cast<ConvolutionInt8CPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution Int8 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionInt8CPUKernel::Run() {
  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }

  int error_code = ParallelLaunch(this->context_->thread_pool_, ConvolutionInt8Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv int8 error error_code[" << error_code << "]";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  FreeTmpBuffer();
  return RET_OK;
}

kernel::LiteKernel *CpuConvInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                             const InnerContext *ctx, const kernel::KernelKey &desc,
                                             const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2D);
  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int stride_h = conv_param->stride_h_;
  int stride_w = conv_param->stride_w_;
  int dilation_h = conv_param->dilation_h_;
  int dilation_w = conv_param->dilation_w_;
  kernel::LiteKernel *kernel;
  if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
#ifdef ENABLE_ARM64
    if (mindspore::lite::IsSupportSDot()) {
      kernel = new (std::nothrow) kernel::ConvolutionInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
    } else {
      kernel = new (std::nothrow) kernel::Convolution3x3Int8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
    }
#else
    kernel = new (std::nothrow) kernel::Convolution3x3Int8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
#endif
  } else if (kernel_h == 1 && kernel_w == 1) {
    kernel = new (std::nothrow) kernel::Convolution1x1Int8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  } else {
    kernel = new (std::nothrow) kernel::ConvolutionInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Conv2D, CpuConvInt8KernelCreator)
}  // namespace mindspore::kernel
