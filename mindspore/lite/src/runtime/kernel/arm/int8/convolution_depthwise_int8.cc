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

#include "src/runtime/kernel/arm/int8/convolution_depthwise_int8.h"
#include "src/runtime/kernel/arm/int8/convolution_depthwise_3x3_int8.h"
#include "src/runtime/kernel/arm/int8/convolution_depthwise_slidewindow_int8.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/int8/conv_depthwise_int8.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DepthwiseConv2D;

namespace mindspore::kernel {
ConvolutionDepthwiseInt8CPUKernel::~ConvolutionDepthwiseInt8CPUKernel() {
  if (packed_weight_ != nullptr) {
    free(packed_weight_);
    packed_weight_ = nullptr;
  }
  FreeQuantParam();
}

int ConvolutionDepthwiseInt8CPUKernel::InitWeightBias() {
  // init weight, int8 -> int16
  auto weight_tensor = in_tensors_[kWeightIndex];
  auto origin_weight = reinterpret_cast<int8_t *>(weight_tensor->MutableData());
  int channel = weight_tensor->Batch();
  int pack_weight_size = channel * weight_tensor->Height() * weight_tensor->Width();
  auto tmp_weight = reinterpret_cast<int8_t *>(malloc(pack_weight_size * sizeof(int8_t)));
  if (tmp_weight == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  PackNCHWToNHWCInt8(origin_weight, tmp_weight, 1, weight_tensor->Height() * weight_tensor->Width(),
                     weight_tensor->Batch());

  packed_weight_ = reinterpret_cast<int16_t *>(malloc(pack_weight_size * sizeof(int16_t)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }

  bool filter_per_channel = conv_param_->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL;
  if (filter_per_channel) {
    for (int i = 0; i < weight_tensor->Height() * weight_tensor->Width(); i++) {
      for (int c = 0; c < channel; c++) {
        int weight_zp = conv_param_->conv_quant_arg_.filter_quant_args_[c].zp_;
        packed_weight_[i * channel + c] = (int16_t)(tmp_weight[i * channel + c] - weight_zp);
      }
    }
  } else {
    int weight_zp = conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_;
    for (int i = 0; i < weight_tensor->ElementsNum(); i++) {
      packed_weight_[i] = (int16_t)(tmp_weight[i] - weight_zp);
    }
  }
  free(tmp_weight);

  bias_data_ = reinterpret_cast<int32_t *>(malloc(channel * sizeof(int32_t)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, channel * sizeof(int32_t));
  if (in_tensors_.size() == kInputSize2) {
    auto bias_tensor = in_tensors_.at(kBiasIndex);
    auto ori_bias = reinterpret_cast<int32_t *>(bias_tensor->MutableData());
    memcpy(bias_data_, ori_bias, bias_tensor->ElementsNum() * sizeof(int32_t));
  }

  return RET_OK;
}

int ConvolutionDepthwiseInt8CPUKernel::Init() {
  auto ret = ConvolutionBaseCPUKernel::SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set quant param failed.";
    return ret;
  }
  conv_param_->thread_num_ = MSMIN(thread_count_, conv_param_->output_h_);
  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Depthwise int8 InitWeightBias error!";
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionDepthwiseInt8CPUKernel::ReSize() {
  ConvolutionBaseCPUKernel::Init();
  return RET_OK;
}

int ConvolutionDepthwiseInt8CPUKernel::Execute(int task_id) {
  auto buffer = row_buffer_ + conv_param_->output_w_ * conv_param_->output_channel_ * task_id;
  ConvDwInt8(output_ptr_, buffer, input_ptr_, packed_weight_, reinterpret_cast<int32_t *>(bias_data_), conv_param_,
             task_id);
  return RET_OK;
}

int ConvDwInt8Run(void *cdata, int task_id) {
  auto conv_dw_int8 = reinterpret_cast<ConvolutionDepthwiseInt8CPUKernel *>(cdata);
  auto ret = conv_dw_int8->Execute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwiseInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseInt8CPUKernel::InitBuffer() {
  int output_row_size = conv_param_->thread_num_ * conv_param_->output_w_ * conv_param_->output_channel_;
  row_buffer_ = reinterpret_cast<int32_t *>(context_->allocator->Malloc(output_row_size * sizeof(int)));
  if (row_buffer_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseInt8CPUKernel::Run() {
  auto ret = InitBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Depthwise int8 ReSize error!";
    return ret;
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  input_ptr_ = reinterpret_cast<int8_t *>(input_tensor->MutableData());

  auto output_tensor = out_tensors_.at(kOutputIndex);
  output_ptr_ = reinterpret_cast<int8_t *>(output_tensor->MutableData());

  ret = ParallelLaunch(this->context_->thread_pool_, ConvDwInt8Run, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDwInt8Run error: error_code[" << ret << "]";
    return RET_ERROR;
  }

  context_->allocator->Free(row_buffer_);
  return RET_OK;
}

kernel::LiteKernel *CpuConvDwInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const InnerContext *ctx, const kernel::KernelKey &desc,
                                               const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_DepthwiseConv2D);
  kernel::LiteKernel *kernel = nullptr;
  auto act_quant_size =
    MSMAX(inputs[kInputIndex]->GetQuantParams().size(), outputs[kOutputIndex]->GetQuantParams().size());
  if (act_quant_size == 1) {  // per tensor
    auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
    if (primitive != nullptr && primitive->GetInferFlag()) {
      conv_param->input_h_ = inputs[kInputIndex]->Height();
      conv_param->input_w_ = inputs[kInputIndex]->Width();
      conv_param->input_channel_ = inputs[kInputIndex]->Channel();
      conv_param->output_h_ = outputs[kOutputIndex]->Height();
      conv_param->output_w_ = outputs[kOutputIndex]->Width();
    }
    if (CheckConvDwUse3X3(conv_param) && conv_param->input_channel_ % C8NUM == 0) {
#ifdef ENABLE_ARM64
      kernel =
        new (std::nothrow) kernel::ConvolutionDepthwise3x3Int8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
#endif
    }
    if (kernel == nullptr) {
      kernel =
        new (std::nothrow) kernel::ConvolutionDepthwiseInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
    }
  } else {  // per channel
    kernel =
      new (std::nothrow) kernel::ConvolutionDepthwiseSWInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
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

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_DepthwiseConv2D, CpuConvDwInt8KernelCreator)
}  // namespace mindspore::kernel
