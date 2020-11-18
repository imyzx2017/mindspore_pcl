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

#include "src/runtime/kernel/arm/fp16/deconvolution_depthwise_fp16.h"
#include "nnacl/fp16/pack_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "src/runtime/kernel/arm/base/dequant.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DeDepthwiseConv2D;

namespace mindspore::kernel {
DeconvolutionDepthwiseFp16CPUKernel::~DeconvolutionDepthwiseFp16CPUKernel() {
  if (sliding_ != nullptr) {
    delete sliding_;
    sliding_ = nullptr;
  }
  if (packed_weight_ != nullptr) {
    delete packed_weight_;
    packed_weight_ = nullptr;
  }
}

int DeconvolutionDepthwiseFp16CPUKernel::InitSlideParam() {
  conv_param_->input_batch_ = out_tensors_.front()->shape().at(kNHWC_N);
  conv_param_->input_h_ = out_tensors_.front()->shape().at(kNHWC_H);
  conv_param_->input_w_ = out_tensors_.front()->shape().at(kNHWC_W);
  conv_param_->input_channel_ = out_tensors_.front()->shape().at(kNHWC_C);
  conv_param_->output_batch_ = in_tensors_.front()->shape().at(kNHWC_N);
  conv_param_->output_h_ = in_tensors_.front()->shape().at(kNHWC_H);
  conv_param_->output_w_ = in_tensors_.front()->shape().at(kNHWC_W);
  conv_param_->output_channel_ = in_tensors_.front()->shape().at(kNHWC_C);
  InitSlidingParamConvDw(sliding_, conv_param_, C8NUM);
  return RET_OK;
}

int DeconvolutionDepthwiseFp16CPUKernel::InitBuffer() {
  if (conv_param_->input_channel_ % C8NUM != 0) {
    need_align_ = true;
    int C8 = UP_DIV(conv_param_->input_channel_, C8NUM);
    int pack_input_size = conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * C8NUM * C8;
    packed_input_ = reinterpret_cast<float16_t *>(context_->allocator->Malloc(pack_input_size * sizeof(float16_t)));
    if (packed_input_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }

    int pack_output_size = conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ * C8NUM * C8;
    packed_output_ = reinterpret_cast<float16_t *>(context_->allocator->Malloc(pack_output_size * sizeof(float16_t)));
    if (packed_output_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
    memset(packed_output_, 0, pack_output_size * sizeof(float16_t));
  }
  return RET_OK;
}

int DeconvolutionDepthwiseFp16CPUKernel::InitWeightBias() {
  // init weight: o, h, w, i; o == group, i == 1
  auto weight_tensor = in_tensors_[kWeightIndex];
  int OC8 = UP_DIV(weight_tensor->Batch(), C8NUM);
  auto origin_weight = reinterpret_cast<float *>(weight_tensor->MutableData());
  int pack_weight_size = C8NUM * OC8 * weight_tensor->Height() * weight_tensor->Width();

  packed_weight_ = reinterpret_cast<float16_t *>(malloc(pack_weight_size * sizeof(float16_t)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  PackNCHWFp32ToNC8HW8Fp16(origin_weight, packed_weight_, 1, weight_tensor->Height() * weight_tensor->Width(),
                           weight_tensor->Batch());

  bias_data_ = reinterpret_cast<float16_t *>(malloc(C8NUM * OC8 * sizeof(float16_t)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, C8NUM * OC8 * sizeof(float16_t));
  if (in_tensors_.size() == kInputSize2) {
    auto bias_tensor = in_tensors_.at(kBiasIndex);
    auto ori_bias = reinterpret_cast<float *>(bias_tensor->MutableData());
    for (int i = 0; i < bias_tensor->ElementsNum(); i++) {
      reinterpret_cast<float *>(bias_data_)[i] = (float16_t)ori_bias[i];
    }
  }

  conv_param_->thread_num_ = MSMIN(thread_count_, OC8);
  return RET_OK;
}

int DeconvolutionDepthwiseFp16CPUKernel::Init() {
  sliding_ = new (std::nothrow) SlidingWindowParam;
  if (sliding_ == nullptr) {
    MS_LOG(ERROR) << "new SlidingWindowParam fail!";
    return RET_ERROR;
  }

  auto ret = InitWeightBias();
  if (ret != 0) {
    MS_LOG(ERROR) << "Deconvolution depthwise fp16 InitWeightBias failed.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int DeconvolutionDepthwiseFp16CPUKernel::ReSize() {
  InitSlideParam();
  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  return RET_OK;
}

int DeconvolutionDepthwiseFp16CPUKernel::Execute(int task_id) {
  DeconvDwC8Fp16(packed_output_, packed_input_, packed_weight_, reinterpret_cast<float16_t *>(bias_data_), conv_param_,
                 sliding_, task_id);
  return RET_OK;
}

static int DeconvDwFp16Run(void *cdata, int task_id) {
  auto deconv_dw_fp16 = reinterpret_cast<DeconvolutionDepthwiseFp16CPUKernel *>(cdata);
  auto ret = deconv_dw_fp16->Execute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DeconvolutionDepthwiseFp16Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeconvolutionDepthwiseFp16CPUKernel::Run() {
  if (conv_param_->input_channel_ != conv_param_->output_channel_) {
    MS_LOG(ERROR) << "Only support input channel equals output channel.";
    return RET_ERROR;
  }
  auto ret = InitBuffer();
  if (ret != 0) {
    MS_LOG(ERROR) << "Deconvolution depthwise fp16 InitBuffer failed.";
    return RET_ERROR;
  }

  ret = ConvolutionBaseFP16CPUKernel::GetExecuteTensor();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get Execute tensor failed.";
    return ret;
  }
  if (need_align_) {
    PackNHWCToNHWC8Fp16(execute_input_, packed_input_, conv_param_->input_batch_,
                        conv_param_->input_h_ * conv_param_->input_w_, conv_param_->input_channel_);
  } else {
    packed_input_ = execute_input_;
  }

  if (!need_align_) {
    memset(execute_output_, 0, out_tensors_.at(kOutputIndex)->ElementsNum() * sizeof(float16_t));
    packed_output_ = execute_output_;
  }
  ret = ParallelLaunch(this->context_->thread_pool_, DeconvDwFp16Run, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DeconvDwFp16Run error: error_code[" << ret << "]";
    return RET_ERROR;
  }

  if (need_align_) {
    PackNHWC8ToNHWCFp16(packed_output_, execute_output_, conv_param_->output_batch_,
                        conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_);
    context_->allocator->Free(packed_input_);
    context_->allocator->Free(packed_output_);
  }
  ConvolutionBaseFP16CPUKernel::IfCastOutput();
  ConvolutionBaseFP16CPUKernel::FreeTmpBuffer();
  return RET_OK;
}

kernel::LiteKernel *CpuDeconvDwFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                 const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                 const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                 const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_DeDepthwiseConv2D);

  auto *weight_tensor = inputs.at(kWeightIndex);
  auto *restore_data = weight_tensor->data_c();
  auto restore_type = weight_tensor->data_type();
  auto dequant_flag = !weight_tensor->GetQuantParams().empty() && weight_tensor->GetQuantParams().front().inited &&
                      restore_data != nullptr;
  if (dequant_flag) {
    auto *dequant_weight = kernel::DequantUtil::DequantWeight(weight_tensor);
    if (dequant_weight == nullptr) {
      MS_LOG(ERROR) << "dequant data is nullptr.";
      free(opParameter);
      return nullptr;
    }
    weight_tensor->set_data_type(kNumberTypeFloat32);
    weight_tensor->set_data(dequant_weight);
  }

  auto kernel = new (std::nothrow) DeconvolutionDepthwiseFp16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    if (dequant_flag) {
      weight_tensor->FreeData();
      weight_tensor->set_data(restore_data);
      weight_tensor->set_data_type(restore_type);
    }
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    if (dequant_flag) {
      weight_tensor->FreeData();
      weight_tensor->set_data(restore_data);
      weight_tensor->set_data_type(restore_type);
    }
    return nullptr;
  }
  if (dequant_flag) {
    weight_tensor->FreeData();
    weight_tensor->set_data(restore_data);
    weight_tensor->set_data_type(restore_type);
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_DeDepthwiseConv2D, CpuDeconvDwFp16KernelCreator)
}  // namespace mindspore::kernel
