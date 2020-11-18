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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_WINOGRAD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_WINOGRAD_H_

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/winograd_transform.h"
#include "nnacl/minimal_filtering_generator.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"

namespace mindspore::kernel {
class ConvolutionWinogradCPUKernel : public ConvolutionBaseCPUKernel {
 public:
  ConvolutionWinogradCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                               const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                               const mindspore::lite::PrimitiveC *primitive, int output_unit)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, primitive),
        output_unit_(output_unit),
        trans_weight_(nullptr) {}
  ~ConvolutionWinogradCPUKernel() override {
    if (trans_weight_ != nullptr) {
      free(trans_weight_);
      trans_weight_ = nullptr;
    }
  };
  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);
  int InitWeightBias();
  int InitTmpBuffer();
  int ConfigInputOutput();
  int WinogradFilterTransform(const float *weight_data, float *matrix_g, float *matrix_gt, int oc_block);

 private:
  void FreeTmpBuffer() {
    if (trans_input_ != nullptr) {
      ctx_->allocator->Free(trans_input_);
      trans_input_ = nullptr;
    }
    if (tmp_data_ != nullptr) {
      ctx_->allocator->Free(tmp_data_);
      tmp_data_ = nullptr;
    }
    if (gemm_out_ != nullptr) {
      ctx_->allocator->Free(gemm_out_);
      gemm_out_ = nullptr;
    }
    if (col_buffer_ != nullptr) {
      ctx_->allocator->Free(col_buffer_);
      col_buffer_ = nullptr;
    }
  }
  int kernel_unit_;
  int input_unit_;
  int output_unit_;
  float *tmp_data_ = nullptr;
  float *trans_input_ = nullptr;
  float *gemm_out_ = nullptr;
  float *col_buffer_ = nullptr;
  float *trans_weight_ = nullptr;
  TmpBufferAddress tmp_buffer_address_list_[4];
  InputTransFunc in_func_;
  OutputTransFunc out_func_;
};

}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_WINOGRAD_H_
