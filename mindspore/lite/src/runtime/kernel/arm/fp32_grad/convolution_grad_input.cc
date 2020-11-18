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

#include "src/runtime/kernel/arm/fp32_grad/convolution_grad_input.h"
#include "src/kernel_registry.h"
#include "nnacl/pack.h"
#include "nnacl/fp32_grad/pack_ext.h"
#include "nnacl/fp32_grad/gemm.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2DGradInput;
using mindspore::schema::PrimitiveType_GroupConv2DGradInput;

namespace mindspore::kernel {
int ConvolutionGradInputCPUKernel::Init() {
  auto *dy_tensor = in_tensors_.at(kInputIndex);
  MS_ASSERT(dy_tensor != nullptr);
  auto *weight_tensor = in_tensors_.at(kWeightIndex);
  MS_ASSERT(weight_tensor != nullptr);
  auto *dx_tensor = out_tensors_.at(kOutputIndex);
  MS_ASSERT(dx_tensor != nullptr);

  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter_);
  conv_param->output_batch_ = dx_tensor->shape()[(kNHWC_N)];
  conv_param->input_batch_ = dy_tensor->shape()[(kNHWC_N)];

  conv_param->input_h_ = dx_tensor->shape()[(kNHWC_H)];
  conv_param->input_w_ = dx_tensor->shape()[(kNHWC_W)];

  // assume OutCh|kh|kw|In
  conv_param->input_channel_ = dx_tensor->shape()[(kNHWC_C)];
  conv_param->output_channel_ = weight_tensor->shape()[(kNHWC_N)];

  conv_param->output_h_ = dy_tensor->shape()[kNHWC_H];
  conv_param->output_w_ = dy_tensor->shape()[kNHWC_W];
  ws_size = chunk * conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_ / conv_param->group_;

  int n = conv_param->kernel_w_ * conv_param->kernel_h_ * conv_param->input_channel_ / conv_param->group_;
  int k = conv_param->output_channel_ / conv_param->group_;

  size_t mat_alloc = MatSizeTotal(chunk, n, k, 0);

  SetWorkspaceSize((ws_size + mat_alloc) * sizeof(float));
  return RET_OK;
}

int ConvolutionGradInputCPUKernel::ReSize() { return RET_OK; }

int ConvolutionGradInputCPUKernel::Execute(int task_id) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter_);
  auto *input_dy = in_tensors_.at(0);
  auto *input_w = in_tensors_.at(1);
  auto *out_dx = out_tensors_.at(0);

  auto dy_addr = reinterpret_cast<float *>(input_dy->MutableData());
  auto w_addr = reinterpret_cast<float *>(input_w->MutableData());
  auto dx_addr = reinterpret_cast<float *>(out_dx->MutableData());

  int i, j;
  int nweights = input_w->ElementsNum();
  int in_ch = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int k_h = conv_param->kernel_h_;  // out_dw->shape()[1];
  int k_w = conv_param->kernel_w_;  // out_dw->shape()[2];
  int batch = conv_param->output_batch_;
  int out_ch = conv_param->output_channel_;
  int groups = conv_param->group_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;

  int m = out_h * out_w;
  int n = k_w * k_h * in_ch / groups;
  int k = out_ch / groups;
  float *workspace = reinterpret_cast<float *>(GetWorkspace());
  float *mat_workspace = workspace + ws_size;
  memset(dx_addr, 0, sizeof(float) * batch * in_ch * in_h * in_w);
  for (i = 0; i < batch; ++i) {
    for (j = 0; j < groups; ++j) {
      GemmCb gcb;
      for (int ci = 0; ci < m; ci += chunk) {
        float *mat_b;
        if (ci == 0) {
          mat_b = w_addr + j * nweights / groups;
          gcb.ca = 0;
          gcb.cb = 0;
          gcb.bias = nullptr;
          gcb.atype = ActType_No;
        } else {
          mat_b = gcb.mat_b;
          gcb.cb = 1;
        }
        int real_chunk = MSMIN(m - ci, chunk);
        float *mat_a = dy_addr + (i * groups) * m * k + j * (out_ch / groups) + ci * out_ch;
        float *mat_c = workspace;
        GemmMatmulPlus(0, 0, real_chunk, n, k, 1, mat_a, out_ch, mat_b, n, 0, mat_c, n, mat_workspace, &gcb);
        rolling_col2im_hwc(mat_c, dx_addr + (i * groups) * (in_ch / groups) * in_h * in_w + j * (in_ch / groups),
                           conv_param, real_chunk, ci);
      }
    }
  }

  return RET_OK;
}

int ConvolutionGradInputRun(void *cdata, int task_id) {
  auto convinput_kernel = reinterpret_cast<ConvolutionGradInputCPUKernel *>(cdata);
  auto error_code = convinput_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv input error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionGradInputCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, ConvolutionGradInputRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "bias function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuConvGradInputFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                      const std::vector<lite::Tensor *> &outputs,
                                                      OpParameter *opParameter, const lite::InnerContext *ctx,
                                                      const kernel::KernelKey &desc,
                                                      const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2DGradInput ||
            desc.type == schema::PrimitiveType_GroupConv2DGradInput);

  auto *kernel = new (std::nothrow) ConvolutionGradInputCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new kernel fail!";
    free(opParameter);
    return nullptr;
  }

  auto ret = kernel->Init();
  if (0 != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Conv2DGradInput, CpuConvGradInputFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_GroupConv2DGradInput, CpuConvGradInputFp32KernelCreator)

}  // namespace mindspore::kernel
