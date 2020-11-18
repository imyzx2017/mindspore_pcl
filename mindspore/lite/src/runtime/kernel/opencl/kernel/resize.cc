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

#include "src/runtime/kernel/opencl/kernel/resize.h"
#include <map>
#include <set>
#include <string>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/cl/resize.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_Resize;

namespace mindspore::kernel {

int ResizeOpenCLKernel::CheckSpecs() {
  auto in_shape = in_tensors_[0]->shape();
  auto out_shape = out_tensors_[0]->shape();
  if (in_shape.size() != 4 || out_shape.size() != 4 || in_shape[0] != out_shape[0] || in_shape[3] != out_shape[3]) {
    MS_LOG(ERROR) << "resize op only support 4D and axes HW";
    return RET_PARAM_INVALID;
  }
  auto resize_param = reinterpret_cast<ResizeParameter *>(op_parameter_);
  if (resize_param->method_ != schema::ResizeMethod_LINEAR && resize_param->method_ != schema::ResizeMethod_NEAREST) {
    MS_LOG(ERROR) << "unsupported resize method:" << resize_param->method_;
    return RET_PARAM_INVALID;
  }
  return RET_OK;
}

int ResizeOpenCLKernel::Prepare() {
  auto resize_param = reinterpret_cast<ResizeParameter *>(op_parameter_);
  alignCorner = resize_param->align_corners_;
  preserveAspectRatio = resize_param->preserve_aspect_ratio_;
  auto in_shape = in_tensors_[0]->shape();
  auto out_shape = out_tensors_[0]->shape();
  std::string kernel_name = "resize";
  if (resize_param->method_ == schema::ResizeMethod_LINEAR) {
    kernel_name += "_bilinear";
  } else if (resize_param->method_ == schema::ResizeMethod_NEAREST) {
    kernel_name += "_nearest_neighbor";
  }
  kernel_name += "_NHWC4";
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::set<std::string> build_options;
  std::string source = resize_source;
  std::string program_name = "Resize";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  SetConstArgs();
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

float ResizeOpenCLKernel::getResizeScaleFactor(int input_size, int output_size) {
  return input_size > 1 && output_size > 1 && alignCorner
           ? static_cast<float>(input_size - 1) / static_cast<float>(output_size - 1)
           : static_cast<float>(input_size) / static_cast<float>(output_size);
}

void ResizeOpenCLKernel::SetConstArgs() {
  auto in_shape = in_tensors_[0]->shape();
  auto out_shape = out_tensors_[0]->shape();
  int n = out_shape[0];
  int h = out_shape[1];
  int w = out_shape[2];
  int c = out_shape[3];
  int c4 = UP_DIV(c, C4NUM);
  float scale_h = getResizeScaleFactor(in_tensors_[0]->shape()[1], out_tensors_[0]->shape()[1]);
  float scale_w = getResizeScaleFactor(in_tensors_[0]->shape()[2], out_tensors_[0]->shape()[2]);
  cl_int4 in_size = {in_shape[0], in_shape[1], in_shape[2], UP_DIV(in_shape[3], C4NUM)};
  cl_int4 out_size = {n, h, w, c4};
  cl_float2 scale = {scale_h, scale_w};
  int arg_idx = 2;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, scale);
}

void ResizeOpenCLKernel::SetGlobalLocal() {
  local_range_ = {};
  auto out_shape = Image2DInfo(out_tensors_[0]);
  global_range_ = {out_shape.Slice, out_shape.W, out_shape.H};
}

int ResizeOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr);
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Resize, OpenCLKernelCreator<ResizeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Resize, OpenCLKernelCreator<ResizeOpenCLKernel>)
}  // namespace mindspore::kernel
