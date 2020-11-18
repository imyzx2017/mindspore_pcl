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
#include <cstring>
#include <string>
#include <algorithm>
#include <set>
#include <utility>
#include <functional>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/kernel/argminmax.h"
#include "src/runtime/kernel/opencl/cl/argminmax.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ArgMax;
using mindspore::schema::PrimitiveType_ArgMin;

namespace mindspore::kernel {

int ArgMinMaxOpenCLKernel::CheckSpecs() {
  if (in_tensors_[0]->data_type() != kNumberTypeFloat32 && in_tensors_[0]->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Unsupported data type " << in_tensors_[0]->data_type();
    return RET_ERROR;
  }
  if (in_tensors_[0]->shape().size() > 4 && in_tensors_[0]->shape().size() == 0) {
    MS_LOG(ERROR) << "input shape size must be (1-4), actual: " << in_tensors_[0]->shape().size() << ", "
                  << out_tensors_[0]->shape().size();
    return RET_ERROR;
  }
  auto *param = reinterpret_cast<ArgMinMaxParameter *>(this->op_parameter_);
  param->dims_size_ = in_tensors_[0]->shape().size();
  param->axis_ = (param->axis_ + param->dims_size_) % param->dims_size_;
  if (param->axis_ < 0 || param->axis_ >= param->dims_size_) {
    MS_LOG(ERROR) << "Invalid axis " << param->axis_;
    return RET_ERROR;
  }
  param->get_max_ = (op_parameter_->type_ == PrimitiveType_ArgMax);
  return RET_OK;
}

void ArgMinMaxOpenCLKernel::SetConstArgs() {
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  cl_int4 in_shape{static_cast<int>(im_in_.N), static_cast<int>(im_in_.H), static_cast<int>(im_in_.W),
                   static_cast<int>(im_in_.C)};
  in_shape.s[0] = UP_ROUND(im_in_.C, C4NUM) - im_in_.C;
  in_shape.s[1] = im_in_.W * im_in_.C;
  cl_int4 flags = {param->out_value_, param->get_max_, param->axis_, param->topk_};
  int arg_cnt = 2;
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, buff_, lite::opencl::MemType::BUF);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, ids_, lite::opencl::MemType::BUF);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, in_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, src_size_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, cus_size_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, strides_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, flags);
}

void ArgMinMaxOpenCLKernel::SetGlobalLocal() {
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  auto in_shape = in_tensors_[0]->shape();
  auto in_shape_align = in_shape;
  in_shape_align[3] = UP_ROUND(in_shape[3], C4NUM);
  im_in_ = Image2DInfo(in_tensors_[0]);
  auto out_shape_align = in_shape_align;
  out_shape_align.at(param->axis_) = param->axis_ == 3 ? UP_ROUND(param->topk_, C4NUM) : param->topk_;
  int reduce_len = GetUpPow2(in_shape.at(param->axis_));
  cus_size_ = {reduce_len, static_cast<int>(im_in_.RowPitch() / C4NUM), 1, 1};
  cus_size_.s[2] = UP_ROUND(im_in_.width * C4NUM, cus_size_.s[1]) - im_in_.width * C4NUM;
  cus_size_.s[3] = im_in_.W * UP_ROUND(param->topk_, C4NUM);
  cus_size_.s[3] = UP_ROUND(cus_size_.s[3], cus_size_.s[1]) - cus_size_.s[3];
  src_size_ = {std::accumulate(in_shape.begin() + param->axis_ + 1, in_shape.end(), 1, std::multiplies<int>()),
               std::accumulate(in_shape.begin(), in_shape.begin() + param->axis_, 1, std::multiplies<int>()),
               std::accumulate(in_shape.begin() + param->axis_, in_shape.end(), 1, std::multiplies<int>()),
               in_shape.at(param->axis_)};
  strides_ = {
    std::accumulate(in_shape_align.begin() + param->axis_ + 1, in_shape_align.end(), 1, std::multiplies<int>()),
    std::accumulate(in_shape_align.begin() + param->axis_, in_shape_align.end(), 1, std::multiplies<int>()),
    std::accumulate(out_shape_align.begin() + param->axis_ + 1, out_shape_align.end(), 1, std::multiplies<int>()),
    std::accumulate(out_shape_align.begin() + param->axis_, out_shape_align.end(), 1, std::multiplies<int>()),
  };
  switch (param->axis_) {
    case 0:
      strides_.s[0] = UP_ROUND(strides_.s[0] / im_in_.H, cus_size_.s[1]) * im_in_.H;
      strides_.s[1] = strides_.s[0] * im_in_.N;
      strides_.s[2] = UP_ROUND(strides_.s[2] / im_in_.H, cus_size_.s[1]) * im_in_.H;
      strides_.s[3] = strides_.s[2] * param->topk_;
      break;
    case 1:
      strides_.s[0] = UP_ROUND(strides_.s[0], cus_size_.s[1]);
      strides_.s[1] = UP_ROUND(strides_.s[1] / im_in_.H, cus_size_.s[1]) * im_in_.H;
      strides_.s[2] = UP_ROUND(strides_.s[2], cus_size_.s[1]);
      strides_.s[3] = UP_ROUND(strides_.s[3] / param->topk_, cus_size_.s[1]) * param->topk_;
      break;
    case 2:
      strides_.s[1] = UP_ROUND(strides_.s[1], cus_size_.s[1]);
      strides_.s[3] = UP_ROUND(strides_.s[3], cus_size_.s[1]);
      break;
    default:  // 3
      break;
  }
  std::vector<size_t> local = {1, 1, 1};
  std::vector<size_t> global = {static_cast<size_t>(strides_.s[0]), static_cast<size_t>(src_size_.s[1]), 1};
  OpenCLKernel::AlignGlobalLocal(global, local);
}

int ArgMinMaxOpenCLKernel::InitWeights() {
  auto allocator = ocl_runtime_->GetAllocator();
  int dtype_size = ocl_runtime_->GetFp16Enable() ? sizeof(int16_t) : sizeof(float);
  buff_ = allocator->Malloc(in_tensors_[0]->ElementsNum() * dtype_size);
  ids_ = allocator->Malloc(in_tensors_[0]->ElementsNum() * sizeof(int32_t));
  return RET_OK;
}

int ArgMinMaxOpenCLKernel::Prepare() {
  std::string kernel_name = "argminmax";

#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else

  std::set<std::string> build_options;
  std::string source = argminmax_source;
  std::string program_name = "argminmax";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif

  InitWeights();
  SetGlobalLocal();
  SetConstArgs();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int ArgMinMaxOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_[0]->data_c(), lite::opencl::MemType::BUF);
  ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_[0]->data_c(), lite::opencl::MemType::BUF);
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr);

  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_ArgMin, OpenCLKernelCreator<ArgMinMaxOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_ArgMin, OpenCLKernelCreator<ArgMinMaxOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_ArgMax, OpenCLKernelCreator<ArgMinMaxOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_ArgMax, OpenCLKernelCreator<ArgMinMaxOpenCLKernel>);
}  // namespace mindspore::kernel
