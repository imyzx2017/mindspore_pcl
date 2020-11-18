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
#include <deque>
#include <string>
#include <algorithm>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/strided_slice.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/strided_slice.cl.inc"
#include "nnacl/strided_slice.h"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Slice;
using mindspore::schema::PrimitiveType_StridedSlice;

namespace mindspore::kernel {

int SliceOpenCLKernel::CheckSpecs() {
  const std::string kernel_name = op_parameter_->type_ == PrimitiveType_Slice ? "Slice" : "StridedSlice";
  if (in_tensors_.size() != 1) {
    MS_LOG(ERROR) << kernel_name + " only supports 1 input Tensor.";
    return RET_ERROR;
  }
  if (out_tensors_.size() != 1) {
    MS_LOG(ERROR) << kernel_name + " only supports 1 output Tensor.";
    return RET_ERROR;
  }
  auto in_ndim = in_tensors_.front()->shape().size();
  if (in_ndim == 0 || in_ndim > 4) {
    MS_LOG(ERROR) << kernel_name + " only supports 1D-4D input tensor";
    return RET_ERROR;
  }
  auto out_ndim = out_tensors_.front()->shape().size();
  if (out_ndim > 4) {
    MS_LOG(ERROR) << kernel_name + " only supports 0D-4D output tensor";
    return RET_ERROR;
  }
  if (InitConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "call SliceOpenCLKernel::InitConstArgs() failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int SliceOpenCLKernel::Prepare() {
  std::set<std::string> build_options;
  std::string program_name = "strided_slice";
  ocl_runtime_->LoadSource(program_name, strided_slice_source);
  ocl_runtime_->BuildKernel(kernel_, program_name, "strided_slice", build_options);
  SetConstArgs();
  SetGlobalLocal();
  return RET_OK;
}

int SliceOpenCLKernel::InitConstArgs() {
  auto input_info = Image2DInfo(in_tensors_.front());
  auto output_info = Image2DInfo(out_tensors_.front());
  input_shape_ = {static_cast<cl_int>(input_info.N), static_cast<cl_int>(input_info.H),
                  static_cast<cl_int>(input_info.W), static_cast<cl_int>(input_info.C)};
  output_shape_ = {static_cast<cl_int>(output_info.N), static_cast<cl_int>(output_info.H),
                   static_cast<cl_int>(output_info.W), static_cast<cl_int>(output_info.C)};
  io_slices_ = {static_cast<cl_int>(input_info.Slice), static_cast<cl_int>(output_info.Slice)};

  if (op_parameter_->type_ == PrimitiveType_Slice) {
    auto param = reinterpret_cast<SliceParameter *>(op_parameter_);
    Broadcast2GpuShape(param->begin_, begin_.s, param->param_length_, 0);
    Broadcast2GpuShape(param->size_, size_.s, param->param_length_, -1);
    for (int i = 0; i < 4; ++i) {
      if (begin_.s[i] < 0) {
        begin_.s[i] += input_shape_.s[i];
      }
      if (begin_.s[i] < 0 || begin_.s[i] >= input_shape_.s[i]) {
        MS_LOG(ERROR) << "Slice kernel only supports 0<=begin<input_shape but begin[i]=" << begin_.s[i]
                      << " input_shape[i]=" << input_shape_.s[i];
        return RET_ERROR;
      }
      if (size_.s[i] < -1 || size_.s[i] == 0) {
        MS_LOG(ERROR) << "Slice kernel only supports size=-1 or size>0 but size[i]=" << size_.s[i];
        return RET_ERROR;
      }
      if (size_.s[i] == -1 || begin_.s[i] + size_.s[i] > input_shape_.s[i]) {
        size_.s[i] = input_shape_.s[i] - begin_.s[i];
      }
    }
  } else {
    auto param = reinterpret_cast<StridedSliceParameter *>(op_parameter_);
    cl_int4 end = input_shape_;
    Broadcast2GpuShape(param->begins_, begin_.s, param->num_axes_, 0);
    Broadcast2GpuShape(param->strides_, stride_.s, param->num_axes_, 1);
    Broadcast2GpuShape(param->ends_, end.s, param->num_axes_);

    for (int i = 0; i < 4; ++i) {
      // begin is negative
      if (begin_.s[i] < 0) {
        begin_.s[i] += input_shape_.s[i];
      }
      // avoid begin is out of range
      begin_.s[i] = std::clamp(begin_.s[i], 0, input_shape_.s[i] - 1);
      // end is negative
      if (end.s[i] < 0) {
        end.s[i] += input_shape_.s[i];
      }
      // avoid end is out of range
      end.s[i] = std::clamp(end.s[i], -1, input_shape_.s[i]);

      // check stride begin end
      if (stride_.s[i] > 0) {
        if (begin_.s[i] >= end.s[i]) {
          MS_LOG(ERROR) << "StridedSlice kernel only supports begin_<end when stride>0";
          return RET_ERROR;
        }
      } else if (stride_.s[i] < 0) {
        if (begin_.s[i] <= end.s[i]) {
          MS_LOG(ERROR) << "StridedSlice kernel only supports begin_>end when stride<0";
          return RET_ERROR;
        }
      } else {
        MS_LOG(ERROR) << "StridedSlice kernel only supports stride!=0";
        return RET_ERROR;
      }
      size_.s[i] = std::ceil(static_cast<float>(end.s[i] - begin_.s[i]) / static_cast<float>(stride_.s[i]));
    }
  }

  // check size
  std::vector<int> shape_not_1;
  std::vector<int> size_not_1;
  std::copy_if(out_tensors_.front()->shape().begin(), out_tensors_.front()->shape().end(), shape_not_1.begin(),
               [](int x) { return x > 1; });
  std::copy_if(size_.s, size_.s + 4, size_not_1.begin(), [](int x) { return x > 1; });
  if (shape_not_1 != size_not_1) {
    MS_LOG(ERROR) << "Slice/StridedSlice kernel output shape infer error";
    return RET_ERROR;
  }
  return RET_OK;
}

void SliceOpenCLKernel::SetConstArgs() {
  int arg_cn = 2;
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, input_shape_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, output_shape_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, io_slices_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, begin_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, stride_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn, size_);
}

void SliceOpenCLKernel::SetGlobalLocal() {
  auto output_info = Image2DInfo(out_tensors_.front());
  std::vector<size_t> global = {output_info.N * output_info.H, output_info.W, output_info.Slice};

  const int max_divider = 8;
  auto max_work_group_size = ocl_runtime_->DeviceMaxWorkGroupSize();
  size_t local_c = GetMaxDivisorStrategy0(global[2], max_divider);
  size_t local_hw = max_work_group_size / local_c;
  size_t local_h = std::min(UP_DIV(global[0], 2), local_hw);
  size_t local_w = std::min(local_hw / local_h, global[1]);
  std::vector<size_t> local = {local_h, local_w, local_c};
  AlignGlobalLocal(global, local);
}

int SliceOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_[0]->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr);
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Slice, OpenCLKernelCreator<SliceOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Slice, OpenCLKernelCreator<SliceOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_StridedSlice, OpenCLKernelCreator<SliceOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_StridedSlice, OpenCLKernelCreator<SliceOpenCLKernel>);
}  // namespace mindspore::kernel
