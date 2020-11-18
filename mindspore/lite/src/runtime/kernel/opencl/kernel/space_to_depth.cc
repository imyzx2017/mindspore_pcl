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

#include <set>
#include <string>
#include <map>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/space_to_depth.h"
#include "src/runtime/kernel/opencl/cl/space_to_depth.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_SpaceToDepth;

namespace mindspore::kernel {
int SpaceToDepthOpenCLKernel::CheckSpecs() { return RET_OK; }

int SpaceToDepthOpenCLKernel::Prepare() {
  std::string kernel_name;
  in_shape_ = Image2DInfo(in_tensors_[0]);
  out_shape_ = Image2DInfo(out_tensors_[0]);
  if (in_shape_.C % C4NUM != 0) {
    kernel_name = "SpaceToDepth";
  } else {
    kernel_name = "SpaceToDepthAlign";
  }
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::set<std::string> build_options;
  std::string source = space_to_depth_source;
  std::string program_name = "SpaceToDepth";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  SetConstArgs();
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return mindspore::lite::RET_OK;
}
void SpaceToDepthOpenCLKernel::SetConstArgs() {
  cl_int4 cl_in_shape = {static_cast<cl_int>(in_shape_.N), static_cast<cl_int>(in_shape_.H),
                         static_cast<cl_int>(in_shape_.W), static_cast<cl_int>(in_shape_.Slice)};
  cl_int4 cl_out_shape = {static_cast<cl_int>(out_shape_.N), static_cast<cl_int>(out_shape_.H),
                          static_cast<cl_int>(out_shape_.W), static_cast<cl_int>(out_shape_.Slice)};
  auto param = reinterpret_cast<SpaceToDepthParameter *>(op_parameter_);
  int arg_idx = 2;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, cl_in_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, cl_out_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, param->block_size_);
  int ci_size = in_shape_.C;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, ci_size);
}
void SpaceToDepthOpenCLKernel::SetGlobalLocal() {
  global_range_ = {out_shape_.Slice, out_shape_.W, out_shape_.H * out_shape_.N};
}

int SpaceToDepthOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr);
  return mindspore::lite::RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_SpaceToDepth, OpenCLKernelCreator<SpaceToDepthOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_SpaceToDepth, OpenCLKernelCreator<SpaceToDepthOpenCLKernel>)
}  // namespace mindspore::kernel
