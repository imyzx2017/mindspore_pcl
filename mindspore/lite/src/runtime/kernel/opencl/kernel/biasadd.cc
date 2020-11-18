/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/runtime/kernel/opencl/kernel/biasadd.h"
#include <string>
#include <map>
#include <set>
#include <vector>

#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/cl/biasadd.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BiasAdd;

namespace mindspore::kernel {

int BiasAddOpenCLKernel::InitWeights() {
  int C = in_tensors_[1]->shape()[0];
  int div_ci = UP_DIV(C, C4NUM);
  auto allocator = ocl_runtime_->GetAllocator();
  size_t img_dtype = CL_FLOAT;
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  std::vector<size_t> img_size{size_t(div_ci), 1, img_dtype};
  BiasAdd_ = allocator->Malloc(div_ci * C4NUM * fp_size, img_size);
  BiasAdd_ = allocator->MapBuffer(BiasAdd_, CL_MAP_WRITE, nullptr, true);
  memset(BiasAdd_, 0x00, div_ci * C4NUM * fp_size);
  memcpy(BiasAdd_, in_tensors_[1]->data_c(), C * fp_size);
  allocator->UnmapBuffer(BiasAdd_);
  return RET_OK;
}

int BiasAddOpenCLKernel::Init() {
  in_size_ = in_tensors_[0]->shape().size();
  out_size_ = out_tensors_[0]->shape().size();
  for (int i = 0; i < in_size_; ++i) {
    input_shape_.s[i + 4 - in_size_] = in_tensors_[0]->shape()[i];
  }
  enable_fp16_ = ocl_runtime_->GetFp16Enable();
  fp_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  if (in_size_ != 4 && in_size_ != 2) {
    MS_LOG(ERROR) << "BiasAdd only support dim=4 or 2, but your dim=" << in_size_;
    return mindspore::lite::RET_ERROR;
  }
  int C = in_tensors_[0]->shape()[3];
  int Bias_Size = in_tensors_[1]->shape()[0];
  if (UP_DIV(Bias_Size, C4NUM) != UP_DIV(C, C4NUM)) {
    MS_LOG(ERROR) << "BiasAdd weight channel size:" << Bias_Size << " must be equal with in_teneors channel size:" << C;
    return mindspore::lite::RET_ERROR;
  }
  InitWeights();
  std::set<std::string> build_options;
  std::string source = biasadd_source;
  std::string program_name = "BiasAdd";
  std::string kernel_name = "BiasAdd";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);

  MS_LOG(DEBUG) << program_name << " Init Done!";
  return mindspore::lite::RET_OK;
}

int BiasAddOpenCLKernel::Run() {
  cl_int4 global_size = GetGlobalshape();
  MS_LOG(DEBUG) << op_parameter_->name_ << " Running!";
  int arg_idx = 0;
  std::map<schema::Format, int> data_type{
    {schema::Format::Format_NC4, 1}, {schema::Format::Format_NHWC4, 2}, {schema::Format::Format_NC4HW4, 3}};
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, input_shape_);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, BiasAdd_);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, data_type[schema::Format::Format_NHWC4]);
  std::vector<size_t> local = {1, 1};
  std::vector<size_t> global = {static_cast<size_t>(global_size.s[1]), static_cast<size_t>(global_size.s[2])};
  auto ret = ocl_runtime_->RunKernel(kernel_, global, local, nullptr);
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Run kernel " << op_parameter_->name_ << " error.";
    return mindspore::lite::RET_ERROR;
  }
  return mindspore::lite::RET_OK;
}

cl_int4 BiasAddOpenCLKernel::GetGlobalshape() {
  cl_int4 global_shape = input_shape_;
  global_shape.s[2] = UP_DIV(global_shape.s[3], C4NUM) * global_shape.s[2];
  return global_shape;
}

kernel::LiteKernel *OpenCLBiasAddKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                               const lite::PrimitiveC *primitive) {
  if (inputs.size() == 0) {
    MS_LOG(ERROR) << "Input data size must be greater than 0, but your size is " << inputs.size();
    free(opParameter);
    return nullptr;
  }
  if (inputs[0]->shape()[0] > 1) {
    MS_LOG(ERROR) << "Input data size unsupported multi-batch.";
    free(opParameter);
    return nullptr;
  }
  auto *kernel = new (std::nothrow) BiasAddOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Kernel " << opParameter->name_ << "is nullptr.";
    free(opParameter);
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Init BiasAdd kernel failed!";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_BiasAdd, OpenCLBiasAddKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_BiasAdd, OpenCLBiasAddKernelCreator)
}  // namespace mindspore::kernel
