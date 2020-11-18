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

#include "src/runtime/kernel/opencl/kernel/scale.h"
#include <set>
#include <vector>
#include <string>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32/common_func.h"
#include "src/runtime/kernel/opencl/utils.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/scale.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::MemType;
using mindspore::schema::PrimitiveType_Scale;

namespace mindspore::kernel {

ScaleOpenCLKernel::~ScaleOpenCLKernel() {
  auto allocator = ocl_runtime_->GetAllocator();
  if (scale_ptr_ != nullptr) {
    allocator->Free(scale_ptr_);
    scale_ptr_ = nullptr;
  }
  if (offset_ptr_ != nullptr) {
    allocator->Free(offset_ptr_);
    offset_ptr_ = nullptr;
  }
}

void ScaleOpenCLKernel::Image2dGetWorkGroupSize() {
  local_size_ = {16, 16};
  auto image2d_info = Image2DInfo(out_tensors_[0]);
  global_size_ = {image2d_info.width, image2d_info.height};
}

int ScaleOpenCLKernel::InitWeights() {
  if (!weight_vector_flag_) {
    return RET_OK;
  }
  if (in_tensors_[1]->IsConst()) {
    auto allocator = ocl_runtime_->GetAllocator();
    std::vector<size_t> img_size;
    GetImageSize(0, &img_size);
    img_size[2] = in_tensors_[1]->data_type() == kNumberTypeFloat16 ? CL_HALF_FLOAT : CL_FLOAT;
    if (broadcast_flag_) {
      img_size[1] = 1;
      img_size[0] = UP_DIV(in_tensors_[1]->shape()[0], C4NUM);
      scale_ptr_ = allocator->Malloc(in_tensors_[1]->ElementsNum(), img_size, in_tensors_[1]->data_c());
      offset_ptr_ = allocator->Malloc(in_tensors_[2]->ElementsNum(), img_size, in_tensors_[2]->data_c());
      return RET_OK;
    }
    auto image2d_info = Image2DInfo(in_tensors_[1]);
    int pack_weight_size = image2d_info.ElementsC4Num;
    int plane = image2d_info.H * image2d_info.W;
    int channel = image2d_info.C;
    int batch = image2d_info.N;
    if (in_tensors_[0]->GetFormat() == in_tensors_[1]->GetFormat()) {
      if (in_tensors_[0]->data_type() == in_tensors_[1]->data_type()) {
        scale_ptr_ = allocator->Malloc(in_tensors_[1]->ElementsNum(), img_size, in_tensors_[1]->data_c());
        offset_ptr_ = allocator->Malloc(in_tensors_[2]->ElementsNum(), img_size, in_tensors_[2]->data_c());
      } else {
        MS_LOG(ERROR) << "Unsupport data type transpose from " << in_tensors_[1]->data_type() << "to "
                      << in_tensors_[0]->data_type();
        return RET_ERROR;
      }
    } else if (in_tensors_[0]->GetFormat() == schema::Format_NHWC) {
      if (in_tensors_[1]->GetFormat() == schema::Format_NHWC) {
        if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
          auto *scale = new (std::nothrow) float[pack_weight_size];
          if (scale == nullptr) {
            MS_LOG(ERROR) << "Malloc buffer failed!";
            return RET_ERROR;
          }
          auto *offset = new (std::nothrow) float[pack_weight_size];
          if (offset == nullptr) {
            MS_LOG(ERROR) << "Malloc buffer failed!";
            delete[] scale;
            return RET_ERROR;
          }
          std::function<float(float)> to_dtype = [](float x) -> float { return x; };
          PackNHWCToNHWC4<float, float>(in_tensors_[1]->data_c(), scale, batch, plane, channel, to_dtype);
          PackNHWCToNHWC4<float, float>(in_tensors_[2]->data_c(), offset, batch, plane, channel, to_dtype);
          scale_ptr_ = allocator->Malloc(in_tensors_[1]->ElementsNum(), img_size, scale);
          offset_ptr_ = allocator->Malloc(in_tensors_[2]->ElementsNum(), img_size, offset);
          delete[] scale;
          delete[] offset;
        } else if (in_tensors_[0]->data_type() == kNumberTypeFloat16) {
          auto *scale = new (std::nothrow) float16_t[pack_weight_size];
          if (scale == nullptr) {
            MS_LOG(ERROR) << "Malloc buffer failed!";
            return RET_ERROR;
          }
          auto *offset = new (std::nothrow) float16_t[pack_weight_size];
          if (offset == nullptr) {
            MS_LOG(ERROR) << "Malloc buffer failed!";
            delete[] scale;
            return RET_ERROR;
          }
          std::function<float16_t(float)> to_dtype = [](float x) -> float16_t { return static_cast<float16_t>(x); };
          PackNHWCToNHWC4<float, float16_t>(in_tensors_[1]->data_c(), scale, batch, plane, channel, to_dtype);
          PackNHWCToNHWC4<float, float16_t>(in_tensors_[2]->data_c(), offset, batch, plane, channel, to_dtype);
          scale_ptr_ = allocator->Malloc(in_tensors_[1]->ElementsNum(), img_size, scale);
          offset_ptr_ = allocator->Malloc(in_tensors_[2]->ElementsNum(), img_size, offset);
          delete[] scale;
          delete[] offset;
        } else {
          MS_LOG(ERROR) << "Unsupport data type transpose from " << in_tensors_[1]->data_type() << "to "
                        << in_tensors_[0]->data_type();
          return RET_ERROR;
        }
      } else {
        MS_LOG(ERROR) << "Unsupport format transpose from " << in_tensors_[1]->GetFormat() << "to "
                      << in_tensors_[0]->GetFormat();
        return RET_ERROR;
      }
    }
    return RET_OK;
  }
  return RET_OK;
}

int ScaleOpenCLKernel::Init() {
  std::string kernel_name;
  auto *scale_param = reinterpret_cast<const ScaleParameter *>(op_parameter_);
  auto in_tensor = in_tensors_.at(0);
  auto in_shape = in_tensor->shape();
  auto scale_tensor = in_tensors_.at(1);
  auto scale_shape = scale_tensor->shape();
  axis_ = scale_param->axis_;
  if (axis_ < 0) {
    axis_ += in_shape.size();
  }
  if (scale_shape.size() != in_shape.size()) {
    if (scale_tensor->ElementsNum() == 1) {
      weight_vector_flag_ = false;
      kernel_name = "BoardcastScale";
    } else if (scale_shape.size() == 1) {
      weight_vector_flag_ = true;
      broadcast_flag_ = true;
      if ((in_shape.size() == 4 && axis_ == 3) || (in_shape.size() == 2 && axis_ == 1)) {
        kernel_name = "Scale_C";
      } else if (in_shape.size() == 4 && axis_ == 1) {
        kernel_name = "Scale_H";
        broadcast_H_flag_ = true;
      } else {
        MS_LOG(ERROR) << "unsupported scale axis " << axis_;
        return RET_ERROR;
      }
    } else {
      MS_LOG(ERROR) << "unsupported scale axis " << axis_ << ", in shape " << in_shape << ", scale shape"
                    << scale_shape;
      return RET_ERROR;
    }
  } else {
    weight_vector_flag_ = true;
    kernel_name = "Scale";
  }
  lite::STATUS error_code;
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  if (out_mem_type_ == MemType::IMG) {
    kernel_name += "_IMG";
  } else {
    kernel_name += "_BUF";
  }
  std::string program_name = "Scale";
  std::set<std::string> build_options;
  std::string source = scale_source;
  ocl_runtime_->LoadSource(program_name, source);
  error_code = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  if (error_code != RET_OK) {
    return error_code;
  }

  Image2dGetWorkGroupSize();
  InitWeights();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int ScaleOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto *param = reinterpret_cast<const ScaleParameter *>(op_parameter_);
  cl_int act_type = 0;
  if (param->activation_type_ == ActType_Relu) {
    act_type = 1;
  } else if (param->activation_type_ == ActType_Relu6) {
    act_type = 3;
  }

  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  if (weight_vector_flag_) {
    void *scale = scale_ptr_ == nullptr ? in_tensors_[1]->data_c() : scale_ptr_;
    void *offset = offset_ptr_ == nullptr ? in_tensors_[2]->data_c() : offset_ptr_;
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, scale);
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, offset);
  } else {
    if (in_tensors_[1]->data_type() == kNumberTypeFloat32) {
      float scale = static_cast<float *>(in_tensors_[1]->data_c())[0];
      float offset = static_cast<float *>(in_tensors_[2]->data_c())[0];
      ocl_runtime_->SetKernelArg(kernel_, arg_idx++, scale);
      ocl_runtime_->SetKernelArg(kernel_, arg_idx++, offset);
    } else if (in_tensors_[1]->data_type() == kNumberTypeFloat16) {
      float16_t scale = static_cast<float16_t *>(in_tensors_[1]->data_c())[0];
      float16_t offset = static_cast<float16_t *>(in_tensors_[2]->data_c())[0];
      ocl_runtime_->SetKernelArg(kernel_, arg_idx++, static_cast<float>(scale));
      ocl_runtime_->SetKernelArg(kernel_, arg_idx++, static_cast<float>(offset));
    } else {
      MS_LOG(ERROR) << "Unsupport data type " << in_tensors_[1]->data_type();
      return RET_ERROR;
    }
  }
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  cl_int2 output_shape{static_cast<int>(global_size_[0]), static_cast<int>(global_size_[1])};
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, output_shape);
  if (weight_vector_flag_ && broadcast_flag_) {
    if (broadcast_H_flag_) {
      ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[1]->shape()[0]);
    } else {
      ocl_runtime_->SetKernelArg(kernel_, arg_idx++, UP_DIV(in_tensors_[1]->shape()[0], C4NUM));
    }
  }
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, act_type);
  ocl_runtime_->RunKernel(kernel_, global_size_, local_size_, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLScaleKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                             const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                             const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) ScaleOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create OpenCL Scale kernel failed!";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: Scale";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Scale, OpenCLScaleKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Scale, OpenCLScaleKernelCreator)
}  // namespace mindspore::kernel
