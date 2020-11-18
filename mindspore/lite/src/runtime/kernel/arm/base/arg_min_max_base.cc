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
#include "src/runtime/kernel/arm/base/arg_min_max_base.h"
#include "nnacl/arg_min_max.h"
#include "src/runtime/kernel/arm/fp32/argminmax_fp32.h"
#include "nnacl/arithmetic_common.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "include/context.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_ArgMax;
using mindspore::schema::PrimitiveType_ArgMin;

namespace mindspore::kernel {
int ArgMinMaxBaseCPUKernel::Init() {
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  switch (op_parameter_->type_) {
    case PrimitiveType_ArgMax:
      param->get_max_ = true;
      break;
    case PrimitiveType_ArgMin:
      param->get_max_ = false;
      break;
    default:
      MS_LOG(ERROR) << "Unexpected type " << op_parameter_->type_;
      return RET_ERROR;
  }

  return RET_OK;
}

int ArgMinMaxBaseCPUKernel::ReSize() {
  auto in_shape = in_tensors_.at(0)->shape();
  auto dims_size = in_shape.size();
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  int axis = param->axis_ < 0 ? param->axis_ + dims_size : param->axis_;
  param->axis_ = axis;
  param->dims_size_ = dims_size;
  if (param->topk_ <= 0) {
    MS_LOG(ERROR) << "Invalid topk " << param->topk_;
    return RET_PARAM_INVALID;
  }
  param->topk_ = MSMIN(param->topk_, in_shape[axis]);
  ComputeStrides(in_shape.data(), param->in_strides_, in_shape.size());
  auto out_shape = out_tensors_.at(0)->shape();
  ComputeStrides(out_shape.data(), param->out_strides_, out_shape.size());
  return RET_OK;
}

int ArgMinMaxBaseCPUKernel::Run() {
  auto input_data = in_tensors_.at(0)->MutableData();
  auto output_data = out_tensors_.at(0)->MutableData();

  auto shape = in_tensors_.at(0)->shape();

  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  MS_ASSERT(context_->allocator != nullptr);
  if (param->topk_ > 1 || param->keep_dims_) {
    param->arg_elements_ =
      reinterpret_cast<ArgElement *>(context_->allocator->Malloc(sizeof(ArgElement) * shape[param->axis_]));
    if (param->arg_elements_ == nullptr) {
      MS_LOG(ERROR) << "malloc memroy fail!";
      return RET_ERROR;
    }
  }
  ArgMinMax(input_data, output_data, reinterpret_cast<const int *>(shape.data()), param);
  context_->allocator->Free(param->arg_elements_);
  param->arg_elements_ = nullptr;
  return RET_OK;
}

kernel::LiteKernel *CpuArgMinMaxFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                                  const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                  const mindspore::lite::PrimitiveC *primitive) {
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Input op_parameter is nullptr!";
    return nullptr;
  }
  auto kernel = new (std::nothrow) ArgMinMaxCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ArgMinMaxCPUKernel fail!";
    free(op_parameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ArgMax, CpuArgMinMaxFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ArgMin, CpuArgMinMaxFp32KernelCreator)

}  // namespace mindspore::kernel
