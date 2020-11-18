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

#include "src/runtime/kernel/arm/fp16/arithmetic_fp16.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "nnacl/fp16/arithmetic_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "src/ops/populate/populate_register.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

using mindspore::schema::PrimitiveType_Add;
using mindspore::schema::PrimitiveType_Div;
using mindspore::schema::PrimitiveType_Eltwise;
using mindspore::schema::PrimitiveType_Equal;
using mindspore::schema::PrimitiveType_FloorDiv;
using mindspore::schema::PrimitiveType_FloorMod;
using mindspore::schema::PrimitiveType_Greater;
using mindspore::schema::PrimitiveType_GreaterEqual;
using mindspore::schema::PrimitiveType_Less;
using mindspore::schema::PrimitiveType_LessEqual;
using mindspore::schema::PrimitiveType_LogicalAnd;
using mindspore::schema::PrimitiveType_LogicalOr;
using mindspore::schema::PrimitiveType_Maximum;
using mindspore::schema::PrimitiveType_Minimum;
using mindspore::schema::PrimitiveType_Mul;
using mindspore::schema::PrimitiveType_NotEqual;
using mindspore::schema::PrimitiveType_SquaredDifference;
using mindspore::schema::PrimitiveType_Sub;

namespace mindspore::kernel {
ARITHMETIC_FUNC_INFO_FP16 arithmetic_fun_table_fp16[] = {
  {PrimitiveType_Mul, schema::ActivationType_RELU, ElementMulReluFp16, ElementOptMulReluFp16},
  {PrimitiveType_Mul, schema::ActivationType_RELU6, ElementMulRelu6Fp16, ElementOptMulRelu6Fp16},
  {PrimitiveType_Mul, schema::ActivationType_NO_ACTIVATION, ElementMulFp16, ElementOptMulFp16},
  {PrimitiveType_Add, schema::ActivationType_RELU, ElementAddReluFp16, ElementOptAddReluFp16},
  {PrimitiveType_Add, schema::ActivationType_RELU6, ElementAddRelu6Fp16, ElementOptAddRelu6Fp16},
  {PrimitiveType_Add, schema::ActivationType_NO_ACTIVATION, ElementAddFp16, ElementOptAddFp16},
  {PrimitiveType_Sub, schema::ActivationType_RELU, ElementSubReluFp16, ElementOptSubReluFp16},
  {PrimitiveType_Sub, schema::ActivationType_RELU6, ElementSubRelu6Fp16, ElementOptSubRelu6Fp16},
  {PrimitiveType_Sub, schema::ActivationType_NO_ACTIVATION, ElementSubFp16, ElementOptSubFp16},
  {PrimitiveType_Div, schema::ActivationType_RELU, ElementDivReluFp16, ElementOptDivReluFp16},
  {PrimitiveType_Div, schema::ActivationType_RELU6, ElementDivRelu6Fp16, ElementOptDivRelu6Fp16},
  {PrimitiveType_Div, schema::ActivationType_NO_ACTIVATION, ElementDivFp16, ElementOptDivFp16},
  {PrimitiveType_FloorMod, schema::ActivationType_NO_ACTIVATION, ElementFloorModFp16, ElementOptFloorModFp16},
  {PrimitiveType_FloorDiv, schema::ActivationType_NO_ACTIVATION, ElementFloorDivFp16, ElementOptFloorDivFp16},
  {PrimitiveType_LogicalAnd, schema::ActivationType_NO_ACTIVATION, ElementLogicalAndFp16, ElementOptLogicalAndFp16},
  {PrimitiveType_LogicalOr, schema::ActivationType_NO_ACTIVATION, ElementLogicalOrFp16, ElementOptLogicalOrFp16},
  {PrimitiveType_SquaredDifference, schema::ActivationType_NO_ACTIVATION, ElementSquaredDifferenceFp16,
   ElementOptSquaredDifferenceFp16},
  {PrimitiveType_Maximum, schema::ActivationType_NO_ACTIVATION, ElementMaximumFp16, ElementOptMaximumFp16},
  {PrimitiveType_Minimum, schema::ActivationType_NO_ACTIVATION, ElementMinimumFp16, ElementOptMinimumFp16}};

ArithmeticFuncFp16 GetArithmeticFun(int primitive_type, int activation_type) {
  for (size_t i = 0; i < sizeof(arithmetic_fun_table_fp16); i++) {
    if (arithmetic_fun_table_fp16[i].primitive_type_ == primitive_type &&
        arithmetic_fun_table_fp16[i].activation_type_ == activation_type) {
      return arithmetic_fun_table_fp16[i].func_;
    }
  }
  return nullptr;
}

ArithmeticOptFuncFp16 GetOptimizedArithmeticFun(int primitive_type, int activation_type) {
  for (size_t i = 0; i < sizeof(arithmetic_fun_table_fp16); i++) {
    if (arithmetic_fun_table_fp16[i].primitive_type_ == primitive_type &&
        arithmetic_fun_table_fp16[i].activation_type_ == activation_type) {
      return arithmetic_fun_table_fp16[i].opt_func_;
    }
  }
  return nullptr;
}

int ArithmeticFP16CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArithmeticFP16CPUKernel::PreProcess() {
  if (!InferShapeDone()) {
    (const_cast<mindspore::lite::PrimitiveC *>(primitive_))->SetInferFlag(true);
    auto ret = (const_cast<mindspore::lite::PrimitiveC *>(primitive_))->InferShape(in_tensors_, out_tensors_);
    if (ret != 0) {
      (const_cast<mindspore::lite::PrimitiveC *>(primitive_))->SetInferFlag(false);
      MS_LOG(ERROR) << "InferShape fail!";
      return ret;
    }
    if (op_parameter_ != nullptr) {
      free(op_parameter_);
      op_parameter_ = nullptr;
    }
    op_parameter_ = PopulateArithmetic(primitive_);
    if (op_parameter_ == nullptr) {
      MS_LOG(ERROR) << "Malloc parameter failed";
      return RET_ERROR;
    }
    param_ = reinterpret_cast<ArithmeticParameter *>(op_parameter_);
    ret = ReSize();
    if (ret != 0) {
      MS_LOG(ERROR) << "ReSize fail!ret: " << ret;
      return ret;
    }
  }

  auto outputs = this->out_tensors();
  for (auto *output : outputs) {
    MS_ASSERT(output != nullptr);
    output->MallocData();
  }
  return RET_OK;
}

int ArithmeticFP16CPUKernel::ReSize() {
  param_->in_elements_num0_ = in_tensors_[0]->ElementsNum();
  param_->in_elements_num1_ = in_tensors_[1]->ElementsNum();
  param_->out_elements_num_ = out_tensors_[0]->ElementsNum();

  if (param_->in_elements_num0_ == 1 || param_->in_elements_num1_ == 1) {
    param_->broadcasting_ = false;
    arithmetic_opt_func_ = GetOptimizedArithmeticFun(param_->op_parameter_.type_, param_->activation_type_);
  } else {
    arithmetic_func_ = GetArithmeticFun(param_->op_parameter_.type_, param_->activation_type_);
  }
  if (arithmetic_opt_func_ == nullptr && arithmetic_func_ == nullptr) {
    MS_LOG(ERROR) << "arithmetic_opt_func_ and arithmetic_func_ function is nullptr!";
    return RET_ERROR;
  }
  if (param_->broadcasting_) {
    outside_ = 1;
    for (int i = param_->ndim_ - 1; i >= 0; --i) {
      if (param_->in_shape0_[i] != param_->in_shape1_[i]) {
        break_pos_ = i;
        break;
      }
      outside_ *= param_->out_shape_[i];
    }
    ComputeStrides(param_->in_shape0_, param_->in_strides0_, param_->ndim_);
    ComputeStrides(param_->in_shape1_, param_->in_strides1_, param_->ndim_);
    ComputeStrides(param_->out_shape_, param_->out_strides_, param_->ndim_);
  }
  return RET_OK;
}

int ArithmeticFP16CPUKernel::BroadcastRun(float16_t *input0, float16_t *input1, float16_t *output, int dim,
                                          int out_count, int cur_offset) {
  if (dim > break_pos_) {
    return arithmetic_func_(input0 + cur_offset, input1 + cur_offset, output + cur_offset, out_count);
  }
  for (int i = 0; i < param_->out_shape_[dim]; ++i) {
    int pos0 = param_->in_shape0_[dim] == 1 ? 0 : i;
    int pos1 = param_->in_shape1_[dim] == 1 ? 0 : i;
    int ret = BroadcastRun(input0 + pos0 * param_->in_strides0_[dim], input1 + pos1 * param_->in_strides1_[dim],
                           output + i * param_->out_strides_[dim], dim + 1, out_count, cur_offset);
    if (ret != RET_OK) {
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ArithmeticFP16CPUKernel::DoArithmetic(int task_id) {
  int stride_per_thread = UP_DIV(param_->broadcasting_ ? outside_ : param_->out_elements_num_, context_->thread_num_);
  int cur_offset = stride_per_thread * task_id;
  int cur_count = param_->broadcasting_ ? MSMIN(stride_per_thread, outside_ - cur_offset)
                                        : MSMIN(stride_per_thread, param_->out_elements_num_ - cur_offset);

  int ret = RET_OK;
  if (param_->broadcasting_) {
    ret = BroadcastRun(input0_fp16_, input1_fp16_, output_fp16_, 0, cur_count, cur_offset);
  } else if (param_->in_elements_num0_ == 1) {
    ret = arithmetic_opt_func_(input0_fp16_, input1_fp16_ + cur_offset, output_fp16_ + cur_offset, cur_count, param_);
  } else if (param_->in_elements_num1_ == 1) {
    ret = arithmetic_opt_func_(input0_fp16_ + cur_offset, input1_fp16_, output_fp16_ + cur_offset, cur_count, param_);
  } else {
    ret = arithmetic_func_(input0_fp16_ + cur_offset, input1_fp16_ + cur_offset, output_fp16_ + cur_offset, cur_count);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoArithmetic failed, ret = " << ret;
  }
  return ret;
}

static int ArithmeticsRunFp16(void *cdata, int task_id) {
  auto arithmetic_kernel = reinterpret_cast<ArithmeticFP16CPUKernel *>(cdata);
  auto ret = arithmetic_kernel->DoArithmetic(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticsRunFp16 error task_id[" << task_id << "] ret[" << ret << "]";
  }
  return ret;
}

int ArithmeticFP16CPUKernel::Run() {
  auto output_tensor = out_tensors_.at(0);
  is_input0_fp32_ = in_tensors_.at(0)->data_type() == kNumberTypeFloat32;
  is_input1_fp32_ = in_tensors_.at(1)->data_type() == kNumberTypeFloat32;
  is_output_fp32_ = output_tensor->data_type() == kNumberTypeFloat32;

  input0_fp16_ = ConvertInputFp32toFp16(in_tensors_.at(0), context_);
  input1_fp16_ = ConvertInputFp32toFp16(in_tensors_.at(1), context_);
  output_fp16_ = MallocOutputFp16(output_tensor, context_);
  if (input0_fp16_ == nullptr || input1_fp16_ == nullptr || output_fp16_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  auto ret = ParallelLaunch(this->context_->thread_pool_, ArithmeticsRunFp16, this, context_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticsRunFp16 run error error_code[" << ret << "]";
  }
  if (is_output_fp32_) {
    Float16ToFloat32(output_fp16_, reinterpret_cast<float *>(output_tensor->MutableData()),
                     output_tensor->ElementsNum());
  }
  FreeTmpBuffer();
  return ret;
}

void ArithmeticFP16CPUKernel::FreeTmpBuffer() {
  if (is_input0_fp32_) {
    context_->allocator->Free(input0_fp16_);
    input0_fp16_ = nullptr;
  }
  if (is_input1_fp32_) {
    context_->allocator->Free(input1_fp16_);
    input1_fp16_ = nullptr;
  }
  if (is_output_fp32_) {
    context_->allocator->Free(output_fp16_);
    output_fp16_ = nullptr;
  }
}

kernel::LiteKernel *CpuArithmeticFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                   const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                                   const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                   const mindspore::lite::PrimitiveC *primitive) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "input parameter is null!";
    return nullptr;
  }
  auto kernel = new (std::nothrow) ArithmeticFP16CPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed, name: " << parameter->name_;
    free(parameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << parameter->name_
                  << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Mul, CpuArithmeticFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Add, CpuArithmeticFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Sub, CpuArithmeticFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Div, CpuArithmeticFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_FloorMod, CpuArithmeticFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_FloorDiv, CpuArithmeticFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_LogicalAnd, CpuArithmeticFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_LogicalOr, CpuArithmeticFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Maximum, CpuArithmeticFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Minimum, CpuArithmeticFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Eltwise, CpuArithmeticFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_SquaredDifference, CpuArithmeticFp16KernelCreator)
}  // namespace mindspore::kernel
