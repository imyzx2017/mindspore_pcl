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

#include "src/runtime/kernel/arm/int8/add_int8.h"
#include <limits>
#include <algorithm>
#include "nnacl/arithmetic_common.h"
#include "nnacl/quantization/quantize.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/common/file_utils.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Add;

namespace mindspore::kernel {
int QuantizedAddCPUKernel::Init() {
  auto *input0 = in_tensors_.at(0);
  auto *input1 = in_tensors_.at(1);
  auto *output = out_tensors_.at(0);

  para_.in0_zp_ = input0->GetQuantParams().front().zeroPoint * -1;
  para_.in1_zp_ = input1->GetQuantParams().front().zeroPoint * -1;
  para_.out_zp_ = output->GetQuantParams().front().zeroPoint;

  const double in0_scale = input0->GetQuantParams().front().scale;
  const double in1_scale = input1->GetQuantParams().front().scale;
  const double out_scale = output->GetQuantParams().front().scale;

  para_.left_shift_ = 20;
  const double twice_max_input_scale = 2 * std::max(in0_scale, in1_scale);
  const double in0_multiplier = in0_scale / twice_max_input_scale;
  const double in1_multiplier = in1_scale / twice_max_input_scale;
  const double out_multiplier = twice_max_input_scale / ((1 << para_.left_shift_) * out_scale);

  QuantizeMultiplierSmallerThanOne(in0_multiplier, &para_.in0_multiplier_, &para_.in0_left_shift_);
  QuantizeMultiplierSmallerThanOne(in1_multiplier, &para_.in1_multiplier_, &para_.in1_left_shift_);
  QuantizeMultiplierSmallerThanOne(out_multiplier, &para_.out_multiplier_, &para_.out_left_shift_);

  para_.in0_right_shift_ = -para_.in0_left_shift_ > 0 ? 0 : para_.in0_left_shift_;
  para_.in1_right_shift_ = -para_.in1_left_shift_ > 0 ? 0 : para_.in1_left_shift_;
  para_.out_right_shift_ = -para_.out_left_shift_ > 0 ? 0 : para_.out_left_shift_;

  para_.in0_left_shift_ = -para_.in0_left_shift_ > 0 ? -para_.in0_left_shift_ : 0;
  para_.in1_left_shift_ = -para_.in1_left_shift_ > 0 ? -para_.in1_left_shift_ : 0;
  para_.out_left_shift_ = -para_.out_left_shift_ > 0 ? -para_.out_left_shift_ : 0;

  auto act = arith_para_->activation_type_;
  CalculateActivationRangeQuantized(act == ActType_Relu, act == ActType_Relu6, 0, 1, &para_.min_, &para_.max_);

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int QuantizedAddCPUKernel::ReSize() {
  auto *input0 = in_tensors_.at(0);
  auto *input1 = in_tensors_.at(1);
  auto *output = out_tensors_.at(0);
  support_opt_add_ = (input0->ElementsNum() == 1) || (input1->ElementsNum() == 1);
  if (support_opt_add_) {
    arith_para_->broadcasting_ = false;
  }

  elements_num_ = output->ElementsNum();
  thread_count_ = MSMIN(elements_num_, op_parameter_->thread_num_);

  arith_para_->in_elements_num0_ = in_tensors_[0]->ElementsNum();
  arith_para_->in_elements_num1_ = in_tensors_[1]->ElementsNum();
  arith_para_->out_elements_num_ = out_tensors_[0]->ElementsNum();

  memcpy(arith_para_->in_shape0_, input0->shape().data(), input0->shape().size() * sizeof(int));
  memcpy(arith_para_->in_shape1_, input1->shape().data(), input1->shape().size() * sizeof(int));
  memcpy(arith_para_->out_shape_, output->shape().data(), output->shape().size() * sizeof(int));

  if (arith_para_->broadcasting_) {
    size_t break_pos_ = 0;
    for (auto i = arith_para_->ndim_ - 1; i >= 0; --i) {
      if (arith_para_->in_shape0_[i] != arith_para_->in_shape1_[i]) {
        break_pos_ = i;
        break;
      }
    }
    in_size_ = 1;
    out_size_ = 1;
    for (size_t i = 0; i < arith_para_->ndim_; i++) {
      if (i > break_pos_) {
        in_size_ *= arith_para_->out_shape_[i];
      } else {
        out_size_ *= arith_para_->out_shape_[i];
      }
    }

    ComputeStrides(arith_para_->in_shape0_, arith_para_->in_strides0_, arith_para_->ndim_);
    ComputeStrides(arith_para_->in_shape1_, arith_para_->in_strides1_, arith_para_->ndim_);
    ComputeStrides(arith_para_->out_shape_, arith_para_->out_strides_, arith_para_->ndim_);
  }
  return RET_OK;
}

int AddInt8Run(void *cdata, int task_id) {
  auto add = reinterpret_cast<QuantizedAddCPUKernel *>(cdata);
  add->DoExecute(task_id);
  return RET_OK;
}

void QuantizedAddCPUKernel::BroadcastRun(int task_id) {
  int stride = UP_DIV(out_size_, thread_count_);
  int real_out_count = MSMIN(stride, out_size_ - stride * task_id);
  if (real_out_count <= 0) {
    return;
  }

  int8_t *const_in = arith_para_->in_elements_num0_ == arith_para_->out_elements_num_ ? input1_data_ : input0_data_;
  int8_t *offset_in = arith_para_->in_elements_num0_ == arith_para_->out_elements_num_ ? input0_data_ : input1_data_;
  offset_in += task_id * stride * in_size_;
  int8_t *cur_out = output_data_ + task_id * stride * in_size_;

  for (int i = 0; i < real_out_count; i++) {
    AddInt8(offset_in + i * in_size_, const_in, cur_out + i * in_size_, in_size_, &para_);
  }
  return;
}

int QuantizedAddCPUKernel::DoExecute(int task_id) {
  /* need broadcast */
  if (arith_para_->broadcasting_) {
    BroadcastRun(task_id);
    return RET_OK;
  }

  /* no need broadcast */
  int stride = UP_DIV(elements_num_, thread_count_);
  int rest_count = elements_num_ - task_id * stride;
  int real_count = MSMIN(stride, rest_count);
  if (real_count <= 0) {
    return RET_OK;
  }
  int8_t *cur_in0 = input0_data_ + stride * task_id;
  int8_t *cur_in1 = input1_data_ + stride * task_id;
  int8_t *cur_out = output_data_ + stride * task_id;
  if (support_opt_add_) {
    int8_t *ptr_in = arith_para_->in_elements_num0_ == 1 ? cur_in1 : cur_in0;
    int8_t element_in = arith_para_->in_elements_num0_ == 1 ? input0_data_[0] : input1_data_[0];
    AddOptInt8(ptr_in, element_in, cur_out, rest_count, &para_);
  } else {
    AddInt8(cur_in0, cur_in1, cur_out, rest_count, &para_);
  }

  return RET_OK;
}

int QuantizedAddCPUKernel::Run() {
  input0_data_ = static_cast<int8_t *>(in_tensors_.at(0)->data_c());
  input1_data_ = static_cast<int8_t *>(in_tensors_.at(1)->data_c());
  output_data_ = static_cast<int8_t *>(out_tensors_.at(0)->data_c());

  ParallelLaunch(this->context_->thread_pool_, AddInt8Run, this, thread_count_);

  return RET_OK;
}

kernel::LiteKernel *CpuAddInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                            const lite::InnerContext *ctx, const KernelKey &desc,
                                            const mindspore::lite::PrimitiveC *primitive) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is nullptr";
    return nullptr;
  }
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "ctx is nullptr";
    free(parameter);
    return nullptr;
  }
  MS_ASSERT(desc.type == PrimitiveType_Add);
  auto *kernel = new (std::nothrow) QuantizedAddCPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
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

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Add, CpuAddInt8KernelCreator)
}  // namespace mindspore::kernel
