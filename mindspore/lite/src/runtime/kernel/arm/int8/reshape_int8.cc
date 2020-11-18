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

#include "src/runtime/kernel/arm/int8/reshape_int8.h"
#include <limits>
#include "nnacl/int8/reshape_int8.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Reshape;

namespace mindspore::kernel {

int ReshapeInt8CPUKernel::Init() {
  ReshapeBaseCPUKernel::Init();
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto in_quant_args = input_tensor->GetQuantParams();
  reshape_param_->quant_para_.in_args_.scale_ = in_quant_args.front().scale;
  reshape_param_->quant_para_.in_args_.zp_ = in_quant_args.front().zeroPoint;

  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->GetQuantParams();
  reshape_param_->quant_para_.out_args_.scale_ = out_quant_args.front().scale;
  reshape_param_->quant_para_.out_args_.zp_ = out_quant_args.front().zeroPoint;

  reshape_param_->quant_para_.output_activation_min_ = std::numeric_limits<int8_t>::min();
  reshape_param_->quant_para_.output_activation_max_ = std::numeric_limits<int8_t>::max();

  return RET_OK;
}

int ReshapeInt8CPUKernel::ReSize() { return 0; }

int ReshapeInt8CPUKernel::Run() {
  MS_ASSERT(in_tensors_.size() == 1 || in_tensors_.size() == 2);
  MS_ASSERT(out_tensors_.size() == 1);
  input_data_ = static_cast<int8_t *>(in_tensors_.at(kInputIndex)->MutableData());
  output_data_ = static_cast<int8_t *>(out_tensors_.at(kOutputIndex)->MutableData());

  elements_num_ = in_tensors_.at(kInputIndex)->ElementsNum();
  count_unit_ = op_parameter_->thread_num_ > 1 ? UP_DIV(elements_num_, op_parameter_->thread_num_) : elements_num_;

  auto ret = ParallelLaunch(this->context_->thread_pool_, ReshapeInt8Run, this, op_parameter_->thread_num_);
  return ret;
}

int ReshapeInt8Run(void *cdata, int task_id) {
  auto reshape = reinterpret_cast<ReshapeInt8CPUKernel *>(cdata);
  reshape->DoExecute(task_id);
  return lite::RET_OK;
}

int ReshapeInt8CPUKernel::DoExecute(int task_id) {
  int64_t real_dst_count = MSMIN(elements_num_ - task_id * count_unit_, count_unit_);
  if (real_dst_count <= 0) {
    return lite::RET_OK;
  }
  int8_t *cur_input0_data = input_data_ + task_id * count_unit_;
  int8_t *cur_output_data = output_data_ + task_id * count_unit_;

  Int8Reshape(cur_input0_data, cur_output_data, real_dst_count, reshape_param_->quant_para_);
  return lite::RET_OK;
}
kernel::LiteKernel *CpuReshapeInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                const InnerContext *ctx, const kernel::KernelKey &desc,
                                                const mindspore::lite::PrimitiveC *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Reshape);
  auto *kernel = new (std::nothrow) ReshapeInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ReshapeInt8CPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Reshape, CpuReshapeInt8KernelCreator)

}  // namespace mindspore::kernel
