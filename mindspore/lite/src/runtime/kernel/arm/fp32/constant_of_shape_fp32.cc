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

#include "src/runtime/kernel/arm/fp32/constant_of_shape_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ConstantOfShape;

namespace mindspore::kernel {
int ConstantOfShapeCPUKernel::Init() { return RET_OK; }

int ConstantOfShapeCPUKernel::ReSize() { return RET_OK; }

int ConstantOfShapeCPUKernel::DoExecute(int task_id) {
  int ret = RET_ERROR;
  switch (param_->data_type_) {
    case kNumberTypeFloat32:
      ret = ConstantOfShape(reinterpret_cast<float *>(out_ptr_), task_id, param_);
      break;
    case kNumberTypeInt32:
      ret = ConstantOfShapeInt(reinterpret_cast<int32_t *>(out_ptr_), task_id, param_);
      break;
    default:
      MS_LOG(ERROR) << "Constant of shape does not support the output data type.";
      return RET_ERROR;
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConstantOfShapeRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int ConstantOfShapeRun(void *cdata, int task_id) {
  auto g_kernel = reinterpret_cast<ConstantOfShapeCPUKernel *>(cdata);
  auto ret = g_kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConstantOfShapeRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int ConstantOfShapeCPUKernel::Run() {
  param_->element_sz_ = out_tensors_.front()->ElementsNum();
  int thread_num = MSMIN(param_->op_parameter_.thread_num_, param_->element_sz_);
  param_->unit_ = UP_DIV(param_->element_sz_, thread_num);
  param_->op_parameter_.thread_num_ = thread_num;
  switch (param_->data_type_) {
    case kNumberTypeFloat32:
      out_ptr_ = reinterpret_cast<float *>(out_tensors_.front()->MutableData());
      break;
    case kNumberTypeInt32:
      out_ptr_ = reinterpret_cast<int32_t *>(out_tensors_.front()->MutableData());
      break;
    default:
      MS_LOG(ERROR) << "Constant of shape does not support the output data type.";
      return RET_ERROR;
  }
  auto ret = ParallelLaunch(this->context_->thread_pool_, ConstantOfShapeRun, this, thread_num);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConstantOfShapeRun error error_code[" << ret << "]";
    return ret;
  }
  return ret;
}

kernel::LiteKernel *CpuConstantOfShapeFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                        const std::vector<lite::Tensor *> &outputs,
                                                        OpParameter *opParameter, const lite::InnerContext *ctx,
                                                        const kernel::KernelKey &desc,
                                                        const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed, opParameter is nullptr, type: PrimitiveType_ConstantOfShape. ";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_ConstantOfShape);
  auto *kernel = new (std::nothrow) ConstantOfShapeCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ConstantOfShapeCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ConstantOfShape, CpuConstantOfShapeFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_ConstantOfShape, CpuConstantOfShapeFp32KernelCreator)
}  // namespace mindspore::kernel
