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

#include "src/runtime/kernel/arm/fp32/fullconnection_fp32.h"
#include "src/runtime/runtime_api.h"
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
FullconnectionCPUKernel::~FullconnectionCPUKernel() {
  FreeBuf();
  return;
}

void FullconnectionCPUKernel::FreeBuf() {
  if (a_pack_ptr_ != nullptr) {
    free(a_pack_ptr_);
    a_pack_ptr_ = nullptr;
  }
  if (b_pack_ptr_ != nullptr) {
    free(b_pack_ptr_);
    b_pack_ptr_ = nullptr;
  }
  if (bias_ptr_ != nullptr) {
    free(bias_ptr_);
    bias_ptr_ = nullptr;
  }
}

int FullconnectionCPUKernel::ReSize() {
  FreeBuf();
  int row = 1;
  for (size_t i = 0; i < out_tensors_[0]->shape().size() - 1; ++i) row *= (out_tensors_[0]->shape())[i];
  fc_param_->row_ = row;
  fc_param_->col_ = out_tensors_[0]->shape().back();
  fc_param_->deep_ = (in_tensors_[1]->shape())[1];

  fc_param_->row_12_ = UP_ROUND(fc_param_->row_, C12NUM);
  fc_param_->col_8_ = UP_ROUND(fc_param_->col_, C8NUM);
  fc_param_->row_4_ = UP_ROUND(fc_param_->row_, C4NUM);

  thread_count_ = MSMIN(thread_count_, UP_DIV(fc_param_->col_8_, 8));
  thread_stride_ = UP_DIV(UP_DIV(fc_param_->col_8_, 8), thread_count_);

#ifdef ENABLE_ARM64
  if (fc_param_->row_ == 1) {
    is_vector_input_ = true;
  } else {
    is_vector_input_ = false;
  }
#endif
  if (in_tensors_.size() == 3) {
    int col_tmp = is_vector_input_ ? fc_param_->col_ : fc_param_->col_8_;
    bias_ptr_ = reinterpret_cast<float *>(malloc(col_tmp * sizeof(float)));
    if (bias_ptr_ == nullptr) {
      MS_LOG(ERROR) << "malloc bias_ptr_ failed";
      return RET_ERROR;
    }
    memcpy(bias_ptr_, in_tensors_[2]->MutableData(), fc_param_->col_ * sizeof(float));
  }

#if defined(ENABLE_ARM32) || defined(ENABLE_X86_64_SSE)
  a_pack_ptr_ = reinterpret_cast<float *>(malloc(fc_param_->row_4_ * fc_param_->deep_ * sizeof(float)));
  if (a_pack_ptr_ == nullptr) {
    return RET_MEMORY_FAILED;
  }
  memset(a_pack_ptr_, 0, fc_param_->row_4_ * fc_param_->deep_ * sizeof(float));
#else
  int row_tmp = is_vector_input_ ? 1 : fc_param_->row_12_;
  a_pack_ptr_ = reinterpret_cast<float *>(malloc(row_tmp * fc_param_->deep_ * sizeof(float)));
  if (a_pack_ptr_ == nullptr) {
    return RET_MEMORY_FAILED;
  }
  memset(a_pack_ptr_, 0, row_tmp * fc_param_->deep_ * sizeof(float));
#endif

  int col_tmp = is_vector_input_ ? fc_param_->col_ : fc_param_->col_8_;
  b_pack_ptr_ = reinterpret_cast<float *>(malloc(col_tmp * fc_param_->deep_ * sizeof(float)));
  if (b_pack_ptr_ == nullptr) {
    FreeBuf();
    return RET_MEMORY_FAILED;
  }
  memset(b_pack_ptr_, 0, col_tmp * fc_param_->deep_ * sizeof(float));

  fc_param_->a_const_ = (in_tensors_[0]->data_c() != nullptr);
  fc_param_->b_const_ = (in_tensors_[1]->data_c() != nullptr);
  if (fc_param_->a_const_) {
    InitMatrixA(reinterpret_cast<float *>(in_tensors_[0]->MutableData()), a_pack_ptr_);
    a_ptr_ = a_pack_ptr_;
  }
  if (fc_param_->b_const_) {
    InitMatrixB(reinterpret_cast<float *>(in_tensors_[1]->MutableData()), b_pack_ptr_);
    b_ptr_ = b_pack_ptr_;
  }
  return RET_OK;
}

int FullconnectionCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

void FullconnectionCPUKernel::InitMatrixA(float *src_ptr, float *dst_ptr) {
  if (is_vector_input_) {
    memcpy(dst_ptr, src_ptr, fc_param_->deep_ * sizeof(float));
    return;
  }

#if defined(ENABLE_ARM32) || defined(ENABLE_X86_64_SSE)
  RowMajor2Col4Major(src_ptr, a_pack_ptr_, fc_param_->row_, fc_param_->deep_);
#else
  RowMajor2Col12Major(src_ptr, a_pack_ptr_, fc_param_->row_, fc_param_->deep_);
#endif
}

void FullconnectionCPUKernel::InitMatrixB(float *src_ptr, float *dst_ptr) {
  if (is_vector_input_) {
    memcpy(dst_ptr, src_ptr, fc_param_->col_ * fc_param_->deep_ * sizeof(float));
    return;
  }

  RowMajor2Col8Major(src_ptr, dst_ptr, fc_param_->col_, fc_param_->deep_);
}

int FcFp32MatmulRun(void *cdata, int task_id) {
  auto fc = reinterpret_cast<FullconnectionCPUKernel *>(cdata);
  auto error_code = fc->DoMatmul(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "FcFp32MatmulRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int FullconnectionCPUKernel::DoMatmul(int task_id) {
  int cur_oc = MSMIN(thread_stride_ * C8NUM, fc_param_->col_ - task_id * thread_stride_ * C8NUM);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  auto b = b_ptr_ + task_id * thread_stride_ * C8NUM * fc_param_->deep_;
  auto bias = (bias_ptr_ == nullptr) ? nullptr : bias_ptr_ + task_id * thread_stride_ * C8NUM;
  auto c = c_ptr_ + task_id * thread_stride_ * C8NUM;
  if (is_vector_input_) {
    MatVecMul(a_ptr_, b, c, bias, fc_param_->act_type_, fc_param_->deep_, cur_oc);
  } else {
    MatMulOpt(a_ptr_, b, c, bias, fc_param_->act_type_, fc_param_->deep_, fc_param_->row_, cur_oc, fc_param_->col_,
              OutType_Nhwc);
  }

  return RET_OK;
}

int FullconnectionCPUKernel::Run() {
  auto a_ptr = reinterpret_cast<float *>(in_tensors_.at(0)->data_c());
  auto b_ptr = reinterpret_cast<float *>(in_tensors_.at(1)->data_c());
  c_ptr_ = reinterpret_cast<float *>(out_tensors_.at(0)->data_c());

  if (!fc_param_->a_const_) {
    if (is_vector_input_) {
      a_ptr_ = a_ptr;
    } else {
      InitMatrixA(a_ptr, a_pack_ptr_);
      a_ptr_ = a_pack_ptr_;
    }
  }
  if (!fc_param_->b_const_) {
    if (is_vector_input_) {
      b_ptr_ = b_ptr;
    } else {
      InitMatrixB(b_ptr, b_pack_ptr_);
      b_ptr_ = b_pack_ptr_;
    }
  }
  ParallelLaunch(this->context_->thread_pool_, FcFp32MatmulRun, this, thread_count_);

  return RET_OK;
}
}  // namespace mindspore::kernel
