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

#ifndef MINDSPORE_LITE_NNACL_FP16_TRANSPOSE_FP16_H_
#define MINDSPORE_LITE_NNACL_FP16_TRANSPOSE_FP16_H_

#include "nnacl/op_base.h"
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif

typedef struct TransposeParameter {
  OpParameter op_parameter_;
  int perm_[8];
  bool conjugate_;
  int num_axes_;
  int strides_[8];
  int out_strides_[8];
  int data_size_;
} TransposeParameter;

#ifdef __cplusplus
extern "C" {
#endif
int Fp16DoTranspose(float16_t *in_data, float16_t *out_data, int *input_shape, int *output_shape,
                    TransposeParameter *transpose_param, int h_start, int h_end);
void TransposeDim2(float16_t *in_data, float16_t *out_data, int *strides, int *out_strides, int *perm,
                   int *output_shape, int h_start, int h_end);
void TransposeDim3(float16_t *in_data, float16_t *out_data, int *strides, int *out_strides, int *perm,
                   int *output_shape, int h_start, int h_end);
void TransposeDim4(float16_t *in_data, float16_t *out_data, int *strides, int *out_strides, int *perm,
                   int *output_shape, int h_start, int h_end);
void TransposeDim5(float16_t *in_data, float16_t *out_data, int *strides, int *out_strides, int *perm,
                   int *output_shape, int h_start, int h_end);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP16_TRANSPOSE_FP16_H_
