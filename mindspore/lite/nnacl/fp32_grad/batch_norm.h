/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_NNACL_FP32_GRAD_BATCH_NORM_H_
#define MINDSPORE_LITE_NNACL_FP32_GRAD_BATCH_NORM_H_

#include "nnacl/op_base.h"

typedef struct BNGradParameter {
  OpParameter op_parameter_;
  float epsilon_;
  float momentum_;
} BNGradParameter;

#ifdef __cplusplus
extern "C" {
#endif

void sumSpatialBatch(const float *in, size_t size, int ch, float *out);
void backwardX(const float *in, const float *dout, const float *scale, const size_t size, int channels, float *mean,
               float *invar, float *xhat_sum, float *dxhat_sum, float *out);
void backwardScale(const float *x, const float *mean, const float *invar, const float *delta, int batch, int n,
                   int size, float *scale_updates);
void var2Invar(float *save_var, size_t size, float eps);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP32_GRAD_BATCH_NORM_H_
