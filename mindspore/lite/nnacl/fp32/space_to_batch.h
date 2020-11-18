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
#ifndef MINDSPORE_LITE_SRC_BACKEND_ARM_NNACL_FP32_SPACE_TO_BATCH_H_
#define MINDSPORE_LITE_SRC_BACKEND_ARM_NNACL_FP32_SPACE_TO_BATCH_H_
#include "nnacl/op_base.h"

typedef struct SpaceToBatchParameter {
  OpParameter op_parameter_;
  bool need_paddings_;
  int m_;
  int block_sizes_[4];
  int paddings_[4];
  int input_shape_[4];
  int output_shape_[4];
  int padded_in_shape_[4];
  int padded_input_element_num;
} SpaceToBatchParameter;
#ifdef __cplusplus
extern "C" {
#endif
void DoSpaceToBatchNHWC(const float *input, float *output, const int *block_sizes, const int *in_shape,
                        const int *out_shape);
void DoSpaceToBatchPaddingNHWC(const float *input, float *output, const int *in_shape, const int *padding,
                               const int *out_shape);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_SRC_BACKEND_ARM_NNACL_FP32_SPACE_TO_BATCH_H_
