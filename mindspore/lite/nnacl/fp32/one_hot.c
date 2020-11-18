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

#include "nnacl/fp32/one_hot.h"
#include "nnacl/errorcode.h"

int OneHot(const int *indices, float *output, const OneHotParameter *one_hot_param, const int tid,
           const int thread_num) {
  if (indices == NULL || one_hot_param == NULL || output == NULL) {
    return NNACL_NULL_PTR;
  }

  int outer_size = one_hot_param->outer_size_;
  int inner_size = one_hot_param->inner_size_;
  int depth = one_hot_param->depth_;
  float on_value = one_hot_param->on_value_;
  float off_value = one_hot_param->off_value_;
  int i, j, k;
  for (i = tid; i < outer_size; i += thread_num) {
    float *output_ptr = output + i * depth * inner_size;
    for (k = 0; k < inner_size; k++) {
      int index = indices[i * inner_size + k];
      for (j = 0; j < depth; j++) {
        *output_ptr = off_value;
        if (index >= depth) {
          return NNACL_ERRCODE_INDEX_OUT_OF_RANGE;
        }
        if (index == j) {
          *output_ptr = on_value;
        }
        output_ptr++;
      }
    }
  }
  return NNACL_OK;
}
