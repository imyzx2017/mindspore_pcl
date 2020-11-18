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

#ifndef MINDSPORE_LITE_NNACL_SPLITFP16_H_
#define MINDSPORE_LITE_NNACL_SPLITFP16_H_

#include <arm_neon.h>
#include "nnacl/op_base.h"
#include "nnacl/split_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
int DoSplitFp16(float16_t *in_data, float16_t **out_data, const int *input_shape, int offset, int num_unit,
                SplitParameter *split_param);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_SPLIT_H_
