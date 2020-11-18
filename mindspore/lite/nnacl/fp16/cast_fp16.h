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
#ifndef MINDSPORE_LITE_NNACL_CAST_FP16_H_
#define MINDSPORE_LITE_NNACL_CAST_FP16_H_

#include <arm_neon.h>
#include "nnacl/op_base.h"
#include "nnacl/fp32/cast.h"
#ifdef __cplusplus
extern "C" {
#endif
void BoolToFloat16(const bool *input, float16_t *output, int number);
void Uint8ToFloat16(const uint8_t *input, float16_t *output, int number);
void Float16ToInt32(const float16_t *input, int32_t *output, int number);
void Float16ToInt64(const float16_t *input, int64_t *output, int number);
void Float32ToFloat16(const float *input, float16_t *output, int number);
void Float16ToFloat32(const float16_t *input, float *output, int number);
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_NNACL_CAST_FP16_H_
