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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP16_BATCHNORM_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP16_BATCHNORM_FP16_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/batchnorm_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

void BatchNormFp16(const float16_t *input, const void *mean, const void *variance, BatchNormParameter *param,
                   int task_id, float16_t *output);
void FusedBatchNormFp16(const void *input, const void *scale, const void *offset, const void *mean,
                        const void *variance, BatchNormParameter *param, int task_id, void *output);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP16_BATCHNORM_FP16_H_
