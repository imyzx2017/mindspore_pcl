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

#ifndef MINDSPORE_LITE_NNACL_FP32_MATMUL_H_
#define MINDSPORE_LITE_NNACL_FP32_MATMUL_H_

#include <float.h>
#include <string.h>
#include "nnacl/errorcode.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif
void MatMulOpt(const float *a, const float *b, float *c, const float *bias, ActType act_type, int deep, int row,
               int col, size_t stride, int out_type);
void MatVecMul(const float *a, const float *b, float *c, const float *bias, ActType act_type, int depth, int col);
void RowMajor2ColMajor(const float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row4Major(float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row8Major(float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Row12Major(float *src_ptr, float *dst_ptr, int row, int col);
void RowMajor2Col4Major(float *src_ptr, float *dst_ptr, size_t row, size_t col);
void RowMajor2Col8Major(float *src_ptr, float *dst_ptr, size_t row, size_t col);
void RowMajor2Col12Major(float *src_ptr, float *dst_ptr, size_t row, size_t col);
#ifdef ENABLE_ARM64
void MatmulFloatNeon64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                       int col, size_t stride, size_t writeNhwc, size_t WriteWino);
void MatmulFloatNeon64Opt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                          int col, size_t stride, size_t write_mode);
void MatVecMulFp32Neon64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col);
#elif ENABLE_ARM32
void MatmulFloatNeon32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                       int col, int stride, size_t writeNhwc, size_t WriteWino);
void MatmulFloatNeon32Opt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                          int col, int stride, int write_mode);
#elif ENABLE_X86_64_SSE
void MatmulFloatSse64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                      int col, int stride, size_t writeNhwc, size_t WriteWino);
void MatmulFloatSse64Opt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                         int col, int stride, int write_mode);
#endif

#ifdef ENABLE_NNACL_INFER_SHAPE
int MatMulInferShape(int **in_shape, int in_num, size_t *dim_size, int *out_shape, int *in_format, int *out_format,
                     int *in_datatype, int *out_datatype, OpParameter *param);
#endif
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP32_MATMUL_H_
