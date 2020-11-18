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

#include "nnacl/fp32/conv.h"
#include <string.h>
#include "nnacl/fp32/common_func.h"
#include "nnacl/winograd_transform.h"
#include "nnacl/fp32/matmul.h"

// fp32 conv common
void ConvFp32(const float *input_data, float *packed_input, const float *packed_weight, const float *bias_data,
              float *col_major_input, float *output_data, int task_id, ConvParameter *conv_param) {
  int out_channel = conv_param->output_channel_;
  int deep = conv_param->kernel_h_ * conv_param->kernel_w_ * conv_param->input_channel_;
  int output_count = conv_param->output_h_ * conv_param->output_w_;
#if defined(ENABLE_ARM32) || defined(ENABLE_X86_64_SSE)
  const int cal_num = C4NUM;
#else
  const int cal_num = C12NUM;
#endif
  int output_tile_count = UP_DIV(output_count, cal_num);

  for (int b = 0; b < conv_param->input_batch_; b++) {
    int in_batch_offset = b * conv_param->input_channel_ * conv_param->input_h_ * conv_param->input_w_;
    int out_batch_offset = b * out_channel * output_count;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += conv_param->thread_num_) {
      int start_index = thread_id * cal_num;
      int real_cal_num = (output_count - start_index) < cal_num ? (output_count - start_index) : cal_num;
      float *gemm_input = packed_input + task_id * deep * cal_num;
      float *col_major_gemm_input = col_major_input + task_id * deep * cal_num;
      size_t packed_input_size = deep * cal_num * sizeof(float);
      memset(gemm_input, 0, packed_input_size);
      memset(col_major_gemm_input, 0, packed_input_size);
      Im2ColPackUnitFp32(input_data + in_batch_offset, conv_param, gemm_input, real_cal_num, start_index);

      int out_offset = thread_id * cal_num * out_channel + out_batch_offset;
      float *gemm_output = output_data + out_offset;
#if defined(ENABLE_ARM32) || defined(ENABLE_X86_64_SSE)
      RowMajor2Col4Major(gemm_input, col_major_gemm_input, cal_num, deep);
#else
      RowMajor2Col12Major(gemm_input, col_major_gemm_input, cal_num, deep);
#endif
      MatMulOpt(col_major_gemm_input, packed_weight, gemm_output, bias_data, conv_param->act_type_, deep, real_cal_num,
                out_channel, out_channel, OutType_Nhwc);
    }
  }
}

// fp32 conv winograd
void ConvWinogardFp32(const float *input_data, const float *trans_weight, const float *bias_data, float *output_data,
                      TmpBufferAddress *buffer_list, int task_id, ConvParameter *conv_param, InputTransFunc in_func,
                      OutputTransFunc out_func) {
  int in_channel = conv_param->input_channel_;
  int out_w_block = UP_DIV(conv_param->output_w_, conv_param->output_unit_);
  int out_h_block = UP_DIV(conv_param->output_h_, conv_param->output_unit_);
  int output_count = out_w_block * out_h_block;
  const int tile_num = C12NUM;
  int output_tile_count = UP_DIV(output_count, tile_num);
  int oc8 = UP_DIV(conv_param->output_channel_, C8NUM);
  int input_unit_square = conv_param->input_unit_ * conv_param->input_unit_;

  float *trans_input = buffer_list[0];
  float *gemm_out = buffer_list[1];
  float *tmp_data = buffer_list[2];
  float *col_buffer = buffer_list[3];
  int trans_input_offset = tile_num * input_unit_square * in_channel;
  int gemm_out_offset = tile_num * input_unit_square * oc8 * C8NUM;
  int tmp_data_offset = input_unit_square * C4NUM;
  int col_buffer_offset = tile_num * in_channel;
  // step 1 : filter transform (pre-processed offline)
  // step 2 : input transform (online)
  for (int b = 0; b < conv_param->input_batch_; b++) {
    int in_batch_offset = b * in_channel * conv_param->input_h_ * conv_param->input_w_;
    int out_batch_offset = b * conv_param->output_channel_ * conv_param->output_w_ * conv_param->output_h_;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += conv_param->thread_num_) {
      int out_tile_index = thread_id * tile_num;
      int cal_num = output_count - out_tile_index;
      cal_num = cal_num > tile_num ? tile_num : cal_num;
      WinogradInputTransform(input_data + in_batch_offset, trans_input + task_id * trans_input_offset,
                             tmp_data + task_id * tmp_data_offset, cal_num, out_tile_index, out_w_block, conv_param,
                             in_func);
      // step 3 : gemm
      float *src_ptr = trans_input + task_id * trans_input_offset;
      float *dst_ptr = gemm_out + task_id * gemm_out_offset;
      float *tmp_col_ptr = col_buffer + task_id * col_buffer_offset;
      for (int i = 0; i < input_unit_square; ++i) {
#if defined(ENABLE_ARM32) || defined(ENABLE_X86_64_SSE)
        RowMajor2Col4Major(src_ptr + i * C12NUM * in_channel, tmp_col_ptr, C12NUM, in_channel);
#else
        RowMajor2Col12Major(src_ptr + i * C12NUM * in_channel, tmp_col_ptr, C12NUM, in_channel);
#endif
        MatMulOpt(tmp_col_ptr, trans_weight + i * in_channel * oc8 * C8NUM, dst_ptr + i * C8NUM, NULL, 0, in_channel,
                  cal_num, oc8 * C8NUM, input_unit_square, 2);
      }

      // step 4 : output transform
      float *output_ptr = output_data + out_batch_offset;
      WinogradOutputTransform(dst_ptr, output_ptr, bias_data, cal_num, out_tile_index, out_w_block, conv_param,
                              out_func);
    }
  }
}
