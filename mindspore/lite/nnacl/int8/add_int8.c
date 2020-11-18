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

#include "nnacl/int8/add_int8.h"
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/quantization/fixed_point.h"

void AddInt8(const int8_t *input0, const int8_t *input1, int8_t *output, int size, AddQuantParameter *params) {
  int in0_left_shift = (1 << params->left_shift_) * (1 << params->in0_left_shift_);
  int in1_left_shift = (1 << params->left_shift_) * (1 << params->in1_left_shift_);
  int index = 0;

#ifdef ENABLE_ARM
  const int8x16_t min_vec = vdupq_n_s8(params->min_);
  const int8x16_t max_vac = vdupq_n_s8(params->max_);

  const int16x8_t in0_zp_vec = vdupq_n_s16(params->in0_zp_);
  const int16x8_t in1_zp_vec = vdupq_n_s16(params->in1_zp_);
  const int16x8_t out_zp_vec = vdupq_n_s16(params->out_zp_);

  const int32x4_t in0_left_vec = vdupq_n_s32(in0_left_shift);
  const int32x4_t in1_left_vec = vdupq_n_s32(in1_left_shift);

  const int32x4_t in0_right_vec = vdupq_n_s32(-params->in0_right_shift_);
  const int32x4_t in1_right_vec = vdupq_n_s32(-params->in1_right_shift_);

  const int32x4_t out_left_vec = vdupq_n_s32(params->out_left_shift_);
  const int32x4_t out_right_vec = vdupq_n_s32(-params->out_right_shift_);

  for (; index <= size - 16; index += 16) {
    const int8x16_t in0_src = vld1q_s8(input0 + index);
    const int8x16_t in1_src = vld1q_s8(input1 + index);

    const int16x8_t in0_s16_low = vmovl_s8(vget_low_s8(in0_src));
    const int16x8_t in0_s16_high = vmovl_s8(vget_high_s8(in0_src));
    const int16x8_t in1_s16_low = vmovl_s8(vget_low_s8(in1_src));
    const int16x8_t in1_s16_high = vmovl_s8(vget_high_s8(in1_src));

    const int16x8_t in0_zp_low = vaddq_s16(in0_s16_low, in0_zp_vec);
    const int16x8_t in0_zp_high = vaddq_s16(in0_s16_high, in0_zp_vec);
    const int16x8_t in1_zp_low = vaddq_s16(in1_s16_low, in1_zp_vec);
    const int16x8_t in1_zp_high = vaddq_s16(in1_s16_high, in1_zp_vec);

    int32x4_t in0_1 = vmovl_s16(vget_low_s16(in0_zp_low));
    int32x4_t in0_2 = vmovl_s16(vget_high_s16(in0_zp_low));
    int32x4_t in0_3 = vmovl_s16(vget_low_s16(in0_zp_high));
    int32x4_t in0_4 = vmovl_s16(vget_high_s16(in0_zp_high));
    int32x4_t in1_1 = vmovl_s16(vget_low_s16(in1_zp_low));
    int32x4_t in1_2 = vmovl_s16(vget_high_s16(in1_zp_low));
    int32x4_t in1_3 = vmovl_s16(vget_low_s16(in1_zp_high));
    int32x4_t in1_4 = vmovl_s16(vget_high_s16(in1_zp_high));

    // Apply left shift
    in0_1 = vmulq_s32(in0_1, in0_left_vec);
    in0_2 = vmulq_s32(in0_2, in0_left_vec);
    in0_3 = vmulq_s32(in0_3, in0_left_vec);
    in0_4 = vmulq_s32(in0_4, in0_left_vec);
    in1_1 = vmulq_s32(in1_1, in1_left_vec);
    in1_2 = vmulq_s32(in1_2, in1_left_vec);
    in1_3 = vmulq_s32(in1_3, in1_left_vec);
    in1_4 = vmulq_s32(in1_4, in1_left_vec);

    // Apply the fixed-point part of the multiplier.
    in0_1 = vqrdmulhq_n_s32(in0_1, params->in0_multiplier_);
    in0_2 = vqrdmulhq_n_s32(in0_2, params->in0_multiplier_);
    in0_3 = vqrdmulhq_n_s32(in0_3, params->in0_multiplier_);
    in0_4 = vqrdmulhq_n_s32(in0_4, params->in0_multiplier_);
    in1_1 = vqrdmulhq_n_s32(in1_1, params->in1_multiplier_);
    in1_2 = vqrdmulhq_n_s32(in1_2, params->in1_multiplier_);
    in1_3 = vqrdmulhq_n_s32(in1_3, params->in1_multiplier_);
    in1_4 = vqrdmulhq_n_s32(in1_4, params->in1_multiplier_);

    // Apply right shift
    in0_1 = vqaddq_s32(in0_1, vshrq_n_s32(vandq_s32(in0_1, in0_right_vec), 31));
    in0_2 = vqaddq_s32(in0_2, vshrq_n_s32(vandq_s32(in0_2, in0_right_vec), 31));
    in0_3 = vqaddq_s32(in0_3, vshrq_n_s32(vandq_s32(in0_3, in0_right_vec), 31));
    in0_4 = vqaddq_s32(in0_4, vshrq_n_s32(vandq_s32(in0_4, in0_right_vec), 31));
    in1_1 = vqaddq_s32(in1_1, vshrq_n_s32(vandq_s32(in1_1, in1_right_vec), 31));
    in1_2 = vqaddq_s32(in1_2, vshrq_n_s32(vandq_s32(in1_2, in1_right_vec), 31));
    in1_3 = vqaddq_s32(in1_3, vshrq_n_s32(vandq_s32(in1_3, in1_right_vec), 31));
    in1_4 = vqaddq_s32(in1_4, vshrq_n_s32(vandq_s32(in1_4, in1_right_vec), 31));

    in0_1 = vrshlq_s32(in0_1, in0_right_vec);
    in0_2 = vrshlq_s32(in0_2, in0_right_vec);
    in0_3 = vrshlq_s32(in0_3, in0_right_vec);
    in0_4 = vrshlq_s32(in0_4, in0_right_vec);
    in1_1 = vrshlq_s32(in1_1, in1_right_vec);
    in1_2 = vrshlq_s32(in1_2, in1_right_vec);
    in1_3 = vrshlq_s32(in1_3, in1_right_vec);
    in1_4 = vrshlq_s32(in1_4, in1_right_vec);

    /* calculate output */
    int32x4_t out1 = vaddq_s32(in0_1, in1_1);
    int32x4_t out2 = vaddq_s32(in0_2, in1_2);
    int32x4_t out3 = vaddq_s32(in0_3, in1_3);
    int32x4_t out4 = vaddq_s32(in0_4, in1_4);

    // Apply left shift
    out1 = vshlq_s32(out1, out_left_vec);
    out2 = vshlq_s32(out2, out_left_vec);
    out3 = vshlq_s32(out3, out_left_vec);
    out4 = vshlq_s32(out4, out_left_vec);

    // Apply the fixed-point part of the multiplier.
    out1 = vqrdmulhq_n_s32(out1, params->out_multiplier_);
    out2 = vqrdmulhq_n_s32(out2, params->out_multiplier_);
    out3 = vqrdmulhq_n_s32(out3, params->out_multiplier_);
    out4 = vqrdmulhq_n_s32(out4, params->out_multiplier_);

    // Apply right shift
    out1 = vqaddq_s32(out1, vshrq_n_s32(vandq_s32(out1, out_right_vec), 31));
    out2 = vqaddq_s32(out2, vshrq_n_s32(vandq_s32(out2, out_right_vec), 31));
    out3 = vqaddq_s32(out3, vshrq_n_s32(vandq_s32(out3, out_right_vec), 31));
    out4 = vqaddq_s32(out4, vshrq_n_s32(vandq_s32(out4, out_right_vec), 31));

    out1 = vrshlq_s32(out1, out_right_vec);
    out2 = vrshlq_s32(out2, out_right_vec);
    out3 = vrshlq_s32(out3, out_right_vec);
    out4 = vrshlq_s32(out4, out_right_vec);

    const int16x4_t out1_s16 = vmovn_s32(out1);
    const int16x4_t out2_s16 = vmovn_s32(out2);
    const int16x4_t out3_s16 = vmovn_s32(out3);
    const int16x4_t out4_s16 = vmovn_s32(out4);

    const int16x8_t out_s16_1 = vaddq_s16(vcombine_s16(out1_s16, out2_s16), out_zp_vec);
    const int16x8_t out_s16_2 = vaddq_s16(vcombine_s16(out3_s16, out4_s16), out_zp_vec);

    const int8x16_t out = vcombine_s8(vqmovn_s16(out_s16_1), vqmovn_s16(out_s16_2));
    const int8x16_t int8_out = vmaxq_s8(min_vec, vminq_s8(max_vac, out));

    vst1q_s8(output + index, int8_out);
  }
#endif

  for (; index < size; index++) {
    const int32_t in0_left = (input0[index] + params->in0_zp_) * in0_left_shift;
    const int32_t in1_left = (input1[index] + params->in1_zp_) * in1_left_shift;
    const int32_t in0 = MultiplyByMultiplierAndRightShift(in0_left, params->in0_multiplier_, params->in0_right_shift_);
    const int32_t in1 = MultiplyByMultiplierAndRightShift(in1_left, params->in1_multiplier_, params->in1_right_shift_);

    int32_t out = MultiplyByQuantizedMultiplier(in0 + in1, params->out_multiplier_, params->out_left_shift_,
                                                -params->out_right_shift_);
    out += params->out_zp_;
    output[index] = (int8_t)MSMAX(params->min_, MSMIN(out, params->max_));
  }
  return;
}

void AddOptInt8(const int8_t *ptr_in, const int8_t element_in, int8_t *output, int size, AddQuantParameter *params) {
  int in0_left_shift = (1 << params->left_shift_) * (1 << params->in0_left_shift_);
  int in1_left_shift = (1 << params->left_shift_) * (1 << params->in1_left_shift_);
  int index = 0;

#ifdef ENABLE_ARM
  const int8x16_t in1_src = vdupq_n_s8(element_in);

  const int8x16_t min_vec = vdupq_n_s8(params->min_);
  const int8x16_t max_vac = vdupq_n_s8(params->max_);

  const int16x8_t in0_zp_vec = vdupq_n_s16(params->in0_zp_);
  const int16x8_t in1_zp_vec = vdupq_n_s16(params->in1_zp_);
  const int16x8_t out_zp_vec = vdupq_n_s16(params->out_zp_);

  const int32x4_t in0_left_vec = vdupq_n_s32(in0_left_shift);
  const int32x4_t in1_left_vec = vdupq_n_s32(in1_left_shift);

  const int32x4_t in0_right_vec = vdupq_n_s32(-params->in0_right_shift_);
  const int32x4_t in1_right_vec = vdupq_n_s32(-params->in1_right_shift_);

  const int32x4_t out_left_vec = vdupq_n_s32(params->out_left_shift_);
  const int32x4_t out_right_vec = vdupq_n_s32(-params->out_right_shift_);

  for (; index <= size - 16; index += 16) {
    const int8x16_t in0_src = vld1q_s8(ptr_in + index);

    const int16x8_t in0_s16_low = vmovl_s8(vget_low_s8(in0_src));
    const int16x8_t in0_s16_high = vmovl_s8(vget_high_s8(in0_src));
    const int16x8_t in1_s16_low = vmovl_s8(vget_low_s8(in1_src));
    const int16x8_t in1_s16_high = vmovl_s8(vget_high_s8(in1_src));

    const int16x8_t in0_zp_low = vaddq_s16(in0_s16_low, in0_zp_vec);
    const int16x8_t in0_zp_high = vaddq_s16(in0_s16_high, in0_zp_vec);
    const int16x8_t in1_zp_low = vaddq_s16(in1_s16_low, in1_zp_vec);
    const int16x8_t in1_zp_high = vaddq_s16(in1_s16_high, in1_zp_vec);

    int32x4_t in0_1 = vmovl_s16(vget_low_s16(in0_zp_low));
    int32x4_t in0_2 = vmovl_s16(vget_high_s16(in0_zp_low));
    int32x4_t in0_3 = vmovl_s16(vget_low_s16(in0_zp_high));
    int32x4_t in0_4 = vmovl_s16(vget_high_s16(in0_zp_high));
    int32x4_t in1_1 = vmovl_s16(vget_low_s16(in1_zp_low));
    int32x4_t in1_2 = vmovl_s16(vget_high_s16(in1_zp_low));
    int32x4_t in1_3 = vmovl_s16(vget_low_s16(in1_zp_high));
    int32x4_t in1_4 = vmovl_s16(vget_high_s16(in1_zp_high));

    // Apply left shift
    in0_1 = vmulq_s32(in0_1, in0_left_vec);
    in0_2 = vmulq_s32(in0_2, in0_left_vec);
    in0_3 = vmulq_s32(in0_3, in0_left_vec);
    in0_4 = vmulq_s32(in0_4, in0_left_vec);
    in1_1 = vmulq_s32(in1_1, in1_left_vec);
    in1_2 = vmulq_s32(in1_2, in1_left_vec);
    in1_3 = vmulq_s32(in1_3, in1_left_vec);
    in1_4 = vmulq_s32(in1_4, in1_left_vec);

    // Apply the fixed-point part of the multiplier.
    in0_1 = vqrdmulhq_n_s32(in0_1, params->in0_multiplier_);
    in0_2 = vqrdmulhq_n_s32(in0_2, params->in0_multiplier_);
    in0_3 = vqrdmulhq_n_s32(in0_3, params->in0_multiplier_);
    in0_4 = vqrdmulhq_n_s32(in0_4, params->in0_multiplier_);
    in1_1 = vqrdmulhq_n_s32(in1_1, params->in1_multiplier_);
    in1_2 = vqrdmulhq_n_s32(in1_2, params->in1_multiplier_);
    in1_3 = vqrdmulhq_n_s32(in1_3, params->in1_multiplier_);
    in1_4 = vqrdmulhq_n_s32(in1_4, params->in1_multiplier_);

    // Apply right shift
    in0_1 = vqaddq_s32(in0_1, vshrq_n_s32(vandq_s32(in0_1, in0_right_vec), 31));
    in0_2 = vqaddq_s32(in0_2, vshrq_n_s32(vandq_s32(in0_2, in0_right_vec), 31));
    in0_3 = vqaddq_s32(in0_3, vshrq_n_s32(vandq_s32(in0_3, in0_right_vec), 31));
    in0_4 = vqaddq_s32(in0_4, vshrq_n_s32(vandq_s32(in0_4, in0_right_vec), 31));
    in1_1 = vqaddq_s32(in1_1, vshrq_n_s32(vandq_s32(in1_1, in1_right_vec), 31));
    in1_2 = vqaddq_s32(in1_2, vshrq_n_s32(vandq_s32(in1_2, in1_right_vec), 31));
    in1_3 = vqaddq_s32(in1_3, vshrq_n_s32(vandq_s32(in1_3, in1_right_vec), 31));
    in1_4 = vqaddq_s32(in1_4, vshrq_n_s32(vandq_s32(in1_4, in1_right_vec), 31));

    in0_1 = vrshlq_s32(in0_1, in0_right_vec);
    in0_2 = vrshlq_s32(in0_2, in0_right_vec);
    in0_3 = vrshlq_s32(in0_3, in0_right_vec);
    in0_4 = vrshlq_s32(in0_4, in0_right_vec);
    in1_1 = vrshlq_s32(in1_1, in1_right_vec);
    in1_2 = vrshlq_s32(in1_2, in1_right_vec);
    in1_3 = vrshlq_s32(in1_3, in1_right_vec);
    in1_4 = vrshlq_s32(in1_4, in1_right_vec);

    /* calculate output */
    int32x4_t out1 = vaddq_s32(in0_1, in1_1);
    int32x4_t out2 = vaddq_s32(in0_2, in1_2);
    int32x4_t out3 = vaddq_s32(in0_3, in1_3);
    int32x4_t out4 = vaddq_s32(in0_4, in1_4);

    // Apply left shift
    out1 = vshlq_s32(out1, out_left_vec);
    out2 = vshlq_s32(out2, out_left_vec);
    out3 = vshlq_s32(out3, out_left_vec);
    out4 = vshlq_s32(out4, out_left_vec);

    // Apply the fixed-point part of the multiplier.
    out1 = vqrdmulhq_n_s32(out1, params->out_multiplier_);
    out2 = vqrdmulhq_n_s32(out2, params->out_multiplier_);
    out3 = vqrdmulhq_n_s32(out3, params->out_multiplier_);
    out4 = vqrdmulhq_n_s32(out4, params->out_multiplier_);

    // Apply right shift
    out1 = vqaddq_s32(out1, vshrq_n_s32(vandq_s32(out1, out_right_vec), 31));
    out2 = vqaddq_s32(out2, vshrq_n_s32(vandq_s32(out2, out_right_vec), 31));
    out3 = vqaddq_s32(out3, vshrq_n_s32(vandq_s32(out3, out_right_vec), 31));
    out4 = vqaddq_s32(out4, vshrq_n_s32(vandq_s32(out4, out_right_vec), 31));

    out1 = vrshlq_s32(out1, out_right_vec);
    out2 = vrshlq_s32(out2, out_right_vec);
    out3 = vrshlq_s32(out3, out_right_vec);
    out4 = vrshlq_s32(out4, out_right_vec);

    const int16x4_t out1_s16 = vmovn_s32(out1);
    const int16x4_t out2_s16 = vmovn_s32(out2);
    const int16x4_t out3_s16 = vmovn_s32(out3);
    const int16x4_t out4_s16 = vmovn_s32(out4);

    const int16x8_t out_s16_1 = vaddq_s16(vcombine_s16(out1_s16, out2_s16), out_zp_vec);
    const int16x8_t out_s16_2 = vaddq_s16(vcombine_s16(out3_s16, out4_s16), out_zp_vec);

    const int8x16_t out = vcombine_s8(vqmovn_s16(out_s16_1), vqmovn_s16(out_s16_2));
    const int8x16_t int8_out = vmaxq_s8(min_vec, vminq_s8(max_vac, out));

    vst1q_s8(output + index, int8_out);
  }
#endif

  for (; index < size; index++) {
    const int32_t in0_left = (ptr_in[index] + params->in0_zp_) * in0_left_shift;
    const int32_t in1_left = (element_in + params->in1_zp_) * in1_left_shift;
    const int32_t in0 = MultiplyByMultiplierAndRightShift(in0_left, params->in0_multiplier_, params->in0_right_shift_);
    const int32_t in1 = MultiplyByMultiplierAndRightShift(in1_left, params->in1_multiplier_, params->in1_right_shift_);

    int32_t out = MultiplyByQuantizedMultiplier(in0 + in1, params->out_multiplier_, params->out_left_shift_,
                                                -params->out_right_shift_);
    out += params->out_zp_;
    output[index] = (int8_t)MSMAX(params->min_, MSMIN(out, params->max_));
  }
  return;
}
