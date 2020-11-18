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

#include "minddata/dataset/kernels/image/lite_cv/image_process.h"

#include <string.h>
#include <cmath>
#include <vector>

namespace mindspore {
namespace dataset {

#define Equ(a, b) ((std::fabs((a) - (b)) < 1e-6))

static inline void InitBilinearWeight(int *data_ptr, int16_t *weight_ptr, double scale, int dst_length, int src_length,
                                      int a) {
  const int RESIZE_SCALE = 1 << 11;
  if (data_ptr == nullptr || weight_ptr == nullptr) {
    return;
  }

  int *data_start_ptr = data_ptr;
  int16_t *weight_start_ptr = weight_ptr;

  for (unsigned int i = 0; i < dst_length; i++) {
    float src_f_x = static_cast<float>((i + 0.5) * scale - 0.5);
    int src_u_x = static_cast<int>(floor(src_f_x));
    src_f_x -= src_u_x;
    if (src_u_x < 0) {
      src_u_x = 0;
      src_f_x = 0.0f;
    }
    if (src_u_x >= src_length - 1) {
      src_u_x = src_length - 2;
      src_f_x = 1.0f;
    }
    data_start_ptr[i] = src_u_x * a;
    int16_t t0 = INT16_CAST((1.0f - src_f_x) * RESIZE_SCALE);
    int16_t t1 = INT16_CAST(src_f_x * RESIZE_SCALE);

    weight_start_ptr[i * 2] = t0;
    weight_start_ptr[i * 2 + 1] = t1;
  }
}

static void ResizeBilinear3C(const unsigned char *src, int src_width, int src_height, unsigned char *dst, int dst_width,
                             int dst_height) {
  double scale_width = static_cast<double>(src_width) / dst_width;
  double scale_height = static_cast<double>(src_height) / dst_height;

  int *data_buf = new int[2 * dst_width + 2 * dst_height];

  int *x_offset = data_buf;
  int *y_offset = data_buf + dst_width;

  int16_t *x_weight = reinterpret_cast<int16_t *>(data_buf + dst_width + dst_height);
  int16_t *y_weight = reinterpret_cast<int16_t *>(x_weight + dst_width);

  InitBilinearWeight(x_offset, x_weight, scale_width, dst_width, src_width, 3);
  InitBilinearWeight(y_offset, y_weight, scale_height, dst_height, src_height, 1);

  LiteMat x_tmp_buf0(dst_width * 3 + 1, LDataType::UINT16);
  LiteMat x_tmp_buf1(dst_width * 3 + 1, LDataType::UINT16);
  int16_t *row0_ptr = reinterpret_cast<int16_t *>(x_tmp_buf0.data_ptr_);
  int16_t *row1_ptr = reinterpret_cast<int16_t *>(x_tmp_buf1.data_ptr_);

  int prev_height = -2;

  for (int y = 0; y < dst_height; y++) {
    int y_span = y_offset[y];

    if (y_span == prev_height) {
    } else if (y_span == prev_height + 1) {
      int16_t *tmp = row0_ptr;
      row0_ptr = row1_ptr;
      row1_ptr = tmp;
      const unsigned char *src_start = src + 3 * src_width * (y_span + 1);
      const int16_t *x_weight_p = x_weight;
      int16_t *row1_ptr1 = row1_ptr;
      for (int x = 0; x < dst_width; x++) {
        const unsigned char *src_start_p = src_start + x_offset[x];
        row1_ptr1[0] = (src_start_p[0] * x_weight_p[0] + src_start_p[3] * x_weight_p[1]) >> 4;
        row1_ptr1[1] = (src_start_p[1] * x_weight_p[0] + src_start_p[4] * x_weight_p[1]) >> 4;
        row1_ptr1[2] = (src_start_p[2] * x_weight_p[0] + src_start_p[5] * x_weight_p[1]) >> 4;
        x_weight_p += 2;
        row1_ptr1 += 3;
      }
    } else {
      const unsigned char *src0 = src + 3 * src_width * (y_span);
      const unsigned char *src1 = src + 3 * src_width * (y_span + 1);

      const int16_t *x_weight_ptr = x_weight;
      int16_t *row0_ptr0 = row0_ptr;
      int16_t *row1_ptr1 = row1_ptr;
      for (int x = 0; x < dst_width; x++) {
        const unsigned char *src0_ptr = src0 + x_offset[x];
        const unsigned char *src1_ptr = src1 + x_offset[x];

        for (int c = 0; c < 3; c++) {
          row0_ptr0[c] = (src0_ptr[c] * x_weight_ptr[0] + src0_ptr[c + 3] * x_weight_ptr[1]) >> 4;
          row1_ptr1[c] = (src1_ptr[c] * x_weight_ptr[0] + src1_ptr[c + 3] * x_weight_ptr[1]) >> 4;
        }

        x_weight_ptr += 2;
        row0_ptr0 += 3;
        row1_ptr1 += 3;
      }
    }
    prev_height = y_span;

    int16_t *row0_ptr0 = row0_ptr;
    int16_t *row1_ptr1 = row1_ptr;
    unsigned char *dst_ptr = dst + dst_width * 3 * (y);

    for (int k = 0; k < dst_width * 3; k++) {
      int16_t t0 = (int16_t)((y_weight[0] * (int16_t)(*row0_ptr0++)) >> 16);
      int16_t t1 = (int16_t)((y_weight[1] * (int16_t)(*row1_ptr1++)) >> 16);
      *dst_ptr++ = static_cast<unsigned char>((t0 + t1 + 2) >> 2);
    }
    y_weight += 2;
  }
  delete[] data_buf;
}

static void ResizeBilinear1C(const unsigned char *src, int src_width, int src_height, unsigned char *dst, int dst_width,
                             int dst_height) {
  double scale_width = static_cast<double>(src_width) / dst_width;
  double scale_height = static_cast<double>(src_height) / dst_height;

  int *data_buf = new int[2 * dst_width + 2 * dst_height];

  int *x_offset = data_buf;
  int *y_offset = data_buf + dst_width;

  int16_t *x_weight = reinterpret_cast<int16_t *>(data_buf + dst_width + dst_height);
  int16_t *y_weight = reinterpret_cast<int16_t *>(x_weight + dst_width);

  InitBilinearWeight(x_offset, x_weight, scale_width, dst_width, src_width, 1);
  InitBilinearWeight(y_offset, y_weight, scale_height, dst_height, src_height, 1);

  LiteMat x_tmp_buf0(dst_width, LDataType::UINT16);
  LiteMat x_tmp_buf1(dst_width, LDataType::UINT16);
  int16_t *row0_ptr = reinterpret_cast<int16_t *>(x_tmp_buf0.data_ptr_);
  int16_t *row1_ptr = reinterpret_cast<int16_t *>(x_tmp_buf1.data_ptr_);

  int prev_height = -2;

  for (int y = 0; y < dst_height; y++) {
    int y_span = y_offset[y];

    if (y_span == prev_height) {
    } else if (y_span == prev_height + 1) {
      int16_t *tmp = row0_ptr;
      row0_ptr = row1_ptr;
      row1_ptr = tmp;
      const unsigned char *src_start = src + src_width * (y_span + 1);
      const int16_t *x_weight_p = x_weight;
      int16_t *row1_ptr1 = row1_ptr;
      for (int x = 0; x < dst_width; x++) {
        const unsigned char *src_start_p = src_start + x_offset[x];
        row1_ptr1[x] = (src_start_p[0] * x_weight_p[0] + src_start_p[3] * x_weight_p[1]) >> 4;
        x_weight_p += 2;
      }
    } else {
      const unsigned char *src0 = src + src_width * (y_span);
      const unsigned char *src1 = src + src_width * (y_span + 1);

      const int16_t *x_weight_ptr = x_weight;
      int16_t *row0_ptr0 = row0_ptr;
      int16_t *row1_ptr1 = row1_ptr;
      for (int x = 0; x < dst_width; x++) {
        const unsigned char *src0_ptr = src0 + x_offset[x];
        const unsigned char *src1_ptr = src1 + x_offset[x];

        row0_ptr0[x] = (src0_ptr[0] * x_weight_ptr[0] + src0_ptr[3] * x_weight_ptr[1]) >> 4;
        row1_ptr1[x] = (src1_ptr[0] * x_weight_ptr[0] + src1_ptr[3] * x_weight_ptr[1]) >> 4;

        x_weight_ptr += 2;
      }
    }
    prev_height = y_span;

    int16_t *row0_ptr0 = row0_ptr;
    int16_t *row1_ptr1 = row1_ptr;
    unsigned char *dst_ptr = dst + dst_width * (y);

    for (int k = 0; k < dst_width; k++) {
      int16_t t0 = (int16_t)((y_weight[0] * (int16_t)(*row0_ptr0++)) >> 16);
      int16_t t1 = (int16_t)((y_weight[1] * (int16_t)(*row1_ptr1++)) >> 16);
      *dst_ptr++ = static_cast<unsigned char>((t0 + t1 + 2) >> 2);
    }

    y_weight += 2;
  }
  delete[] data_buf;
}

bool ResizeBilinear(const LiteMat &src, LiteMat &dst, int dst_w, int dst_h) {
  if (dst_h <= 0 || dst_w <= 0) {
    return false;
  }
  if (src.data_type_ != LDataType::UINT8) {
    return false;
  }
  if (src.channel_ == 3) {
    (void)dst.Init(dst_w, dst_h, 3, LDataType::UINT8);
    const unsigned char *src_start_p = src;
    unsigned char *dst_start_p = dst;
    (void)ResizeBilinear3C(src_start_p, src.width_, src.height_, dst_start_p, dst_w, dst_h);
  } else if (src.channel_ == 1) {
    (void)dst.Init(dst_w, dst_h, 1, LDataType::UINT8);
    const unsigned char *src_start_p = src;
    unsigned char *dst_start_p = dst;
    (void)ResizeBilinear1C(src_start_p, src.width_, src.height_, dst_start_p, dst_w, dst_h);
  } else {
    return false;
  }
  return true;
}

static bool ConvertBGR(const unsigned char *data, LDataType data_type, int w, int h, LiteMat &mat) {
  if (data_type == LDataType::UINT8) {
    mat.Init(w, h, 3, LDataType::UINT8);
    unsigned char *dst_ptr = mat;
    (void)memcpy(dst_ptr, data, w * h * 3 * sizeof(unsigned char));
  } else {
    return false;
  }
  return true;
}

static bool ConvertRGBAToBGR(const unsigned char *data, LDataType data_type, int w, int h, LiteMat &mat) {
  if (data_type == LDataType::UINT8) {
    mat.Init(w, h, 3, LDataType::UINT8);
    unsigned char *ptr = mat;
    const unsigned char *data_ptr = data;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        ptr[0] = data_ptr[2];
        ptr[1] = data_ptr[1];
        ptr[2] = data_ptr[0];
        ptr += 3;
        data_ptr += 4;
      }
    }
  } else {
    return false;
  }
  return true;
}

static bool ConvertRGBAToRGB(const unsigned char *data, LDataType data_type, int w, int h, LiteMat &mat) {
  if (data_type == LDataType::UINT8) {
    mat.Init(w, h, 3, LDataType::UINT8);
    unsigned char *ptr = mat;
    const unsigned char *data_ptr = data;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        ptr[0] = data_ptr[0];
        ptr[1] = data_ptr[1];
        ptr[2] = data_ptr[2];
        ptr += 3;
        data_ptr += 4;
      }
    }
  } else {
    return false;
  }
  return true;
}

static bool ConvertYUV420SPToBGR(const uint8_t *data, LDataType data_type, bool flag, int w, int h, LiteMat &mat) {
  if (data == nullptr || w <= 0 || h <= 0) {
    return false;
  }
  if (data_type == LDataType::UINT8) {
    mat.Init(w, h, 3, LDataType::UINT8);
    const uint8_t *y_ptr = data;
    const uint8_t *uv_ptr = y_ptr + w * h;
    uint8_t *bgr_ptr = mat;
    int bgr_stride = 3 * w;

    for (int y = 0; y < h; ++y) {
      uint8_t *bgr_buf = bgr_ptr;
      const uint8_t *uv_buf = uv_ptr;
      const uint8_t *y_buf = y_ptr;
      uint8_t u;
      uint8_t v;
      for (int x = 0; x < w - 1; x += 2) {
        if (flag) {
          // NV21
          u = uv_buf[1];
          v = uv_buf[0];
        } else {
          // NV12
          u = uv_buf[0];
          v = uv_buf[1];
        }
        uint32_t tmp_y = (uint32_t)(y_buf[0] * YSCALE * YTOG) >> 16;
        // b
        bgr_buf[0] = std::clamp((int32_t)(-(u * UTOB) + tmp_y + BTOB) >> 6, 0, 255);
        // g
        bgr_buf[1] = std::clamp((int32_t)(-(u * UTOG + v * VTOG) + tmp_y + BTOG) >> 6, 0, 255);
        // r
        bgr_buf[2] = std::clamp((int32_t)(-(v * VTOR) + tmp_y + BTOR) >> 6, 0, 255);

        tmp_y = (uint32_t)(y_buf[1] * YSCALE * YTOG) >> 16;
        bgr_buf[3] = std::clamp((int32_t)(-(u * UTOB) + tmp_y + BTOB) >> 6, 0, 255);
        bgr_buf[4] = std::clamp((int32_t)(-(u * UTOG + v * VTOG) + tmp_y + BTOG) >> 6, 0, 255);
        bgr_buf[5] = std::clamp((int32_t)(-(v * VTOR) + tmp_y + BTOR) >> 6, 0, 255);

        y_buf += 2;
        uv_buf += 2;
        bgr_buf += 6;
      }
      if (w & 1) {
        if (flag) {
          // NV21
          u = uv_buf[1];
          v = uv_buf[0];
        } else {
          // NV12
          u = uv_buf[0];
          v = uv_buf[1];
        }
        uint32_t tmp_y = (uint32_t)(y_buf[0] * YSCALE * YTOG) >> 16;
        bgr_buf[0] = std::clamp((int32_t)(-(u * UTOB) + tmp_y + BTOB) >> 6, 0, 255);
        bgr_buf[1] = std::clamp((int32_t)(-(u * UTOG + v * VTOG) + tmp_y + BTOG) >> 6, 0, 255);
        bgr_buf[2] = std::clamp((int32_t)(-(v * VTOR) + tmp_y + BTOR) >> 6, 0, 255);
      }

      bgr_ptr += bgr_stride;
      y_ptr += w;
      if (y & 1) {
        uv_ptr += w;
      }
    }
  }
  return true;
}

static bool ConvertRGBAToGRAY(const unsigned char *data, LDataType data_type, int w, int h, LiteMat &mat) {
  if (data_type == LDataType::UINT8) {
    mat.Init(w, h, 1, LDataType::UINT8);
    if (mat.IsEmpty()) {
      return false;
    }
    unsigned char *ptr = mat;
    const unsigned char *data_ptr = data;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        *ptr = (data_ptr[2] * B2GRAY + data_ptr[1] * G2GRAY + data_ptr[0] * R2GRAY) >> GRAYSHIFT;
        ptr++;
        data_ptr += 4;
      }
    }
  } else {
    return false;
  }
  return true;
}

bool InitFromPixel(const unsigned char *data, LPixelType pixel_type, LDataType data_type, int w, int h, LiteMat &m) {
  if (data == nullptr) {
    return false;
  }
  if (w <= 0 || h <= 0) {
    return false;
  }

  if (data_type != LDataType::UINT8) {
    return false;
  }
  if (pixel_type == LPixelType::RGBA2BGR) {
    return ConvertRGBAToBGR(data, data_type, w, h, m);
  } else if (pixel_type == LPixelType::RGBA2GRAY) {
    return ConvertRGBAToGRAY(data, data_type, w, h, m);
  } else if (pixel_type == LPixelType::RGBA2RGB) {
    return ConvertRGBAToRGB(data, data_type, w, h, m);
  } else if (pixel_type == LPixelType::NV212BGR) {
    return ConvertYUV420SPToBGR(data, data_type, true, w, h, m);
  } else if (pixel_type == LPixelType::NV122BGR) {
    return ConvertYUV420SPToBGR(data, data_type, false, w, h, m);
  } else if (pixel_type == LPixelType::BGR) {
    return ConvertBGR(data, data_type, w, h, m);
  } else if (pixel_type == LPixelType::RGB) {
    return ConvertBGR(data, data_type, w, h, m);
  } else {
    return false;
  }
  return true;
}

bool ConvertTo(const LiteMat &src, LiteMat &dst, double scale) {
  if (src.data_type_ != LDataType::UINT8) {
    return false;
  }

  (void)dst.Init(src.width_, src.height_, src.channel_, LDataType::FLOAT32);
  const unsigned char *src_start_p = src;
  float *dst_start_p = dst;
  for (int h = 0; h < src.height_; h++) {
    for (int w = 0; w < src.width_; w++) {
      uint32_t index = (h * src.width_ + w) * src.channel_;
      for (int c = 0; c < src.channel_; c++) {
        dst_start_p[index + c] = (static_cast<float>(src_start_p[index + c] * scale));
      }
    }
  }
  return true;
}

template <typename T>
static void CropInternal(const LiteMat &src, LiteMat &dst, int x, int y, int w, int h) {
  int dst_h = h;
  int dst_w = w;
  int dst_c = src.channel_;
  dst.Init(dst_w, dst_h, dst_c, src.data_type_);
  const T *src_start_p = src;
  T *dst_start_p = dst;
  for (int i_h = 0; i_h < dst_h; i_h++) {
    const T *src_index_p = src_start_p + (y + i_h) * src.width_ * dst_c + x * dst_c;
    T *dst_index_p = dst_start_p + i_h * dst_w * dst_c;
    (void)memcpy(dst_index_p, src_index_p, dst_w * dst_c * sizeof(T));
  }
}

bool Crop(const LiteMat &src, LiteMat &dst, int x, int y, int w, int h) {
  if (x < 0 || y < 0 || w <= 0 || h <= 0) {
    return false;
  }
  if (y + h > src.height_ || x + w > src.width_) {
    return false;
  }

  if (src.data_type_ == LDataType::UINT8) {
    CropInternal<uint8_t>(src, dst, x, y, w, h);
  } else if (src.data_type_ == LDataType::FLOAT32) {
    CropInternal<float>(src, dst, x, y, w, h);
  } else {
    return false;
  }
  return true;
}

static bool CheckZero(const std::vector<float> &vs) {
  for (int i = 0; i < vs.size(); i++) {
    if (Equ(vs[i], 0.0f)) {
      return true;
    }
  }
  return false;
}

static bool CheckZero(const std::vector<size_t> &vs) {
  for (int i = 0; i < vs.size(); i++) {
    if (vs[i] == 0) return true;
  }
  return false;
}

static bool CheckMeanAndStd(int channel, const std::vector<float> &mean, const std::vector<float> &std) {
  if (mean.size() == 0 && std.size() == 0) {
    return false;
  }
  if (mean.size() > 0) {
    if (CheckZero(mean)) {
      return false;
    }
    if (mean.size() != channel) {
      return false;
    }
  }
  if (std.size() > 0) {
    if (CheckZero(std)) {
      return false;
    }
    if (std.size() != channel) {
      return false;
    }
  }
  return true;
}
bool SubStractMeanNormalize(const LiteMat &src, LiteMat &dst, const std::vector<float> &mean,
                            const std::vector<float> &std) {
  if (src.data_type_ != LDataType::FLOAT32) {
    return false;
  }
  if (!CheckMeanAndStd(src.channel_, mean, std)) {
    return false;
  }

  dst.Init(src.width_, src.height_, src.channel_, LDataType::FLOAT32);

  const float *src_start_p = src;
  float *dst_start_p = dst;
  if ((!mean.empty()) && std.empty()) {
    for (int h = 0; h < src.height_; h++) {
      for (int w = 0; w < src.width_; w++) {
        uint32_t src_start = (h * src.width_ + w) * src.channel_;
        for (int c = 0; c < src.channel_; c++) {
          uint32_t index = src_start + c;
          dst_start_p[index] = src_start_p[index] - mean[c];
        }
      }
    }
  } else if (mean.empty() && (!std.empty())) {
    for (int h = 0; h < src.height_; h++) {
      for (int w = 0; w < src.width_; w++) {
        uint32_t src_start = (h * src.width_ + w) * src.channel_;
        for (int c = 0; c < src.channel_; c++) {
          uint32_t index = src_start + c;
          dst_start_p[index] = src_start_p[index] / std[c];
        }
      }
    }
  } else if ((!mean.empty()) && (!std.empty())) {
    for (int h = 0; h < src.height_; h++) {
      for (int w = 0; w < src.width_; w++) {
        uint32_t src_start = (h * src.width_ + w) * src.channel_;
        for (int c = 0; c < src.channel_; c++) {
          uint32_t index = src_start + c;
          dst_start_p[index] = (src_start_p[index] - mean[c]) / std[c];
        }
      }
    }
  } else {
    return false;
  }
  return true;
}

template <typename T>
static void PadWithConstant(const LiteMat &src, LiteMat &dst, const int top, const int bottom, const int left,
                            const int right, const PaddBorderType pad_type, uint8_t fill_b_or_gray, uint8_t fill_g,
                            uint8_t fill_r) {
  std::vector<uint8_t> row_buffer(dst.width_ * dst.channel_);
  uint8_t *const_ptr = row_buffer.data();
  int src_step = src.width_ * dst.channel_;
  int dst_step = dst.width_ * dst.channel_;
  if (dst.channel_ == 1) {
    for (int i = 0; i < dst_step; i++) {
      const_ptr[i] = fill_b_or_gray;
    }
  } else if (dst.channel_ == 3) {
    for (int i = 0; i < dst.width_; i++) {
      const_ptr[i * dst.channel_] = fill_b_or_gray;
      const_ptr[i * dst.channel_ + 1] = fill_g;
      const_ptr[i * dst.channel_ + 2] = fill_r;
    }
  }

  uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(dst.data_ptr_);
  uint8_t *src_ptr = reinterpret_cast<uint8_t *>(src.data_ptr_);
  for (int i = 0; i < top; i++) {
    memcpy(dst_ptr + i * dst_step, const_ptr, dst_step);
  }

  int left_size = left * dst.channel_;
  int right_size = right * dst.channel_;
  uint8_t *dst_raw_data = dst_ptr + top * dst_step + left_size;
  for (int i = 0; i < src.width_; i++, dst_raw_data += dst_step, src_ptr += src_step) {
    memcpy(dst_raw_data, src_ptr, src_step);
    memcpy(dst_raw_data - left_size, const_ptr, left_size);
    memcpy(dst_raw_data + src_step, const_ptr, right_size);
  }

  for (int i = dst.height_ - bottom; i < dst.height_; i++) {
    memcpy(dst_ptr + i * dst_step, const_ptr, dst_step);
  }
}

bool ExtractChannel(const LiteMat &src, LiteMat &dst, int col) {
  if (src.IsEmpty() || col < 0 || col > src.channel_ - 1) {
    return false;
  }
  if (src.data_type_ == LDataType::FLOAT32) {
    (void)dst.Init(src.width_, src.height_, 1, src.data_type_);
    const float *src_start_p = src;
    float *dst_start_p = dst;
    for (int h = 0; h < src.height_; h++) {
      uint32_t src_start = h * src.width_ * src.channel_ + col;
      uint32_t dst_start = h * dst.width_;
      for (int w = 0; w < src.width_; w++) {
        uint32_t src_index = src_start + w * src.channel_;
        uint32_t dst_index = dst_start + w;
        dst_start_p[dst_index] = src_start_p[src_index];
      }
    }
    return true;
  } else if (src.data_type_ == LDataType::UINT8) {
    (void)dst.Init(src.width_, src.height_, 1, src.data_type_);
    const uint8_t *src_start_p = src;
    uint8_t *dst_start_p = dst;
    for (int h = 0; h < src.height_; h++) {
      uint32_t src_start = h * src.width_ * src.channel_ + col;
      uint32_t dst_start = h * dst.width_;
      for (int w = 0; w < src.width_; w++) {
        uint32_t src_index = src_start + w * src.channel_;
        uint32_t dst_index = dst_start + w;
        dst_start_p[dst_index] = src_start_p[src_index];
      }
    }
    return true;
  } else {
    return false;
  }
  return false;
}

bool Split(const LiteMat &src, std::vector<LiteMat> &mv) {
  if (src.data_type_ == LDataType::FLOAT32) {
    const float *src_start_p = src;
    for (int c = 0; c < src.channel_; c++) {
      LiteMat dst;
      (void)dst.Init(src.width_, src.height_, 1, src.data_type_);
      float *dst_start_p = dst;
      for (int h = 0; h < src.height_; h++) {
        uint32_t src_start = h * src.width_ * src.channel_;
        uint32_t dst_start = h * dst.width_;
        for (int w = 0; w < src.width_; w++) {
          uint32_t src_index = src_start + w * src.channel_ + c;
          uint32_t dst_index = dst_start + w;
          dst_start_p[dst_index] = src_start_p[src_index];
        }
      }
      mv.emplace_back(dst);
    }
    return true;
  } else if (src.data_type_ == LDataType::UINT8) {
    const uint8_t *src_start_p = src;
    for (int c = 0; c < src.channel_; c++) {
      LiteMat dst;
      (void)dst.Init(src.width_, src.height_, 1, src.data_type_);
      uint8_t *dst_start_p = dst;
      for (int h = 0; h < src.height_; h++) {
        uint32_t src_start = h * src.width_ * src.channel_;
        uint32_t dst_start = h * dst.width_;
        for (int w = 0; w < src.width_; w++) {
          uint32_t src_index = src_start + w * src.channel_ + c;
          uint32_t dst_index = dst_start + w;
          dst_start_p[dst_index] = src_start_p[src_index];
        }
      }
      mv.emplace_back(dst);
    }
    return true;
  } else {
    return false;
  }
  return false;
}

template <typename T>
inline void MergeImpl(const std::vector<LiteMat> &mv, T *dst_ptr, int height, int width, int channel) {
  T *mv_ptr[4];
  int area = height * width;
  for (int c = 0; c < channel; c++) {
    mv_ptr[c] = reinterpret_cast<T *>(mv[c].data_ptr_);
  }
  for (int i = 0; i < area; i++) {
    for (int c = 0; c < channel; c++) {
      dst_ptr[c] = *mv_ptr[c];
      mv_ptr[c]++;
    }
    dst_ptr += channel;
  }
}

bool Merge(const std::vector<LiteMat> &mv, LiteMat &dst) {
  if (mv.size() != 1 && mv.size() != 3 && mv.size() != 4) return false;

  int width = mv[0].width_;
  int height = mv[0].height_;
  int channel = mv.size();
  LDataType data_type = mv[0].data_type_;

  // The arrays in list must be single-channel
  for (int i = 0; i < mv.size(); i++) {
    if (mv[i].channel_ != 1) return false;
  }

  for (int i = 1; i < mv.size(); i++) {
    if (width != mv[i].width_ || height != mv[i].height_ || data_type != mv[i].data_type_) {
      return false;
    }
  }

  if (dst.IsEmpty() || dst.width_ != width || dst.height_ != height || dst.channel_ != channel ||
      dst.data_type_ != data_type) {
    dst.Init(width, height, channel, data_type);
  }

  if (dst.data_type_ == LDataType::FLOAT32) {
    MergeImpl<float>(mv, dst, height, width, channel);
  } else if (dst.data_type_ == LDataType::UINT8) {
    MergeImpl<uint8_t>(mv, dst, height, width, channel);
  } else {
    return false;
  }
  return true;
}

bool Pad(const LiteMat &src, LiteMat &dst, int top, int bottom, int left, int right, PaddBorderType pad_type,
         uint8_t fill_b_or_gray, uint8_t fill_g, uint8_t fill_r) {
  if (top < 0 || bottom <= 0 || left < 0 || right < 0) {
    return false;
  }
  if (src.IsEmpty()) {
    return false;
  }
  int dst_width = src.width_ + left + right;
  int dst_height = src.height_ + top + bottom;
  if (dst.IsEmpty()) {
    dst.Init(dst_width, dst_height, src.channel_, src.data_type_);
  } else if (dst.width_ != dst_width || dst.height_ != dst_height || src.channel_ != dst.channel_) {
    return false;
  } else if (src.data_type_ != dst.data_type_) {
    return false;
  }
  if (pad_type == PADD_BORDER_CONSTANT && src.data_type_ == LDataType::FLOAT32) {
    PadWithConstant<float>(src, dst, top, bottom, left, right, pad_type, fill_b_or_gray, fill_g, fill_r);
  } else if (pad_type == PADD_BORDER_CONSTANT && src.data_type_ == LDataType::UINT8) {
    PadWithConstant<uint8_t>(src, dst, top, bottom, left, right, pad_type, fill_b_or_gray, fill_g, fill_r);
  } else {
    return false;
  }
  return true;
}

std::vector<std::vector<float>> GetDefaultBoxes(BoxesConfig config) {
  std::vector<float> fk;
  float num = static_cast<float>(config.img_shape[0]);
  for (int i = 0; i < config.steps.size(); i++) {
    fk.push_back(num / config.steps[i]);
  }
  float scale_rate = (config.max_scale - config.min_scale) / (config.num_default.size() - 1);
  std::vector<float> scales(config.num_default.size());
  for (int i = 0; i < scales.size(); i++) {
    scales[i] = config.min_scale + scale_rate * i;
  }
  scales.push_back(1.0f);
  std::vector<std::vector<float>> default_boxes;
  for (int i = 0; i < config.feature_size.size(); i++) {
    float sk1 = scales[i];
    float sk2 = scales[i + 1];
    float sk3 = sqrt(sk1 * sk2);
    std::vector<std::vector<float>> all_sizes;
    float w, h;
    if (i == 0) {
      w = sk1 * sqrt(2);
      h = sk1 / sqrt(2);
      all_sizes = {{0.1, 0.1}, {w, h}, {h, w}};
    } else {
      all_sizes = {{sk1, sk1}};
      for (int j = 0; j < config.aspect_rations[i].size(); j++) {
        w = sk1 * sqrt(config.aspect_rations[i][j]);
        h = sk1 / sqrt(config.aspect_rations[i][j]);
        all_sizes.push_back({w, h});
        all_sizes.push_back({h, w});
      }
      all_sizes.push_back({sk3, sk3});
    }

    for (int j = 0; j < config.feature_size[i]; j++) {
      for (int k = 0; k < config.feature_size[i]; k++) {
        for (int m = 0; m < all_sizes.size(); m++) {
          float cx = (k + 0.5) / fk[i];
          float cy = (j + 0.5) / fk[i];
          default_boxes.push_back({cy, cx, all_sizes[m][1], all_sizes[m][0]});
        }
      }
    }
  }
  return default_boxes;
}

void ConvertBoxes(std::vector<std::vector<float>> &boxes, const std::vector<std::vector<float>> &default_boxes,
                  const BoxesConfig config) {
  for (int i = 0; i < default_boxes.size(); i++) {
    boxes[i][0] = boxes[i][0] * config.prior_scaling[0] * default_boxes[i][2] + default_boxes[i][0];
    boxes[i][1] = boxes[i][1] * config.prior_scaling[0] * default_boxes[i][3] + default_boxes[i][1];
    boxes[i][2] = exp(boxes[i][2] * config.prior_scaling[1]) * default_boxes[i][2];
    boxes[i][3] = exp(boxes[i][3] * config.prior_scaling[1]) * default_boxes[i][3];
  }
}

std::vector<int> ApplyNms(const std::vector<std::vector<float>> &all_boxes, std::vector<float> &all_scores, float thres,
                          int max_boxes) {
  int boxes_num = all_boxes.size();
  std::vector<float> areas(boxes_num);
  std::vector<int> order(boxes_num);
  for (int i = 0; i < boxes_num; i++) {
    areas[i] = (all_boxes[i][3] - all_boxes[i][1] + 1) * (all_boxes[i][2] - all_boxes[i][0] + 1);
    order[i] = i;
  }

  std::sort(order.begin(), order.end(),
            [&all_scores](int pos1, int pos2) { return (all_scores[pos1] > all_scores[pos2]); });
  std::vector<int> keep;
  while (order.size() > 0) {
    int i = order[0];
    keep.push_back(i);
    if (keep.size() >= max_boxes) {
      break;
    }
    int len = order.size() - 1;
    std::vector<float> ovr(len);
    for (int j = 0; j < len; j++) {
      float xx1 = std::max(all_boxes[i][1], all_boxes[order[j + 1]][1]);
      float yy1 = std::max(all_boxes[i][0], all_boxes[order[j + 1]][0]);
      float xx2 = std::min(all_boxes[i][3], all_boxes[order[j + 1]][3]);
      float yy2 = std::min(all_boxes[i][2], all_boxes[order[j + 1]][2]);

      float w = std::max(0.0f, xx2 - xx1 + 1);
      float h = std::max(0.0f, yy2 - yy1 + 1);
      float inter = w * h;
      ovr[j] = inter / (areas[i] + areas[order[j + 1]] - inter);
    }
    std::vector<int> inds;
    for (int j = 0; j < len; j++) {
      if (ovr[j] <= thres) {
        inds.push_back(j + 1);
      }
    }
    std::vector<int> new_order;
    for (int k = 0; k < inds.size(); k++) {
      new_order.push_back(order[inds[k]]);
    }
    order = new_order;
  }
  return keep;
}

template <typename Pixel_Type>
bool ImplementAffine(LiteMat &src, LiteMat &out_img, const double M[6], std::vector<size_t> &dsize,
                     Pixel_Type borderValue) {
  if (dsize.size() != 2 || CheckZero(dsize)) {
    return false;
  }

  double IM[6];
  for (int i = 0; i < 6; i++) {
    IM[i] = M[i];
  }

  double D = IM[0] * IM[4] - IM[1] * IM[3];
  D = D != 0 ? 1.0f / D : 0;
  double A11 = IM[4] * D, A22 = IM[0] * D;
  IM[0] = A11;
  IM[1] *= -D;
  IM[3] *= -D;
  IM[4] = A22;
  double b1 = -IM[0] * IM[2] - IM[1] * IM[5];
  double b2 = -IM[3] * IM[2] - IM[4] * IM[5];
  IM[2] = b1;
  IM[5] = b2;

  out_img.Init(dsize[0], dsize[1], sizeof(Pixel_Type));
  for (int y = 0; y < out_img.height_; y++) {
    for (int x = 0; x < out_img.width_; x++) {
      int src_x = IM[0] * x + IM[1] * y + IM[2];
      int src_y = IM[3] * x + IM[4] * y + IM[5];
      if (src_x >= 0 && src_y >= 0 && src_x < src.width_ && src_y < src.height_) {
        Pixel_Type src_pixel = static_cast<Pixel_Type *>(src.data_ptr_)[src_y * src.width_ + src_x];
        static_cast<Pixel_Type *>(out_img.data_ptr_)[y * out_img.width_ + x] = src_pixel;
      } else {
        static_cast<Pixel_Type *>(out_img.data_ptr_)[y * out_img.width_ + x] = borderValue;
      }
    }
  }

  return true;
}

bool Affine(LiteMat &src, LiteMat &out_img, const double M[6], std::vector<size_t> dsize, UINT8_C1 borderValue) {
  return ImplementAffine(src, out_img, M, dsize, borderValue);
}

bool Affine(LiteMat &src, LiteMat &out_img, const double M[6], std::vector<size_t> dsize, UINT8_C3 borderValue) {
  return ImplementAffine(src, out_img, M, dsize, borderValue);
}

}  // namespace dataset
}  // namespace mindspore
