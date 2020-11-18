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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_UTILS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_UTILS_H_

#include <string>
#include <vector>
#include "CL/cl2.hpp"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"
#include "src/lite_kernel.h"
#include "src/common/utils.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/opencl_kernel.h"

namespace mindspore::lite {
kernel::LiteKernel *GetOpenCLKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                                    OpParameter *parameter, const InnerContext *ctx, const kernel::KernelKey &key);
}

namespace mindspore::kernel {

int GetUpPow2(int n);

int GetMaxDivisor(int x, int divisor);

int GetMaxDivisorStrategy0(int x, int divisor);

int GetMaxDivisorStrategy1(int x, int divisor);

std::vector<size_t> GetCommonGlobalSize(const std::vector<size_t> &local, const std::vector<size_t> &global);

std::vector<size_t> GetCommonLocalSize(const std::vector<size_t> &global, int max_size);

std::string CLErrorCode(cl_int error_code);

int WriteToBin(const std::string &file_path, void *data, size_t size);

void PrintTensor(const lite::Tensor *tensor, lite::opencl::MemType mem_type, int n = 10,
                 const std::string &out_file = "");

void PrintKernelOutput(OpenCLKernel *kernel, int n = 10, const std::string &out_file = "");

std::vector<int> GetNHWCShape(const std::vector<int> &tensor_shape);

std::vector<size_t> GetImage2dShapeFromNHWC(const std::vector<int> &tensor_shape, schema::Format format);

template <class T1, class T2>
void PackNCHWToNC4HW4(void *src, void *dst, int batch, int plane, int channel, const std::function<T2(T1)> &to_dtype) {
  int c4 = UP_DIV(channel, C4NUM);
  for (int b = 0; b < batch; b++) {
    int src_offset = b * plane * channel;
    int dst_offset = b * plane * c4 * C4NUM;
    for (int c = 0; c < channel; c++) {
      int c4_block_num = c / C4NUM;
      int c4_block_rem = c % C4NUM;
      int src_c_offset = src_offset + c * plane;
      int dst_c_offset = dst_offset + c4_block_num * plane * C4NUM;
      for (int k = 0; k < plane; k++) {
        int src_kernel_offset = src_c_offset + k;
        int dst_kernel_offset = dst_c_offset + C4NUM * k + c4_block_rem;
        (static_cast<T2 *>(dst) + dst_kernel_offset)[0] = to_dtype((static_cast<T1 *>(src) + src_kernel_offset)[0]);
      }
    }
  }
}

template <class T1, class T2>
void PackNHWCToNHWC4(void *src, void *dst, int batch, int plane, int channel, const std::function<T2(T1)> &to_dtype) {
  int c4 = UP_DIV(channel, C4NUM);
  int nhwc4_batch_unit_offset = c4 * C4NUM * plane;
  int ic_remainder_ = channel % C4NUM;
  if (ic_remainder_ != 0) {
    int nhwc4_batch_offset = 0;
    for (int b = 0; b < batch; b++) {
      int batch_offset = b * channel * plane;
      for (int i = 0; i < plane; ++i) {
        for (int c = 0; c < channel; ++c) {
          (static_cast<T2 *>(dst) + nhwc4_batch_offset + i * c4 * C4NUM + c)[0] =
            to_dtype((static_cast<T1 *>(src) + batch_offset + i * channel + c)[0]);
        }
      }
      nhwc4_batch_offset += nhwc4_batch_unit_offset;
    }
  } else {
    size_t ori_input_size = batch * plane * channel;
    for (size_t n = 0; n < ori_input_size; ++n) {
      (static_cast<T2 *>(dst) + n)[0] = to_dtype((static_cast<T1 *>(src) + n)[0]);
    }
  }
}

template <class T1, class T2>
void PackNHWCToNC4HW4(void *src, void *dst, int batch, int plane, int channel, const std::function<T2(T1)> &to_dtype) {
  int c4 = UP_DIV(channel, C4NUM);
  for (int b = 0; b < batch; b++) {
    int src_oc_offset = b * plane * channel;
    int dst_oc_offset = b * plane * c4 * C4NUM;
    for (int k = 0; k < plane; k++) {
      int src_kernel_offset = src_oc_offset + k * channel;
      int dst_kernel_offset = dst_oc_offset + k * C4NUM;
      for (int i = 0; i < channel; i++) {
        int c4_block_num = i / C4NUM;
        int c4_block_rem = i % C4NUM;
        int src_ic_offset = src_kernel_offset + i;
        int dst_ic_offset = dst_kernel_offset + c4_block_num * plane * C4NUM + c4_block_rem;
        (static_cast<T2 *>(dst) + dst_ic_offset)[0] = to_dtype((static_cast<T1 *>(src) + src_ic_offset)[0]);
      }
    }
  }
}

template <class T>
std::vector<T> MatrixMultiply(const T A[], const T B[], int M, int N, int K) {
  std::vector<T> C(M * K);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      float s = 0.0f;
      for (int k = 0; k < N; ++k) {
        s += A[i * N + k] * B[k * K + j];
      }
      C[i * K + j] = s;
    }
  }
  return C;
}

template <typename SRC_T, typename DST_T>
void ConvertConvWeight4DTo7D(void *src, void *dst, size_t CO, size_t KH, size_t KW, size_t CI, size_t OGroup = 1,
                             const size_t CI_TILE = 4, const size_t CO_TILE = 4) {
  if (CO_TILE == 0 || CI_TILE == 0) return;
  auto origin_weight = reinterpret_cast<SRC_T *>(src);
  auto packed_weight = reinterpret_cast<DST_T *>(dst);
  auto CI_SLICES = UP_DIV(CI, CI_TILE);
  for (size_t co = 0, src_idx = 0; co < CO; ++co) {
    for (size_t kh = 0; kh < KH; ++kh) {
      for (size_t kw = 0; kw < KW; ++kw) {
        for (size_t ci = 0; ci < CI; ++ci) {
          size_t co_outer = co / (CO_TILE * OGroup);
          size_t group_idx = co % (CO_TILE * OGroup) / CO_TILE;
          size_t co_inner = co % CO_TILE;
          size_t ci_outer = ci / CI_TILE;
          size_t ci_inner = ci % CI_TILE;
          size_t dst_idx =
            (((((co_outer * KH + kh) * KW + kw) * CI_SLICES + ci_outer) * OGroup + group_idx) * CI_TILE + ci_inner) *
              CO_TILE +
            co_inner;
          packed_weight[dst_idx] = static_cast<DST_T>(origin_weight[src_idx++]);
        }
      }
    }
  }
}

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_UTILS_H_
