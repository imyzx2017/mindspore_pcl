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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_UTIL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_UTIL_H_

#include <cuda_fp16.h>

__device__ static inline float MsAtomicAdd(float *address, const float val) { return atomicAdd(address, val); }

__device__ static inline int MsAtomicAdd(int *address, int val) { return atomicAdd(address, val); }

__device__ static inline unsigned int MsAtomicAdd(unsigned int *address, unsigned int val) {
  return atomicAdd(address, val);
}

__device__ static inline unsigned char MsAtomicAdd(short *address, short val) {  // NOLINT
  bool is_4_byte_aligned = ((size_t) address & 2) == 0;
  unsigned int *aligned = (unsigned int *) ((size_t) address & ~2);
  unsigned int old = *aligned;
  unsigned int assumed;

  do {
    assumed = old;
    unsigned int replacement;

    if (is_4_byte_aligned) {
      replacement = (old & 0xffff0000) | (((old & 0xffff) + val) & 0xffff);
    } else {
      replacement = old + ((unsigned int) val << 16);
    }

    old = atomicCAS(aligned, assumed, replacement);
  } while (assumed != old);

  if (is_4_byte_aligned) {
    return (short) (old & 0xffff);  // NOLINT
  } else {
    return (short) (old >> 16);  // NOLINT
  }
}

__device__ static inline half MsAtomicAdd(half *address, half val) {
  unsigned int *aligned =
    reinterpret_cast<unsigned int *>(reinterpret_cast<size_t>(address) - (reinterpret_cast<size_t>(address) & 2));
  unsigned int old = *aligned;
  unsigned int assumed;
  unsigned short old_as_us;  // NOLINT
  do {
    assumed = old;
    old_as_us = static_cast<unsigned short>(reinterpret_cast<size_t>(address) & 2 ? old >> 16 : old & 0xffff);  // NOLINT
    half sum = __float2half_rn(__half2float(__ushort_as_half(old_as_us)) + static_cast<float>(val));
    unsigned short sum_as_us = __half_as_ushort(sum);  // NOLINT
    unsigned int sum_as_ui =
      reinterpret_cast<size_t>(address) & 2 ? (sum_as_us << 16) | (old & 0xffff) : (old & 0xffff0000) | sum_as_us;
    old = atomicCAS(aligned, assumed, sum_as_ui);
  } while (assumed != old);
  __half_raw raw = {old_as_us};
  return half(raw);
}

__device__ static inline unsigned char MsAtomicAdd(unsigned char* address, unsigned char val) {
  // We use cuda's atomicCAS(unsigned int*, unsigned int, unsigned int) to
  // implement MsAtomicAdd. An unsigned char may not be 4 byte aligned, but
  // unsigned int* must be 4 byte aligned. This variable contains the offset,
  // in bytes, of the beginning of address, within the 4 byte aligned space that
  // contains it.
  size_t address_offset = (size_t) address & 3;

  // Address of the 4 byte aligned space that contains address.
  unsigned int* aligned = (unsigned int*) ((unsigned char*) address - address_offset);

  // Constants which will be used later with __byte_perm. __byte_perm is a cuda
  // function which takes 3 unsigned int's (x, y, selector) as parameters and
  // returns an int. __byte_perm returns an integer by selecting bytes from x
  // and y based on the given selector. The selector 0x3210 in will select all
  // four bytes from x, preserving their original order. The position of the
  // "4" in the selector indicates the position in the output where the first
  // byte of y will end up.
  unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};

  // Gets the selector that will select the bytes at address from aligned
  unsigned int selector = selectors[address_offset];

  unsigned int old = *aligned;
  unsigned int assumed = 0;

  do {
    assumed = old;

    // Selects the byte associated with address and put it as the first byte of
    // this variable, so that we can add val to the value at address.
    unsigned int sum = val + __byte_perm(old, 0, address_offset);

    // Takes old and replaces the byte corresponding to address with the sum.
    unsigned int replacement = __byte_perm(old, sum, selector);

    // Try to replace the old value with the new value
    old = atomicCAS(aligned, assumed, replacement);
  } while (old != assumed);
  // Select the single byte corredsponding to address and return it.
  return __byte_perm(old, 0, address_offset);
}

__device__ static inline char MsAtomicAdd(char* address, char val) {
  size_t address_offset = (size_t) address & 3;
  unsigned int* aligned = reinterpret_cast<unsigned int *>(reinterpret_cast<char *>(address) - address_offset);
  unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
  unsigned int selector = selectors[address_offset];
  unsigned int old = *aligned;
  unsigned int assumed = 0;

  do {
    assumed = old;

    unsigned int sum = val + __byte_perm(old, 0, address_offset);
    unsigned int replacement = __byte_perm(old, sum, selector);

    old = atomicCAS(aligned, assumed, replacement);
  } while (old != assumed);
  return __byte_perm(old, 0, address_offset);
}

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_UTIL_H_
