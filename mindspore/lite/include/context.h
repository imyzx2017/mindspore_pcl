/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_LITE_INCLUDE_CONTEXT_H_
#define MINDSPORE_LITE_INCLUDE_CONTEXT_H_

#include <string>
#include <memory>
#include "include/ms_tensor.h"
#include "include/lite_utils.h"

namespace mindspore::lite {
/// \brief CpuBindMode defined for holding bind cpu strategy argument.
typedef enum {
  NO_BIND = 0,    /**< no bind */
  HIGHER_CPU = 1, /**< bind higher cpu first */
  MID_CPU = 2     /**< bind middle cpu first */
} CpuBindMode;

/// \brief DeviceType defined for holding user's preferred backend.
typedef enum {
  DT_CPU, /**< CPU device type */
  DT_GPU, /**< GPU device type */
  DT_NPU  /**< NPU device type, not supported yet */
} DeviceType;

/// \brief CpuDeviceInfo defined for CPU's configuration information.
typedef struct {
  bool enable_float16_ = false; /**< prior enable float16 inference */
  CpuBindMode cpu_bind_mode_ = MID_CPU;
} CpuDeviceInfo;

/// \brief GpuDeviceInfo defined for GPU's configuration information.
typedef struct {
  bool enable_float16_ = false; /**< prior enable float16 inference */
} GpuDeviceInfo;

/// \brief DeviceInfo defined for backend's configuration information.
union DeviceInfo {
  CpuDeviceInfo cpu_device_info_;
  GpuDeviceInfo gpu_device_info_;
};

/// \brief DeviceContext defined for holding backend's configuration information.
struct DeviceContext {
  DeviceType device_type_ = DT_CPU;
  DeviceInfo device_info_;
};

/// \brief Context defined for holding environment variables during runtime.
struct Context {
  std::string vendor_name_;
  int thread_num_ = 2; /**< thread number config for thread pool */
  AllocatorPtr allocator = nullptr;
  DeviceContextVector device_list_ = {{DT_CPU, {false, MID_CPU}}};
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_INCLUDE_CONTEXT_H_
