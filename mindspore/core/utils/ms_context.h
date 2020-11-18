/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_MS_CONTEXT_H_
#define MINDSPORE_CORE_UTILS_MS_CONTEXT_H_
#include <thread>
#include <memory>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <utility>
#include "utils/log_adapter.h"

namespace mindspore {
enum MsBackendPolicy {
  kMsBackendGeOnly = 0,
  kMsBackendVmOnly = 1,
  kMsBackendGePrior = 2,
  kMsBackendVmPrior = 3,
  kMsBackendMsPrior = 4,
  kMsBackendUnknown = 5,
};

const int kGraphMode = 0;
const int kPynativeMode = 1;
const char kCPUDevice[] = "CPU";
const char kGPUDevice[] = "GPU";
const char kAscendDevice[] = "Ascend";
const char kDavinciInferenceDevice[] = "AscendInference";
const char kDavinciDevice[] = "Davinci";
const char KNpuLog[] = "_npu_log";
const unsigned int MAX_CALL_DEPTH_DEFAULT = 1000;

const std::set<std::string> kTargetSet = {kCPUDevice, kGPUDevice, kAscendDevice, kDavinciDevice};
// The default max available device memory is 1024GB.
const float kDefaultMaxDeviceMemory = 1024;

// enum definition for MindSpore Context Parameter
enum MsCtxParam : unsigned {
  // paramater of type bool
  MS_CTX_TYPE_BOOL_BEGIN,
  MS_CTX_ENABLE_AUTO_MIXED_PRECISION = MS_CTX_TYPE_BOOL_BEGIN,
  MS_CTX_CHECK_BPROP_FLAG,
  MS_CTX_ENABLE_DUMP,
  MS_CTX_ENABLE_DYNAMIC_MEM_POOL,
  MS_CTX_ENABLE_GPU_SUMMARY,
  MS_CTX_ENABLE_GRAPH_KERNEL,
  MS_CTX_ENABLE_HCCL,
  MS_CTX_ENABLE_LOOP_SINK,
  MS_CTX_ENABLE_MEM_REUSE,
  MS_CTX_ENABLE_PYNATIVE_HOOK,
  MS_CTX_ENABLE_PYNATIVE_INFER,
  MS_CTX_ENABLE_REDUCE_PRECISION,
  MS_CTX_ENABLE_SPARSE,
  MS_CTX_ENABLE_TASK_SINK,
  MS_CTX_IR_FUSION_FLAG,
  MS_CTX_IS_MULTI_GRAPH_SINK,
  MS_CTX_IS_PYNATIVE_GE_INIT,
  MS_CTX_PRECOMPILE_ONLY,
  MS_CTX_ENABLE_PROFILING,
  MS_CTX_SAVE_GRAPHS_FLAG,
  MS_CTX_TYPE_BOOL_END,

  // paramater of type int
  MS_CTX_TYPE_INT_BEGIN = MS_CTX_TYPE_BOOL_END,
  MS_CTX_EXECUTION_MODE = MS_CTX_TYPE_INT_BEGIN,
  MS_CTX_TYPE_INT_END,

  // paramater of type uint32
  MS_CTX_TYPE_UINT32_BEGIN = MS_CTX_TYPE_INT_END,
  MS_CTX_DEVICE_ID = MS_CTX_TYPE_UINT32_BEGIN,
  MS_CTX_GE_REF,
  MS_CTX_MAX_CALL_DEPTH,
  MS_CTX_TSD_REF,
  MS_CTX_TYPE_UINT32_END,

  // paramater of type float
  MS_CTX_TYPE_FLOAT_BEGIN = MS_CTX_TYPE_UINT32_END,
  MS_CTX_MAX_DEVICE_MEMORY = MS_CTX_TYPE_FLOAT_BEGIN,
  MS_CTX_TYPE_FLOAT_END,

  // paramater of type string
  MS_CTX_TYPE_STRING_BEGIN = MS_CTX_TYPE_FLOAT_END,
  MS_CTX_DEVICE_TARGET = MS_CTX_TYPE_STRING_BEGIN,
  MS_CTX_GRAPH_MEMORY_MAX_SIZE,
  MS_CTX_PRINT_FILE_PATH,
  MS_CTX_PROFILING_OPTIONS,
  MS_CTX_SAVE_DUMP_PATH,
  MS_CTX_SAVE_GRAPHS_PATH,
  MS_CTX_VARIABLE_MEMORY_MAX_SIZE,
  MS_CTX_TYPE_STRING_END,

  // parameter numbers of each type
  NUM_BOOL_PARAMS = MS_CTX_TYPE_BOOL_END - MS_CTX_TYPE_BOOL_BEGIN,
  NUM_INT_PARAMS = MS_CTX_TYPE_INT_END - MS_CTX_TYPE_INT_BEGIN,
  NUM_UINT32_PARAMS = MS_CTX_TYPE_UINT32_END - MS_CTX_TYPE_UINT32_BEGIN,
  NUM_FLOAT_PARAMS = MS_CTX_TYPE_FLOAT_END - MS_CTX_TYPE_FLOAT_BEGIN,
  NUM_STRING_PARAMS = MS_CTX_TYPE_STRING_END - MS_CTX_TYPE_STRING_BEGIN
};

class MsContext {
 public:
  MsContext(const std::string &backend_policy, const std::string &target);
  ~MsContext() = default;
  MsContext(const MsContext &) = delete;
  MsContext &operator=(const MsContext &) = delete;
  using DeviceSeter = std::function<void(const std::string &device_target)>;
  using DeviceTypeSeter = std::function<void(std::shared_ptr<MsContext> &)>;
  static std::shared_ptr<MsContext> GetInstance();

  std::string backend_policy() const;
  bool set_backend_policy(const std::string &policy);

  static void device_seter(DeviceSeter device) { seter_ = device; }
  static void device_type_seter(DeviceTypeSeter device_type) { device_type_seter_ = device_type; }

  std::thread tdt_print_;

  template <typename T>
  void set_param(MsCtxParam param, const T &value) {
    MS_LOG(EXCEPTION) << "Need to implement " << __FUNCTION__ << " for type " << typeid(T).name() << ".";
  }

  template <typename T>
  const T &get_param(MsCtxParam param) const {
    MS_LOG(EXCEPTION) << "Need to implement " << __FUNCTION__ << " for type " << typeid(T).name() << ".";
  }

  template <typename T>
  void increase_param(MsCtxParam param) {
    MS_LOG(EXCEPTION) << "Need to implement " << __FUNCTION__ << " for type " << typeid(T).name() << ".";
  }

  template <typename T>
  void decrease_param(MsCtxParam param) {
    MS_LOG(EXCEPTION) << "Need to implement " << __FUNCTION__ << " for type " << typeid(T).name() << ".";
  }

 private:
  inline static DeviceSeter seter_ = nullptr;
  inline static DeviceTypeSeter device_type_seter_ = nullptr;
  static std::shared_ptr<MsContext> inst_context_;
  static std::map<std::string, MsBackendPolicy> policy_map_;

  bool bool_params_[MsCtxParam::NUM_BOOL_PARAMS];
  int int_params_[MsCtxParam::NUM_INT_PARAMS];
  uint32_t uint32_params_[MsCtxParam::NUM_UINT32_PARAMS];
  float float_params_[MsCtxParam::NUM_FLOAT_PARAMS];
  std::string string_params_[MsCtxParam::NUM_STRING_PARAMS];

  MsBackendPolicy backend_policy_;
};

// set method implementation for type bool/int/uint32_t/float/std::string
template <>
inline void MsContext::set_param<bool>(MsCtxParam param, const bool &value) {
  bool_params_[param - MS_CTX_TYPE_BOOL_BEGIN] = value;
}

template <>
inline void MsContext::set_param<int>(MsCtxParam param, const int &value) {
  int_params_[param - MS_CTX_TYPE_INT_BEGIN] = value;
}

template <>
inline void MsContext::set_param<uint32_t>(MsCtxParam param, const uint32_t &value) {
  uint32_params_[param - MS_CTX_TYPE_UINT32_BEGIN] = value;
}

template <>
inline void MsContext::set_param<float>(MsCtxParam param, const float &value) {
  float_params_[param - MS_CTX_TYPE_FLOAT_BEGIN] = value;
}

template <>
inline void MsContext::set_param<std::string>(MsCtxParam param, const std::string &value) {
  if (seter_ != nullptr && param == MS_CTX_DEVICE_TARGET) {
    MS_LOG(INFO) << "ms set context device target:" << value;
    seter_(value);
  }
  string_params_[param - MS_CTX_TYPE_STRING_BEGIN] = value;
}

// get method implementation for type bool/int/uint32_t/float/std::string
template <>
inline const bool &MsContext::get_param<bool>(MsCtxParam param) const {
  return bool_params_[param - MS_CTX_TYPE_BOOL_BEGIN];
}

template <>
inline const int &MsContext::get_param<int>(MsCtxParam param) const {
  return int_params_[param - MS_CTX_TYPE_INT_BEGIN];
}

template <>
inline const uint32_t &MsContext::get_param<uint32_t>(MsCtxParam param) const {
  return uint32_params_[param - MS_CTX_TYPE_UINT32_BEGIN];
}

template <>
inline const float &MsContext::get_param<float>(MsCtxParam param) const {
  return float_params_[param - MS_CTX_TYPE_FLOAT_BEGIN];
}

template <>
inline const std::string &MsContext::get_param<std::string>(MsCtxParam param) const {
  return string_params_[param - MS_CTX_TYPE_STRING_BEGIN];
}

// increate method implementation for type uint32_t
template <>
inline void MsContext::increase_param<uint32_t>(MsCtxParam param) {
  uint32_params_[param - MS_CTX_TYPE_UINT32_BEGIN]++;
}

// decreate method implementation for type uint32_t
template <>
inline void MsContext::decrease_param<uint32_t>(MsCtxParam param) {
  uint32_params_[param - MS_CTX_TYPE_UINT32_BEGIN]--;
}
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_MS_CONTEXT_H_
