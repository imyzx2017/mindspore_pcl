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

#include "runtime/device/ascend/executor/ai_core_dynamic_kernel.h"

#include <regex>
#include <algorithm>
#include <memory>
#include "framework/common/debug/log.h"
#include "utils/log_adapter.h"
#include "runtime/device/ascend/executor/tiling/op_tiling_calculater.h"
#include "register/op_tiling.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_context.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "common/trans.h"

namespace mindspore {
namespace device {
namespace ascend {
AiCoreDynamicKernel::~AiCoreDynamicKernel() {
  if (tiling_data_ptr_ != nullptr) {
    auto ret = rtFree(tiling_data_ptr_);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "rtFree tiling_data_ptr_ failed";
    }
  }
}

void AiCoreDynamicKernel::Execute() {
  if (stream_ == nullptr) {
    MS_LOG(EXCEPTION) << "stream_ptr should not be nullptr.";
  }
  MS_LOG(INFO) << "Start Execute node:" << cnode_ptr_->fullname_with_scope();
  rtL2Ctrl_t *l2ctrl = nullptr;
  auto args_size = static_cast<uint32_t>(UlongToUint(sizeof(void *)) * runtime_args_.size());
  if (RT_ERROR_NONE != rtKernelLaunch(stub_func_, block_dim_, runtime_args_.data(), args_size, l2ctrl, stream_)) {
    MS_LOG(EXCEPTION) << "Call runtime rtKernelLaunch error.";
  }
  MS_LOG(INFO) << "End Execute node:" << cnode_ptr_->fullname_with_scope();
}

std::string ReplaceInvalidJsonStr(const std::string &str) {
  auto ret = std::regex_replace(str, std::regex("100000000"), R"("100000000")");
  ret = std::regex_replace(ret, std::regex("100000001"), R"("100000001")");
  ret = std::regex_replace(ret, std::regex("100000002"), R"("100000002")");
  ret = std::regex_replace(ret, std::regex("True"), R"(true)");
  ret = std::regex_replace(ret, std::regex("False"), R"(false)");
  return ret;
}

void AiCoreDynamicKernel::ParseCompileJson() {
  if (!AnfAlgo::IsDynamicShape(cnode_ptr_)) {
    return;
  }
  if (!AnfAlgo::HasNodeAttr(kAttrCompileInfo, cnode_ptr_)) {
    MS_LOG(EXCEPTION) << "Get compile_info failed";
  }
  auto compile_info_attr = AnfAlgo::GetNodeAttr<std::string>(cnode_ptr_, kAttrCompileInfo);
  std::replace(compile_info_attr.begin(), compile_info_attr.end(), '\'', '\"');
  compile_info_attr = ReplaceInvalidJsonStr(compile_info_attr);
  MS_LOG(INFO) << "Get compile_info:" << compile_info_attr;

  try {
    compile_info_json_ = std::make_shared<nlohmann::json>(nlohmann::json::parse(compile_info_attr));
  } catch (nlohmann::json::parse_error &e) {
    MS_LOG(EXCEPTION) << "parse json failed, error:" << e.what();
  }

  if (AnfAlgo::HasNodeAttr(kAttrFusionType, cnode_ptr_)) {
    auto fusion_type = AnfAlgo::GetNodeAttr<std::string>(cnode_ptr_, kAttrFusionType);
    MS_LOG(INFO) << "Get fusion_type:" << fusion_type;
    (*compile_info_json_)["_pattern"] = fusion_type;
  }
}

void AiCoreDynamicKernel::Initialize() {
  DynamicKernel::Initialize();
  ParseCompileJson();
}

void AiCoreDynamicKernel::UpdateArgs() {
  ComputeTiling();

  if (!CopyTilingToDevice()) {
    MS_LOG(EXCEPTION) << "Copy tiling to device failed";
  }

  AllocateWorkspace();

  auto kernel_mod = AnfAlgo::GetKernelMod(cnode_ptr_);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  AddressPtrList kernel_inputs;
  AddressPtrList kernel_workspaces;
  AddressPtrList kernel_outputs;
  KernelRuntime::GenLaunchArgs(*kernel_mod, cnode_ptr_, &kernel_inputs, &kernel_workspaces, &kernel_outputs);

  runtime_args_.clear();

  (void)std::transform(std::begin(kernel_inputs), std::end(kernel_inputs), std::back_inserter(runtime_args_),
                       [](const AddressPtr &input) { return input->addr; });
  (void)std::transform(std::begin(kernel_outputs), std::end(kernel_outputs), std::back_inserter(runtime_args_),
                       [](const AddressPtr &output) { return output->addr; });
  // Update workspace
  if (!workspace_addr_.empty()) {
    (void)std::transform(std::begin(workspace_addr_), std::end(workspace_addr_), std::back_inserter(runtime_args_),
                         [](const DeviceAddressPtr &address_ptr) { return address_ptr->GetMutablePtr(); });
  }

  if (is_dynamic_shape_ && !tiling_data_.empty() && tiling_data_ptr_ != nullptr) {
    runtime_args_.push_back(tiling_data_ptr_);
  }
}

void AiCoreDynamicKernel::ComputeTiling() {
  MS_EXCEPTION_IF_NULL(cnode_ptr_);
  MS_LOG(INFO) << "Start compute tiling of:" << cnode_ptr_->fullname_with_scope();
  optiling::OpRunInfo op_run_info;

  OpTilingCalculater::GetInstance().CalculateTiling(NOT_NULL(cnode_ptr_), NOT_NULL(compile_info_json_),
                                                    depend_tensor_map_, NOT_NULL(&op_run_info));
  block_dim_ = op_run_info.block_dim;
  workspaces_size_ = op_run_info.workspaces;
  tiling_data_ = op_run_info.tiling_data.str();
}

void AiCoreDynamicKernel::AllocateWorkspace() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto runtime_instance = KernelRuntimeManager::Instance().GetSingleKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);

  workspace_addr_.clear();
  for (auto size : workspaces_size_) {
    auto device_address_ptr = std::make_shared<AscendDeviceAddress>(nullptr, size);
    auto device_ptr = runtime_instance->MallocMem(MemType::kDynamicMem, size, device_address_ptr);
    if (device_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "MallocMem from memory pool failed";
    }
    workspace_addr_.emplace_back(device_address_ptr);
  }
}

bool AiCoreDynamicKernel::CopyTilingToDevice() {
  if (tiling_data_.size() > op_para_size_) {
    MS_LOG(EXCEPTION) << "compute tiling size:" << tiling_data_.size()
                      << " larger than tbe build op_para_size:" << op_para_size_;
  }

  if (tiling_data_.empty() || tiling_data_ptr_ == nullptr) {
    MS_LOG(INFO) << "tiling size is 0, skip rtMemcpyAsync";
    return true;
  }

  auto ret = rtMemcpyAsync(tiling_data_ptr_, tiling_data_.size(), tiling_data_.c_str(), tiling_data_.size(),
                           RT_MEMCPY_HOST_TO_DEVICE_EX, stream_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "tiling rtMemcpyAsync failed, ret:" << ret;
  }
  return true;
}

void AiCoreDynamicKernel::PostExecute() {}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
