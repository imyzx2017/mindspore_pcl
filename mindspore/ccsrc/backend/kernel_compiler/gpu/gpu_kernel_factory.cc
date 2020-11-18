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

#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

#include <iostream>
#include <string>

#include "utils/ms_utils.h"
#include "runtime/device/kernel_info.h"
#include "runtime/device/gpu/cuda_common.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace kernel {
GpuKernelFactory &GpuKernelFactory::GetInstance() {
  static GpuKernelFactory instance;
  return instance;
}

void GpuKernelFactory::Register(const std::string &kernel_name, const KernelAttr &kernel_attr,
                                GpuKernelCreater &&creater) {
  map_kernel_name_to_creater_[kernel_name].emplace_back(kernel_attr, creater);
}

bool GpuKernelFactory::CheckIOParam(const std::string &kernel_name, const KernelBuildInfo *kernel_info,
                                    std::vector<std::pair<KernelAttr, GpuKernelCreater>> *iter_second,
                                    size_t attr_index) {
  if (kernel_info->GetInputNum() != iter_second->at(attr_index).first.GetInputSize()) {
    if (!iter_second->at(attr_index).first.GetAllSame()) {
      return false;
    }
  }
  if (kernel_info->GetOutputNum() != iter_second->at(attr_index).first.GetOutputSize()) {
    if (!iter_second->at(attr_index).first.GetAllSame()) {
      return false;
    }
  }
  return true;
}

std::string GpuKernelFactory::SupportedTypeList(const std::string &kernel_name) {
  std::string type_lists = "";
  auto iter = map_kernel_name_to_creater_.find(kernel_name);
  if (map_kernel_name_to_creater_.end() == iter) {
    return type_lists;
  }
  for (size_t attr_index = 0; attr_index < (iter->second).size(); ++attr_index) {
    std::string type_list = "in[";
    auto attr = (iter->second)[attr_index].first;
    for (size_t input_index = 0; input_index < attr.GetInputSize(); ++input_index) {
      type_list = type_list + TypeId2String(attr.GetInputAttr(input_index).first) +
                  ((input_index == (attr.GetInputSize() - 1)) ? "" : " ");
    }
    type_list = type_list + "], out[";
    for (size_t input_index = 0; input_index < attr.GetOutputSize(); ++input_index) {
      type_list = type_list + TypeId2String(attr.GetOutputAttr(input_index).first) +
                  ((input_index == (attr.GetOutputSize() - 1)) ? "" : " ");
    }
    type_lists = type_lists + type_list + "]; ";
  }
  return type_lists;
}

bool GpuKernelFactory::ReducePrecision(
  const std::string &kernel_name, std::shared_ptr<mindspore::kernel::KernelBuildInfo::KernelBuildInfoBuilder> builder) {
  auto kernel_info = builder->Build();
  auto iter = map_kernel_name_to_creater_.find(kernel_name);
  if (map_kernel_name_to_creater_.end() == iter) {
    MS_LOG(INFO) << "Not registered GPU kernel: op[" << kernel_name << "]!";
    return false;
  }
  reduce_flag_.first.clear();
  for (size_t attr_index = 0; attr_index < (iter->second).size(); ++attr_index) {
    auto attr_size = (&(iter->second))->at(attr_index).first.GetInputSize();
    for (size_t input_index = 0; input_index < kernel_info->GetInputNum(); input_index++) {
      if (kernel_info->GetInputDeviceType(input_index) == kNumberTypeInt64 &&
          (iter->second)[attr_index].first.GetInputAttr(input_index % attr_size).first == kNumberTypeInt32) {
        builder->SetInputDeviceType(kNumberTypeInt32, input_index);
        reduce_flag_.first.push_back(input_index);
        MS_LOG(WARNING) << "Kernel [" << kernel_name << "] does not support int64, cast input " << input_index
                        << " to int32.";
      }
    }
    for (size_t output_index = 0; output_index < kernel_info->GetOutputNum(); output_index++) {
      if (kernel_info->GetOutputDeviceType(output_index) == kNumberTypeInt64 &&
          (iter->second)[attr_index].first.GetOutputAttr(output_index % attr_size).first == kNumberTypeInt32) {
        builder->SetOutputDeviceType(kNumberTypeInt32, output_index);
        MS_LOG(WARNING) << "Kernel [" << kernel_name << "] does not support int64, cast output " << output_index
                        << " to int32.";
      }
    }
  }
  return GpuKernelFactory::SearchRegistered(kernel_name, builder->Build());
}

std::pair<bool, size_t> GpuKernelFactory::GpuKernelAttrCheck(const std::string &kernel_name,
                                                             const KernelBuildInfo *kernel_info) {
  auto iter = map_kernel_name_to_creater_.find(kernel_name);
  const int marjor_sm = GET_MAJOR_SM;
  if (map_kernel_name_to_creater_.end() == iter) {
    MS_LOG(INFO) << "Not registered GPU kernel: op[" << kernel_name << "]!";
    return std::make_pair(false, 0);
  }
  if ((iter->second).size() == 1 && (iter->second)[0].first.GetInputSize() == 0) {
    return std::make_pair(true, 0);
  }

  for (size_t attr_index = 0; attr_index < (iter->second).size(); ++attr_index) {
    if (!CheckIOParam(kernel_name, kernel_info, &(iter->second), attr_index)) {
      continue;
    }
    bool flag = true;
    auto attr_size = (&(iter->second))->at(attr_index).first.GetInputSize();
    // data type matching check of all input parameters of kernel
    for (size_t input_index = 0; input_index < kernel_info->GetInputNum(); input_index++) {
      const bool check_sm = mindspore::device::gpu::CudaCommon::GetInstance().check_sm();
      if (check_sm && marjor_sm < RECOMMEND_SM && kernel_info->GetInputDeviceType(input_index) == kNumberTypeFloat16) {
        if (marjor_sm < MINIUM_SM) {
          MS_LOG(EXCEPTION) << "Half precision ops can be used on Devices which computing capacity is >= " << MINIUM_SM
                            << ", but the current device's computing capacity is " << marjor_sm;
        }
        MS_LOG(WARNING) << "It is recommended to use devices with a computing capacity >= " << RECOMMEND_SM
                        << ", but the current device's computing capacity is " << marjor_sm;
        mindspore::device::gpu::CudaCommon::GetInstance().set_check_sm(false);
      }
      if (kernel_info->GetInputDeviceType(input_index) !=
          (iter->second)[attr_index].first.GetInputAttr(input_index % attr_size).first) {
        flag = false;
        break;
      }
    }
    if (!flag) {
      continue;
    }
    attr_size = (&(iter->second))->at(attr_index).first.GetOutputSize();
    // data type matching check of all output parameters of kernel
    for (size_t output_index = 0; output_index < kernel_info->GetOutputNum(); output_index++) {
      if (kernel_info->GetOutputDeviceType(output_index) !=
          (iter->second)[attr_index].first.GetOutputAttr(output_index % attr_size).first) {
        flag = false;
        break;
      }
    }
    // finish data type matching check and return a pair maintain the whether matching is success,
    // if first is true, second is index of matching KernelAttr and creater pair in vector;
    if (flag) {
      size_t match_index = attr_index;
      return std::make_pair(true, match_index);
    }
  }
  return std::make_pair(false, 0);
}

GpuKernel *GpuKernelFactory::Create(const std::string &kernel_name, const CNodePtr &apply_kernel) {
  auto kernel_info = dynamic_cast<device::KernelInfo *>(apply_kernel->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const KernelBuildInfo *kernel_build_Info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(kernel_build_Info);
  std::pair<bool, size_t> ret_pair = GpuKernelAttrCheck(kernel_name, kernel_build_Info);
  if (ret_pair.first) {
    return (map_kernel_name_to_creater_.find(kernel_name)->second)[ret_pair.second].second();
  }
  return nullptr;
}

bool GpuKernelFactory::SearchRegistered(const std::string &kernel_name, const KernelBuildInfoPtr &kernel_build_info) {
  std::pair<bool, size_t> ret_pair = GpuKernelAttrCheck(kernel_name, kernel_build_info.get());
  return ret_pair.first;
}
}  // namespace kernel
}  // namespace mindspore
