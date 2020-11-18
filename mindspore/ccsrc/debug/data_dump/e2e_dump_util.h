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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_E_2_E_DUMP_UTIL_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_E_2_E_DUMP_UTIL_H_

#include <map>
#include <string>

#include "backend/session/kernel_graph.h"
#include "runtime/device/device_address.h"
#include "debug/data_dump/dump_json_parser.h"

#ifndef ENABLE_DEBUGGER
class Debugger;
#endif
namespace mindspore {
class E2eDumpUtil {
 public:
  E2eDumpUtil() = default;
  ~E2eDumpUtil() = default;
  static bool DumpData(const session::KernelGraph *graph, uint32_t device_id, Debugger *debugger = nullptr);
  static void GetFileKernelName(NotNull<std::string *> kernel_name);
  // Dump data when task error.
  static void DumpInputImpl(const CNodePtr &node, bool trans_flag, const std::string &dump_path,
                            std::string *kernel_name, Debugger *debugger);
  static void DumpOutputImpl(const CNodePtr &node, bool trans_flag, const std::string &dump_path,
                             std::string *kernel_name, Debugger *debugger);

 private:
  static void DumpOutput(const session::KernelGraph *graph, const std::string &dump_path, Debugger *debugger);
  static void DumpInput(const session::KernelGraph *graph, const std::string &dump_path, Debugger *debugger);
  static void DumpParametersAndConst(const session::KernelGraph *graph, const std::string &dump_path,
                                     Debugger *debugger);

  static void DumpMemToFile(const std::string &file_path, NotNull<const device::DeviceAddress *> addr, bool trans_flag,
                            const ShapeVector &int_shapes, const TypeId &type);
  static void DumpGPUMemToFile(const std::string &file_path, const std::string &original_kernel_name,
                               NotNull<const device::DeviceAddress *> addr, bool trans_flag,
                               const ShapeVector &int_shapes, const TypeId &type, size_t slot, Debugger *debugger);
  static void GetDumpIntShape(const AnfNodePtr &node, size_t index, bool trans_flag, NotNull<ShapeVector *> int_shapes);
  static bool IsDeviceTargetGPU();
  static void DumpSingleAnfnode(const AnfNodePtr &anf_node, const size_t output_index, const std::string &dump_path,
                                bool trans_flag, std::map<std::string, size_t> *const_map, Debugger *debugger);
};
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_E_2_E_DUMP_UTIL_H_
