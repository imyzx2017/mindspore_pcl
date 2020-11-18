
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

#include "runtime/device/kernel_runtime.h"
#include <functional>
#include <numeric>
#include <utility>
#include <vector>
#include "backend/optimizer/common/helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/kernel_graph.h"
#include "common/trans.h"
#include "debug/data_dump/dump_json_parser.h"
#include "frontend/operator/ops.h"
#include "ir/value.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "utils/shape_utils.h"
#include "utils/utils.h"
using mindspore::kernel::Address;
using mindspore::kernel::AddressPtr;

namespace mindspore {
namespace device {
KernelRuntime::~KernelRuntime() {}

bool KernelRuntime::Load(session::KernelGraph *graph, bool is_task_sink) { return true; }

bool KernelRuntime::LoadData(session::KernelGraph *graph) { return false; }

bool KernelRuntime::NodeOutputDeviceAddressExist(const AnfNodePtr &kernel, size_t index) {
  MS_EXCEPTION_IF_NULL(kernel);
  if (AnfAlgo::OutputAddrExist(kernel, index)) {
    return true;
  }
  return false;
}

size_t KernelRuntime::CountNodeDeviceMemorySize(const mindspore::AnfNodePtr &node, size_t output_index) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_index >= AnfAlgo::GetOutputTensorNum(node)) {
    MS_EXCEPTION(ArgumentError) << "output index [" << output_index << "] large than the output size ["
                                << AnfAlgo::GetOutputTensorNum(node) << "] of node!";
  }
  TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(node, output_index);
  if (output_type_id == kTypeUnknown) {
    output_type_id = AnfAlgo::GetOutputInferDataType(node, output_index);
  }
  size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
  std::vector<size_t> shape = AnfAlgo::GetOutputDeviceShape(node, output_index);
  auto format = AnfAlgo::GetOutputFormat(node, output_index);
  if (shape.empty() && format != kOpFormat_DEFAULT) {
    shape = trans::PaddingShapeTo4d(shape, AnfAlgo::GetOutputReshapeType(node, output_index));
    shape = trans::TransShapeToDevice(shape, format);
  }
  // scalar's output shape is a empty vector
  size_t tensor_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
  return tensor_size;
}

void KernelRuntime::AssignMemory(session::KernelGraph *graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->ResetDynamicMemory();
  AssignStaticMemory(graph);
  AssignDynamicMemory(graph);
  UpdateRefNodeOutputMem(graph);
}

void KernelRuntime::RunOpAssignMemory(const ValuePtr &pre_output_value,
                                      const std::vector<tensor::TensorPtr> &input_tensors,
                                      session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->ResetDynamicMemory();
  RunOpAssignInputMemory(input_tensors, graph);
  AssignStaticMemoryValueNode(graph);
  RunOpAssignOutputNodeMemory(pre_output_value, graph);
  for (const auto &cnode : graph->execution_order()) {
    RunOpAssignOutputMemory(cnode);
    RunOpAssignWorkSpaceMemory(cnode);
  }
  UpdateRefNodeOutputMem(graph);
}

void KernelRuntime::RunOpClearMemory(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  // clear input parameter memory resource
  for (const auto &input_node : graph->inputs()) {
    MS_EXCEPTION_IF_NULL(input_node);
    AnfAlgo::SetOutputAddr(nullptr, 0, input_node.get());
  }
  // clear input value node memory resource
  for (const auto &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    AnfAlgo::SetOutputAddr(nullptr, 0, value_node.get());
  }
  for (const auto &cnode : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    // clear output memory resource
    for (size_t index = 0; index < AnfAlgo::GetOutputTensorNum(cnode); ++index) {
      AnfAlgo::SetOutputAddr(nullptr, index, cnode.get());
    }
    // clear workspace memory resource
    auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_lists = kernel_mod->GetWorkspaceSizeList();
    for (size_t index = 0; index < workspace_lists.size(); ++index) {
      AnfAlgo::SetWorkspaceAddr(nullptr, index, cnode.get());
    }
  }
}

bool KernelRuntime::DumpDataEnabled() {
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  return dump_json_parser.e2e_dump_enabled();
}

bool KernelRuntime::DumpDataEnabledIteration() {
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!dump_json_parser.e2e_dump_enabled()) {
    return false;
  }

  auto cur_iter = dump_json_parser.cur_dump_iter() + 1;
  if (dump_json_parser.iteration() != 0) {
    return cur_iter == dump_json_parser.iteration();
  }
  return true;
}

void KernelRuntime::AssignStaticMemory(session::KernelGraph *graph) {
  AssignStaticMemoryInput(graph);
  AssignStaticMemoryValueNode(graph);
  AssignStaticMemoryOutput(graph);
}

void KernelRuntime::RunOpAssignInputMemory(const std::vector<tensor::TensorPtr> &input_tensors,
                                           const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (input_tensors.size() != graph->inputs().size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size()
                      << " should be equal to graph input parameter size " << graph->inputs().size();
  }

  for (size_t input_index = 0; input_index < graph->inputs().size(); ++input_index) {
    auto item = graph->inputs()[input_index];
    MS_EXCEPTION_IF_NULL(item);
    if (!item->isa<Parameter>()) {
      continue;
    }
    auto output_size = AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      MS_EXCEPTION_IF_NULL(input_tensors[input_index]);
      auto output_address =
        std::dynamic_pointer_cast<device::DeviceAddress>(input_tensors[input_index]->device_address());
      if (output_address != nullptr) {
        AnfAlgo::SetOutputAddr(output_address, index, item.get());
        continue;
      }
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
      if (output_type_id == kTypeUnknown) {
        output_type_id = AnfAlgo::GetOutputInferDataType(item, index);
      }
      auto tensor_size = CountNodeDeviceMemorySize(item, index);
      auto device_address =
        CreateDeviceAddress(nullptr, tensor_size, AnfAlgo::GetOutputFormat(item, index), output_type_id);
      MS_EXCEPTION_IF_NULL(device_address);
      MS_EXCEPTION_IF_NULL(mem_manager_);
      auto ret = mem_manager_->MallocMemFromMemPool(device_address, tensor_size);
      if (!ret) {
        MS_LOG(EXCEPTION) << "Malloc device memory failed.";
      }
      AnfAlgo::SetOutputAddr(device_address, index, item.get());
    }
  }
}

void KernelRuntime::RunOpAssignOutputMemory(const AnfNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_sizes = kernel_mod->GetOutputSizeList();
  if (output_sizes.empty()) {
    return;
  }

  for (size_t i = 0; i < output_sizes.size(); ++i) {
    if (AnfAlgo::OutputAddrExist(kernel, i)) {
      continue;
    }
    if (AnfAlgo::GetCNodeName(kernel) == kApplyMomentumOpName) {
      auto device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i);
      AnfAlgo::SetOutputAddr(device_address, i, kernel.get());
      continue;
    }
    std::string output_format = AnfAlgo::GetOutputFormat(kernel, i);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
    auto device_address = CreateDeviceAddress(nullptr, output_sizes[i], output_format, output_type);
    device_address->set_host_shape(trans::GetRuntimePaddingShape(kernel, i));
    MS_EXCEPTION_IF_NULL(device_address);
    auto ret = mem_manager_->MallocMemFromMemPool(device_address, output_sizes[i]);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Malloc device memory failed.";
    }
    AnfAlgo::SetOutputAddr(device_address, i, kernel.get());
  }
}

void KernelRuntime::RunOpAssignWorkSpaceMemory(const AnfNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (kernel->isa<CNode>()) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_lists = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_lists.size(); ++i) {
      auto device_address = CreateDeviceAddress(nullptr, workspace_lists[i], "", kTypeUnknown);
      MS_EXCEPTION_IF_NULL(device_address);
      auto ret = mem_manager_->MallocMemFromMemPool(device_address, workspace_lists[i]);
      if (!ret) {
        MS_LOG(EXCEPTION) << "Malloc device memory failed.";
      }
      AnfAlgo::SetWorkspaceAddr(device_address, i, kernel.get());
    }
  }
}

void KernelRuntime::RunOpAssignOutputNodeMemory(const ValuePtr &pre_output_value, session::KernelGraph *graph) {
  if (pre_output_value == nullptr) {
    return;
  }
  std::vector<tensor::TensorPtr> pre_output_tensors;
  TensorValueToTensor(pre_output_value, &pre_output_tensors);
  MS_EXCEPTION_IF_NULL(graph);
  auto output_nodes = graph->outputs();
  if (pre_output_tensors.size() != output_nodes.size()) {
    MS_LOG(EXCEPTION) << "The size of pre output tensors [" << pre_output_tensors.size()
                      << "] is not equal to the size of output nodes of graph [" << output_nodes.size() << "]";
  }
  // share output address with pre output tensors
  for (size_t i = 0; i < output_nodes.size(); ++i) {
    auto output_node_with_index = AnfAlgo::VisitKernel(output_nodes[i], 0);
    if (!output_node_with_index.first->isa<CNode>()) {
      if (output_node_with_index.first->isa<Parameter>()) {
        auto param = output_node_with_index.first->cast<ParameterPtr>();
        if (!param->has_default()) {
          MS_LOG(EXCEPTION) << "The output parameter should be real parameter!";
        }
      }
      continue;
    }
    auto real_output_cnode = output_node_with_index.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(real_output_cnode);
    MS_EXCEPTION_IF_NULL(pre_output_tensors[i]);
    if (pre_output_tensors[i]->device_address() == nullptr) {
      MS_LOG(INFO) << "The address of pre output tensor [" << i << "] is a nullptr!";
      continue;
    }
    if (opt::IsNopNode(real_output_cnode)) {
      if (real_output_cnode->inputs().size() < 2) {
        MS_LOG(EXCEPTION) << "The input size of output node: " << real_output_cnode->DebugString()
                          << " should large than one!";
      }
      AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(pre_output_tensors[i]->device_address()),
                             output_node_with_index.second, real_output_cnode->input(1).get());
    } else {
      AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(pre_output_tensors[i]->device_address()),
                             output_node_with_index.second, output_node_with_index.first.get());
    }
  }
}

void KernelRuntime::AssignStaticMemoryInput(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  MS_LOG(INFO) << "AssignStaticMemoryInput start";
  auto graph_inputs = graph->inputs();
  auto graph_valid_input = graph->valid_inputs();
  graph_inputs.insert(graph_inputs.end(), graph->child_graph_result().begin(), graph->child_graph_result().end());
  std::vector<AnfNodePtr> need_alloc_nodes;
  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    auto item = graph_inputs[i];
    MS_EXCEPTION_IF_NULL(item);
    if (i < graph_valid_input.size() && !graph_valid_input[i]) {
      continue;
    }

    if (AnfAlgo::CheckPrimitiveType(item, prim::kPrimMakeTuple)) {
      auto outs = AnfAlgo::GetAllOutput(item);
      for (auto &out : outs) {
        MS_EXCEPTION_IF_NULL(out);
        if (!out->isa<Parameter>()) {
          continue;
        }
        if (NodeOutputDeviceAddressExist(out, 0)) {
          continue;
        }
        need_alloc_nodes.push_back(out);
      }
    }
    if (!item->isa<Parameter>()) {
      continue;
    }
    if (NodeOutputDeviceAddressExist(item, 0)) {
      continue;
    }
    need_alloc_nodes.push_back(item);
  }

  for (auto &item : need_alloc_nodes) {
    auto output_size = AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
      // if graph output is a weight and doesn't link to any cnode, it's data type will be unknown
      if (output_type_id == kTypeUnknown) {
        MS_LOG(WARNING) << "It is not suggested to use a lonely weight parameter as the output of graph";
        continue;
      }
      auto tensor_size = CountNodeDeviceMemorySize(item, index);
      auto address = CreateDeviceAddress(nullptr, tensor_size, AnfAlgo::GetOutputFormat(item, index), output_type_id);
      MS_LOG(DEBUG) << "Malloc static memory for " << item->fullname_with_scope();
      if (mem_manager_->MallocMem(kStaticMem, tensor_size, address) == nullptr) {
        MS_LOG(EXCEPTION) << "Cannot alloc address when flag is: " << kStaticMem << ", tensor size is: " << tensor_size;
      }
      MS_LOG(INFO) << "Malloc Input for graph " << graph->graph_id() << ", node: " << item->fullname_with_scope()
                   << " index: " << index << " size: " << tensor_size;
      AnfAlgo::SetOutputAddr(address, index, item.get());
    }
  }
  MS_LOG(INFO) << "AssignStaticMemoryInput end";
}

void KernelRuntime::AssignStaticMemoryOutput(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "AssignStaticMemoryOutput start";
  auto nodes = AnfAlgo::GetAllOutput(graph->output(), {prim::kPrimTupleGetItem});
  std::vector<session::KernelWithIndex> non_communication_op;
  // Assign Communicate Op Memory firstly.
  for (const auto &node : nodes) {
    auto item_with_index = AnfAlgo::VisitKernelWithReturnType(node, 0, true);
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    if (!item_with_index.first->isa<CNode>() || !AnfAlgo::IsRealKernel(item_with_index.first)) {
      continue;
    }
    if (AnfAlgo::IsCommunicationOp(item_with_index.first)) {
      AssignCommunicationNodeMem(kStaticMem, item_with_index.first);
    } else {
      non_communication_op.emplace_back(item_with_index);
    }
  }

  for (const auto &item_with_index : non_communication_op) {
    MS_LOG(DEBUG) << "AssignNodeOutputMem for " << item_with_index.first->fullname_with_scope();
    AssignNodeOutputMem(kStaticMem, item_with_index.first, SizeToInt(item_with_index.second));
  }
  MS_LOG(INFO) << "AssignStaticMemoryOutput end";
}

void KernelRuntime::UpdateRefNodeOutputMem(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &kernels = graph->execution_order();
  for (auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);

    auto output_sizes = kernel_mod->GetOutputSizeList();
    if (output_sizes.empty()) {
      MS_LOG(INFO) << "This kernel has no output size.";
      continue;
    }
    for (size_t i = 0; i < output_sizes.size(); ++i) {
      session::AnfWithOutIndex out_pair(kernel, i);
      if (graph->IsInRefOutputMap(out_pair)) {
        auto origin_pair = graph->GetRefCorrespondOutput(out_pair);
        MS_EXCEPTION_IF_NULL(origin_pair.first);
        auto origin_node_output_addr = AnfAlgo::GetMutableOutputAddr(origin_pair.first, origin_pair.second);
        MS_EXCEPTION_IF_NULL(origin_node_output_addr);
        auto cur_node_output_addr = AnfAlgo::GetMutableOutputAddr(kernel, i);
        if (origin_node_output_addr.get() != cur_node_output_addr.get()) {
          MS_LOG(INFO) << "REF address is not same, ref node output need address update";
          MS_LOG(INFO) << "REF origin op is " << origin_pair.first->DebugString() << ", output index is "
                       << origin_pair.second << ", cur op is " << kernel->DebugString() << ", out index is " << i;
          AnfAlgo::SetOutputAddr(origin_node_output_addr, i, kernel.get());
        }
      }
    }
  }
}

void KernelRuntime::AssignCommunicationNodeMem(MemType type, const AnfNodePtr &node) {
  AssignCommunicationNodeInputMem(type, node);
  AssignCommunicationNodeOutputMem(type, node);
}

void KernelRuntime::AssignCommunicationNodeOutputMem(MemType type, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_sizes = kernel_mod->GetOutputSizeList();
  if (output_sizes.empty()) {
    MS_LOG(INFO) << "This kernel[" << node->DebugString() << "] has no output size.";
    return;
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  size_t total_size = 0;
  size_t output_index = 0;
  std::vector<size_t> align_size_list;
  for (uint64_t mem_size : output_sizes) {
    if (AnfAlgo::OutputAddrExist(node, output_index++)) {
      MS_LOG(INFO) << "communication op addr exist";
      continue;
    }
    if (context_ptr->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
      mem_size = mem_manager_->GetCommonAlignSize(mem_size);
    }
    total_size += mem_size;
    align_size_list.emplace_back(mem_size);
  }

  if (type == kReuseDynamicMem) {
    // reuse communication op's all outputs' memory
    type = kReuseDynamicCommMem;
  }

  if (type == kReuseDynamicCommMem || type == kSomasReuseDynamicMem) {
    bool not_reuse = KernelMemNotReuse(node);
    if (not_reuse) {
      type = kDynamicMem;
      MS_LOG(INFO) << "Disable Memory Reuse for " << node->fullname_with_scope() << "'s output.";
    }
  }

  uint8_t *output_ptr = nullptr;
  for (size_t j = 0; j < align_size_list.size(); ++j) {
    std::string output_format = AnfAlgo::GetOutputFormat(node, j);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(node, j);
    auto address = CreateDeviceAddress(nullptr, output_sizes[j], output_format, output_type);
    MS_EXCEPTION_IF_NULL(address);
    if (output_ptr == nullptr) {
      output_ptr = mem_manager_->MallocOutputMem(node, 0, type, total_size, address, true);
      MS_EXCEPTION_IF_NULL(output_ptr);
    } else {
      address->set_ptr(output_ptr);
    }
    AnfAlgo::SetOutputAddr(address, j, node.get());
    output_ptr += align_size_list[j];
  }
}
bool KernelRuntime::KernelMemNotReuse(const AnfNodePtr &node) { return false; }

DeviceAddressPtr KernelRuntime::PreAssignCNodeMemory(const AnfNodePtr &anf_node, size_t index) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto kernel_mod = AnfAlgo::GetKernelMod(anf_node);
  auto output_sizes = kernel_mod->GetOutputSizeList();
  if (output_sizes.size() <= index) {
    MS_LOG(EXCEPTION) << "Previous node output size < node index";
  }
  std::string output_format = AnfAlgo::GetOutputFormat(anf_node, index);
  auto output_type = AnfAlgo::GetOutputDeviceDataType(anf_node, index);
  auto address = CreateDeviceAddress(nullptr, output_sizes[index], output_format, output_type);
  AnfAlgo::SetOutputAddr(address, index, anf_node.get());
  return address;
}

void KernelRuntime::AssignCommunicationNodeInputMem(MemType type, const AnfNodePtr &node) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  size_t total_size = 0;
  std::vector<std::pair<DeviceAddressPtr, size_t>> addr_size;
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(node); ++i) {
    auto input_node_with_index = AnfAlgo::GetPrevNodeOutput(node, i);
    auto input_node = input_node_with_index.first;
    DeviceAddressPtr address = nullptr;
    if (input_node->isa<CNode>()) {
      address = PreAssignCNodeMemory(input_node, input_node_with_index.second);
    } else {
      MS_LOG(EXCEPTION) << "Communication node inputs only support CNode";
    }
    MS_EXCEPTION_IF_NULL(address);
    auto mem_size = mem_manager_->GetCommonAlignSize(address->size());
    total_size += mem_size;
    addr_size.emplace_back(address, mem_size);
  }
  if (addr_size.empty()) {
    return;
  }

  if (type == kReuseDynamicMem || type == kSomasReuseDynamicMem) {
    bool not_reuse = KernelMemNotReuse(node);
    if (not_reuse) {
      type = kDynamicMem;
      MS_LOG(INFO) << "Disable Memory Reuse for " << node->fullname_with_scope() << "'s input.";
    }
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().size() < 2) {
    // communication node's input should contain itself and at least on input
    MS_LOG(ERROR) << "No inputs for " << cnode->fullname_with_scope();
    return;
  }
  auto first_input_node = cnode->input(1);
  auto prenode_index = AnfAlgo::VisitKernelWithReturnType(first_input_node, 0, true);
  uint8_t *input_ptr = mem_manager_->MallocOutputMem(prenode_index.first, prenode_index.second, type, total_size,
                                                     addr_size[0].first, true);
  for (const auto &iter : addr_size) {
    MS_EXCEPTION_IF_NULL(iter.first);
    iter.first->set_ptr(input_ptr);
    input_ptr += iter.second;
  }
}

void KernelRuntime::AssignNodeOutputMem(MemType type, const AnfNodePtr &node, int index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (AnfAlgo::IsGetNext(NOT_NULL(node)) && type == kReuseDynamicMem) {
    MS_LOG(INFO) << "GetNext disable mem_reuse";
    type = kDynamicMem;
  }

  if (node->isa<CNode>()) {
    bool independent = AnfAlgo::IsIndependentNode(node->cast<CNodePtr>());
    if (independent && (type == kReuseDynamicMem || type == kSomasReuseDynamicMem)) {
      MS_LOG(INFO) << "Independent node " << node->fullname_with_scope() << " disable memory reuse";
      type = kDynamicMem;
    }
  }

  if (type == kReuseDynamicMem || type == kSomasReuseDynamicMem) {
    bool not_reuse = KernelMemNotReuse(node);
    if (not_reuse) {
      type = kDynamicMem;
      MS_LOG(INFO) << "Disable Memory Reuse for " << node->fullname_with_scope() << "'s output.";
    }
  }

  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_sizes = kernel_mod->GetOutputSizeList();
  if (output_sizes.empty()) {
    MS_LOG(INFO) << "This kernel[" << node->DebugString() << "] has no output size.";
    return;
  }
  for (size_t i = 0; i < output_sizes.size(); ++i) {
    if ((kGetAllOuts != index) && (SizeToInt(i) != index)) {
      continue;
    }
    if (NodeOutputDeviceAddressExist(node, i)) {
      MS_LOG(INFO) << "Already malloc index:" << i;
      continue;
    }
    MS_LOG(DEBUG) << "Assign Node:" << node->fullname_with_scope() << " output memory size:" << output_sizes[i];
    std::string output_format = AnfAlgo::GetOutputFormat(node, i);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(node, i);
    auto device_address = CreateDeviceAddress(nullptr, output_sizes[i], output_format, output_type);
    MS_EXCEPTION_IF_NULL(device_address);
    uint8_t *ptr = mem_manager_->MallocOutputMem(node, i, type, output_sizes[i], device_address, false);
    MS_EXCEPTION_IF_NULL(ptr);
    device_address->set_host_shape(trans::GetRuntimePaddingShape(node, i));
    AnfAlgo::SetOutputAddr(device_address, i, node.get());
  }
}

void KernelRuntime::AssignValueNodeTensor(const ValueNodePtr &value_node, const ValuePtr &node_value,
                                          size_t output_idx) {
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(node_value);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::vector<tensor::TensorPtr> tensors;
  TensorValueToTensor(node_value, &tensors);
  for (const auto &tensor : tensors) {
    if (tensor == nullptr) {
      MS_LOG(WARNING) << "Tensor is null";
      return;
    }
    if (tensor->device_address() != nullptr) {
      AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address()), output_idx++,
                             value_node.get());
      continue;
    }
    size_t tensor_size = tensor->data().nbytes();
    auto node_size = CountNodeDeviceMemorySize(value_node, output_idx);
    TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(value_node, output_idx);
    if (output_type_id == kTypeUnknown) {
      output_type_id = AnfAlgo::GetOutputInferDataType(value_node, output_idx);
    }
    auto output_format = AnfAlgo::GetOutputFormat(value_node, output_idx);
    DeviceAddressPtr address = nullptr;
    address = CreateDeviceAddress(nullptr, node_size, output_format, output_type_id);
    MS_EXCEPTION_IF_NULL(address);
    if (ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER) &&
        !mem_manager_->MallocMemFromMemPool(address, node_size)) {
      MS_LOG(EXCEPTION) << "Cannot alloc address from memory pool when tensor size is: " << node_size;
    } else if (mem_manager_->MallocMem(kStaticMem, node_size, address) == nullptr) {
      MS_LOG(EXCEPTION) << "Cannot alloc address when flag is: " << kStaticMem << ", tensor size is: " << node_size;
    }
    AnfAlgo::SetOutputAddr(address, output_idx, value_node.get());
    if (!address->SyncHostToDevice(trans::GetRuntimePaddingShape(value_node, 0), tensor_size, tensor->data_type(),
                                   tensor->data_c())) {
      MS_EXCEPTION(NotExistsError) << "ValueNode SyncHostToDevice fail!" << value_node->DebugString()
                                   << "node format is" << AnfAlgo::GetOutputFormat(value_node, output_idx)
                                   << "node dtype is " << AnfAlgo::GetOutputInferDataType(value_node, output_idx);
    }
  }
}

void KernelRuntime::AssignStaticMemoryValueNode(session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  MS_LOG(INFO) << "AssignStaticMemoryValueNode start";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  for (auto &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    if (NodeOutputDeviceAddressExist(value_node, 0)) {
      MS_LOG(INFO) << "value_node[" << value_node->DebugString() << "] address already exist";
      continue;
    }
    auto &node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    MS_LOG(DEBUG) << "Malloc memory for " << value_node->fullname_with_scope();
    if (node_value->isa<Tensor>() || node_value->isa<ValueTuple>()) {
      AssignValueNodeTensor(value_node, node_value, 0);
    } else if (node_value->isa<StringImm>()) {
      auto value = GetValue<std::string>(node_value);
      size_t tensor_size = value.size();
      DeviceAddressPtr address = nullptr;
      address = CreateDeviceAddress(nullptr, tensor_size, kOpFormat_DEFAULT, kNumberTypeUInt8);
      MS_EXCEPTION_IF_NULL(address);
      if (ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER) &&
          !mem_manager_->MallocMemFromMemPool(address, tensor_size)) {
        MS_LOG(EXCEPTION) << "Cannot alloc address from memory pool when tensor size is: " << tensor_size;
      } else if (mem_manager_->MallocMem(kStaticMem, tensor_size, address) == nullptr) {
        MS_LOG(EXCEPTION) << "Cannot alloc address when flag is: " << kStaticMem << ", tensor size is: " << tensor_size;
      }
      AnfAlgo::SetOutputAddr(address, 0, value_node.get());
      ShapeVector shape = {1, SizeToLong(tensor_size)};
      if (!address->SyncHostToDevice(shape, tensor_size, kNumberTypeUInt8, value.data())) {
        MS_LOG(EXCEPTION) << "kValueNode SyncHostToDevice fail!";
      }
    }
  }
  MS_LOG(INFO) << "AssignStaticMemoryValueNode end";
}

void KernelRuntime::AssignDynamicMemory(session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool is_enable_mem_reuse = context_ptr->get_param<bool>(MS_CTX_ENABLE_MEM_REUSE);
  auto mem_type = kDynamicMem;
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (dump_json_parser.e2e_dump_enabled() && dump_json_parser.dump_mode() == 0) {
    context_ptr->set_param<bool>(MS_CTX_ENABLE_MEM_REUSE, false);
    is_enable_mem_reuse = false;
    MS_LOG(INFO) << "Disable Memory Reuse when e2e dump is enable and dump mode is set to dump all kernels";
  }

  if (is_enable_mem_reuse) {
    MS_LOG(INFO) << "Memory Reuse is enable...";
#ifdef MEM_REUSE_DEBUG
    mem_manager_->MallocReusedDynamicMem(graph);
    mem_type = kReuseDynamicMem;
#else
    mem_manager_->MallocSomasDynamicMem(graph);
    mem_type = kSomasReuseDynamicMem;
#endif
  } else {
    MS_LOG(INFO) << "Memory Reuse is disable...";
  }
  auto &execution_nodes = graph->execution_order();
  std::vector<CNodePtr> compute_nodes;
  // communication nodes first
  for (auto &node : execution_nodes) {
    if (AnfAlgo::IsCommunicationOp(node)) {
      // skip if the memory is already allocated
      AssignCommunicationNodeMem(mem_type, node);
    } else {
      compute_nodes.emplace_back(node);
    }
  }

  // then compute nodes
  for (auto &node : compute_nodes) {
    AssignNodeOutputMem(mem_type, node, kGetAllOuts);
    AssignWorkSpaceMem(mem_type, node);
  }
}

void KernelRuntime::AssignWorkSpaceMem(MemType type, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  size_t index = 0;
  for (auto &size : kernel_mod->GetWorkspaceSizeList()) {
    auto ptr = mem_manager_->MallocWorkSpaceMem(node, index, type, size);
    AnfAlgo::SetWorkspaceAddr(CreateDeviceAddress(ptr, size, "", kTypeUnknown), index, node.get());
    index++;
  }
}

void KernelRuntime::GenLaunchArgs(const mindspore::kernel::KernelMod &kernel_mod, const mindspore::AnfNodePtr &kernel,
                                  AddressPtrList *kernel_inputs, AddressPtrList *const kernel_workspaces,
                                  AddressPtrList *kernel_outputs) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_inputs);
  MS_EXCEPTION_IF_NULL(kernel_workspaces);
  MS_EXCEPTION_IF_NULL(kernel_outputs);
  auto cnode = kernel->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::GetCNodeName(cnode) == kAtomicAddrCleanOpName) {
    return GenAddrCleanLaunchArgs(cnode, kernel_inputs);
  }
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(kernel); ++i) {
    auto real_input = AnfAlgo::GetRealInputIndex(kernel, i);
    auto device_address = AnfAlgo::GetPrevNodeOutputAddr(kernel, real_input);
    MS_EXCEPTION_IF_NULL(device_address);
    kernel::AddressPtr input = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(input);
    input->addr = device_address->ptr_;
    MS_EXCEPTION_IF_NULL(input->addr);
    input->size = device_address->size_;
    kernel_inputs->emplace_back(input);
  }

  for (size_t i = 0; i < kernel_mod.GetOutputSizeList().size(); ++i) {
    auto device_address = AnfAlgo::GetOutputAddr(kernel, i);
    kernel::AddressPtr output = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(output);
    output->addr = device_address->ptr_;
    MS_EXCEPTION_IF_NULL(output->addr);
    output->size = device_address->size_;
    kernel_outputs->emplace_back(output);
  }

  for (size_t i = 0; i < kernel_mod.GetWorkspaceSizeList().size(); ++i) {
    auto device_address = AnfAlgo::GetWorkspaceAddr(kernel, i);
    kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(workspace);
    workspace->addr = device_address->ptr_;
    MS_EXCEPTION_IF_NULL(workspace->addr);
    workspace->size = device_address->size_;
    kernel_workspaces->emplace_back(workspace);
  }
}

void KernelRuntime::GenAddrCleanLaunchArgs(const CNodePtr &cnode, AddressPtrList *kernel_inputs) {
  if (cnode->inputs().size() != 2) {
    MS_LOG(EXCEPTION) << "Atomic Addr clean Node Input nodes not equal 2.";
  }
  MS_EXCEPTION_IF_NULL(cnode->inputs()[1]);
  auto pre_node = (cnode->inputs()[1])->cast<CNodePtr>();
  // set clean output address
  if (AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, pre_node)) {
    auto clean_output_indexes = AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicOutputIndexs);
    for (auto index : clean_output_indexes) {
      auto device_address = AnfAlgo::GetOutputAddr(pre_node, index);
      kernel::AddressPtr input = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(input);
      input->addr = device_address->ptr_;
      MS_EXCEPTION_IF_NULL(input->addr);
      input->size = device_address->size_;
      kernel_inputs->emplace_back(input);
    }
    MS_LOG(INFO) << "AtomicAddClean clean output size:" << clean_output_indexes.size();
  }
  // set clean workspace address
  if (AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, pre_node)) {
    auto clean_workspaces_indexes = AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicWorkspaceIndexs);
    for (const auto &index : clean_workspaces_indexes) {
      auto device_address = AnfAlgo::GetWorkspaceAddr(pre_node, index);
      kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(workspace);
      workspace->addr = device_address->ptr_;
      MS_EXCEPTION_IF_NULL(workspace->addr);
      workspace->size = device_address->size_;
      kernel_inputs->emplace_back(workspace);
    }
  }
}

bool KernelRuntime::LaunchKernelMod(const session::KernelGraph &graph) {
  auto &kernels = graph.execution_order();
  std::vector<DynamicKernelPtr> dynamic_kernel_list;
  auto iter = graph_dynamic_kernel_map_.find(graph.graph_id());
  if (iter != graph_dynamic_kernel_map_.end()) {
    dynamic_kernel_list = iter->second;
  }
  if (!dynamic_kernel_list.empty() && dynamic_kernel_list.size() != kernels.size()) {
    MS_LOG(EXCEPTION) << "The size of dynamic kernels " << dynamic_kernel_list.size()
                      << " should be equal to the size of kernels " << kernels.size();
  }
  for (size_t i = 0; i < kernels.size(); ++i) {
    if (!dynamic_kernel_list.empty() && dynamic_kernel_list[i] != nullptr &&
        dynamic_kernel_list[i]->is_dynamic_shape()) {
      dynamic_kernel_list[i]->InferShape();
      dynamic_kernel_list[i]->UpdateArgs();
      dynamic_kernel_list[i]->Execute();
      if (!SyncStream()) {
        MS_LOG(ERROR) << "SyncStream failed";
        return false;
      }
      dynamic_kernel_list[i]->PostExecute();
    } else {
      auto kernel_mod = AnfAlgo::GetKernelMod(kernels[i]);
      MS_EXCEPTION_IF_NULL(kernel_mod);
      AddressPtrList kernel_inputs;
      AddressPtrList kernel_workspaces;
      AddressPtrList kernel_outputs;
      GenLaunchArgs(*kernel_mod, kernels[i], &kernel_inputs, &kernel_workspaces, &kernel_outputs);
      auto ret = kernel_mod->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, stream_);
      if (!ret) {
        MS_LOG(ERROR) << "Launch kernel failed.";
        return false;
      }
    }
  }
  return true;
}

bool KernelRuntime::LaunchKernel(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (!LaunchKernelMod(*graph)) {
    MS_LOG(ERROR) << "LaunchKernelMod failed!";
    return false;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    if (!SyncStream()) {
      MS_LOG(ERROR) << "SyncStream failed";
      return false;
    }
  }
  return true;
}

void KernelRuntime::ClearGraphRuntimeResource(uint32_t graph_id, const std::vector<AnfNodePtr> &,
                                              const std::unordered_set<ValueNodePtr> &, const std::vector<CNodePtr> &) {
  MS_LOG(INFO) << "Clear graph:" << graph_id << " runtime resource";
}

void KernelRuntime::ClearOutputAddress(const std::vector<AnfNodePtr> &inputs,
                                       const std::unordered_set<ValueNodePtr> &value_nodes,
                                       const std::vector<CNodePtr> &execution_order) {
  // clear input parameter output address.
  for (const auto &input_node : inputs) {
    MS_EXCEPTION_IF_NULL(input_node);
    if (!input_node->isa<Parameter>()) {
      continue;
    }
    auto parameter = input_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(parameter);
    parameter->DecreaseUsedGraphCount();
    // Only the parameter has no graph used, then clear the output address.
    if (parameter->used_graph_count() != 0) {
      continue;
    }
    for (size_t index = 0; index < AnfAlgo::GetOutputTensorNum(input_node); ++index) {
      if (!AnfAlgo::OutputAddrExist(input_node, index)) {
        continue;
      }
      AnfAlgo::SetOutputAddr(nullptr, index, input_node.get());
    }
  }
  // clear input value node output address.
  for (const auto &value_node : value_nodes) {
    if (!AnfAlgo::OutputAddrExist(value_node, 0)) {
      continue;
    }
    AnfAlgo::SetOutputAddr(nullptr, 0, value_node.get());
  }
  // clear cnode output address.
  for (const auto &cnode : execution_order) {
    for (size_t index = 0; index < AnfAlgo::GetOutputTensorNum(cnode); ++index) {
      if (!AnfAlgo::OutputAddrExist(cnode, index)) {
        continue;
      }
      AnfAlgo::SetOutputAddr(nullptr, index, cnode.get());
    }
  }
}

bool KernelRuntime::LaunchTaskBasedOnSingleKernel(kernel::KernelModPtr kernel_mod_ptr,
                                                  const AddressPtrList &kernel_inputs,
                                                  const AddressPtrList &kernel_outputs,
                                                  const AddressPtrList &kernel_workspaces) const {
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  auto ret = kernel_mod_ptr->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, stream_);
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel failed.";
    return false;
  }
  return true;
}

DeviceAddressPtr KernelRuntime::AssignSingleOpLaunchMemory(size_t size, const std::string &format, TypeId type) {
  auto device_address = CreateDeviceAddress(nullptr, size, format, type);
  MS_EXCEPTION_IF_NULL(device_address);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto base_ptr = mem_manager_->MallocMem(kStaticMem, size, device_address);
  MS_EXCEPTION_IF_NULL(base_ptr);
  return device_address;
}
}  // namespace device
}  // namespace mindspore
