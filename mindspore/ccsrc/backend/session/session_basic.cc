/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "backend/session/session_basic.h"
#include <utility>
#include <algorithm>
#include <unordered_map>

#include "c_ops/primitive_c.h"
#include "ir/manager.h"
#include "ir/param_info.h"
#include "backend/kernel_compiler/common_utils.h"
#include "base/core_ops.h"
#include "common/trans.h"
#include "utils/config_manager.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/executor.h"
#include "backend/session/executor_manager.h"
#include "backend/optimizer/common/common_backend_optimization.h"
#include "backend/optimizer/common/helper.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "utils/ms_utils.h"
#include "ir/dtype.h"
#include "ir/anf.h"
#include "ir/func_graph_cloner.h"
#include "utils/utils.h"
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
#include "ps/worker.h"
#include "ps/common.h"
#include "ps/util.h"
#endif

namespace mindspore {
namespace session {
static std::shared_ptr<std::map<ValuePtr, ParameterPtr>> python_paras;
void ClearPythonParasMap() { python_paras = nullptr; }
namespace {
const int kSummaryGetItem = 2;

ValuePtr GetParamDefaultValue(const AnfNodePtr &node) {
  if (node == nullptr) {
    return nullptr;
  }
  auto parameter = node->cast<ParameterPtr>();
  if (parameter == nullptr || !parameter->has_default()) {
    return nullptr;
  }
  return parameter->default_param();
}

tensor::TensorPtr CreateCNodeOutputTensor(const session::KernelWithIndex &node_output_pair,
                                          const KernelGraphPtr &graph) {
  auto &node = node_output_pair.first;
  auto &output_index = node_output_pair.second;
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  TypeId type_id = AnfAlgo::GetOutputDeviceDataType(node, output_index);
  if (type_id == kTypeUnknown) {
    type_id = AnfAlgo::GetOutputInferDataType(node, output_index);
  }
  tensor::TensorPtr tensor = nullptr;
  std::vector<int64_t> temp_shape;
  if (graph->IsUniqueTargetInternalOutput(node, output_index)) {
    temp_shape.emplace_back(1);
    tensor = std::make_shared<tensor::Tensor>(type_id, temp_shape);
    tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(node, output_index));
    tensor->set_sync_status(kNoNeedSync);
    tensor->SetNeedWait(true);
    tensor->SetIsGraphOutput();
    return tensor;
  }

  tensor = graph->GetInternalOutputTensor(node, output_index);
  if (tensor == nullptr) {
    auto shape = AnfAlgo::GetOutputInferShape(node, output_index);
    (void)std::copy(shape.begin(), shape.end(), std::back_inserter(temp_shape));
    tensor = std::make_shared<tensor::Tensor>(type_id, temp_shape);
    bool is_internal_output = graph->IsInternalOutput(node, output_index);
    if (is_internal_output) {
      graph->AddInternalOutputTensor(node, output_index, tensor);
    }
  }
  tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(node, output_index));
  // if in paynative mode,data only copyed to host when user want to print data
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode &&
      ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kGPUDevice) {
    tensor->set_sync_status(kNeedSyncDeviceToHostImmediately);
  } else {
    tensor->set_sync_status(kNeedSyncDeviceToHost);
  }
  tensor->SetNeedWait(true);
  tensor->SetIsGraphOutput();
  return tensor;
}

BaseRef CreateNodeOutputTensor(const session::KernelWithIndex &node_output_pair, const KernelGraphPtr &graph,
                               const std::vector<tensor::TensorPtr> &input_tensors,
                               std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node) {
  auto &node = node_output_pair.first;
  auto &output_index = node_output_pair.second;
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(tensor_to_node);
  MS_LOG(INFO) << "Create tensor for output[" << node->DebugString() << "] index[" << node_output_pair.second << "]";
  // if node is a value node, no need sync addr from device to host
  if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    return value_node->value();
  }
  if (!AnfAlgo::OutputAddrExist(node, output_index)) {
    if (node->isa<Parameter>()) {
      for (size_t input_idx = 0; input_idx < graph->inputs().size(); input_idx++) {
        if (input_idx >= input_tensors.size()) {
          MS_LOG(EXCEPTION) << "Input idx:" << input_idx << "out of range:" << input_tensors.size();
        }
        if (graph->inputs()[input_idx] == node) {
          return input_tensors[input_idx];
        }
      }
      MS_LOG(EXCEPTION) << "Parameter : " << node->DebugString() << " has no output addr";
    }
  }
  auto tensor = CreateCNodeOutputTensor(node_output_pair, graph);
  (*tensor_to_node)[tensor] = node_output_pair;
  return tensor;
}

BaseRef CreateNodeOutputTensors(const AnfNodePtr &anf, const KernelGraphPtr &graph,
                                const std::vector<tensor::TensorPtr> &input_tensors,
                                std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(tensor_to_node);
  MS_LOG(INFO) << "Create tensor for output[" << anf->DebugString() << "]";
  auto item_with_index = AnfAlgo::VisitKernelWithReturnType(anf, 0);
  MS_EXCEPTION_IF_NULL(item_with_index.first);
  MS_LOG(INFO) << "Create tensor for output after visit:" << item_with_index.first->DebugString();
  // special handle for maketuple
  if (AnfAlgo::CheckPrimitiveType(item_with_index.first, prim::kPrimMakeTuple)) {
    auto cnode = item_with_index.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    VectorRef ret;
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      auto out = CreateNodeOutputTensors(cnode->input(i), graph, input_tensors, tensor_to_node);
      ret.push_back(out);
    }
    return ret;
  }
  // if is graph return nothing ,the function should return a null anylist
  size_t size = AnfAlgo::GetOutputTensorNum(item_with_index.first);
  if (size == 0) {
    return VectorRef();
  }
  return CreateNodeOutputTensor(item_with_index, graph, input_tensors, tensor_to_node);
}

ValueNodePtr CreateNewValueNode(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  auto value_node = anf->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<None>()) {
    return nullptr;
  }
  auto new_value_node = graph->NewValueNode(value_node);
  graph->FrontBackendlMapAdd(anf, new_value_node);
  graph->AddValueNodeToGraph(new_value_node);
  return new_value_node;
}

size_t LoadCtrlInputTensor(const std::shared_ptr<KernelGraph> &graph, std::vector<tensor::TensorPtr> *inputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Load kInputCtrlTensors";
  auto inputs_params = graph->input_ctrl_tensors();
  if (inputs_params == nullptr) {
    return 0;
  }
  if (inputs_params->size() < 3) {
    MS_LOG(EXCEPTION) << "Illegal inputs_params size";
  }
  // update current loop tensor to 0 per iterator
  auto cur_loop_tensor = (*inputs_params)[0];
  MS_EXCEPTION_IF_NULL(cur_loop_tensor);
  auto *cur_val = static_cast<int32_t *>(cur_loop_tensor->data_c());
  MS_EXCEPTION_IF_NULL(cur_val);
  *cur_val = 0;
  cur_loop_tensor->set_sync_status(kNeedSyncHostToDevice);
  // set loop_count to zero
  MS_EXCEPTION_IF_NULL(inputs);
  inputs->push_back(cur_loop_tensor);

  // update next loop tensor to 0 per iterator
  auto next_loop_tensor = (*inputs_params)[1];
  MS_EXCEPTION_IF_NULL(next_loop_tensor);
  auto *next_val = static_cast<int32_t *>(next_loop_tensor->data_c());
  MS_EXCEPTION_IF_NULL(next_val);
  *next_val = 0;
  next_loop_tensor->set_sync_status(kNeedSyncHostToDevice);
  // set loop_count to zero
  MS_EXCEPTION_IF_NULL(inputs);
  inputs->push_back(next_loop_tensor);

  auto epoch_tensor = (*inputs_params)[2];
  MS_EXCEPTION_IF_NULL(epoch_tensor);
  auto *epoch_val = static_cast<int32_t *>(epoch_tensor->data_c());
  MS_EXCEPTION_IF_NULL(epoch_val);
  *epoch_val = graph->current_epoch();
  epoch_tensor->set_sync_status(kNeedSyncHostToDevice);
  inputs->push_back(epoch_tensor);
  MS_LOG(INFO) << "Load epoch_val:" << *epoch_val;

  graph->set_current_epoch(graph->current_epoch() + 1);

  return inputs_params->size();
}

ValueNodePtr ConstructRunOpValueNode(const std::shared_ptr<KernelGraph> &graph, const tensor::TensorPtr &input_tensor) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto value_node = std::make_shared<ValueNode>(input_tensor);
  MS_EXCEPTION_IF_NULL(value_node);
  // construct abstract of value node
  auto type_of_tensor = input_tensor->Dtype();
  auto shape_of_tensor = input_tensor->shape();
  auto abstract = std::make_shared<abstract::AbstractTensor>(type_of_tensor, shape_of_tensor);
  value_node->set_abstract(abstract);
  // add value node to graph
  auto input_value_node = graph->NewValueNode(value_node);
  graph->AddValueNodeToGraph(input_value_node);
  return input_value_node;
}

ParameterPtr ConstructRunOpParameter(const std::shared_ptr<KernelGraph> &graph, const tensor::TensorPtr &input_tensor,
                                     int64_t tensor_mask) {
  MS_EXCEPTION_IF_NULL(graph);
  auto param = graph->NewParameter();
  MS_EXCEPTION_IF_NULL(param);
  if (tensor_mask == kParameterWeightTensorMask) {
    param->set_default_param(input_tensor);
  }
  // set the kernel info of parameter
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(input_tensor->device_address());
  if (device_address == nullptr) {
    kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
    TypeId param_init_data_type = AnfAlgo::IsParameterWeight(param) ? kTypeUnknown : input_tensor->data_type();
    kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{param_init_data_type});
  } else {
    kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{device_address->format()});
    kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{device_address->type_id()});
    kernel_build_info_builder->SetOutputsReshapeType({input_tensor->padding_type()});
    AnfAlgo::SetOutputAddr(device_address, 0, param.get());
  }
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), param.get());
  // construct abstract of parameter
  auto type_of_tensor = input_tensor->Dtype();
  auto shape_of_tensor = input_tensor->shape();
  auto abstract = std::make_shared<abstract::AbstractTensor>(type_of_tensor, shape_of_tensor);
  param->set_abstract(abstract);
  return param;
}

void DumpGraphOutput(const Any &any, size_t recurse_level = 0) {
  MS_LOG(INFO) << "Graph outputs:";
  const size_t max_deep = 10;
  if (recurse_level > max_deep) {
    MS_LOG(INFO) << "Recurse too deep";
    return;
  }
  std::string tab_str;
  for (size_t i = 0; i < recurse_level; i++) {
    tab_str = tab_str.append("  ");
  }
  if (any.is<AnyList>()) {
    (void)tab_str.append("{");
    MS_LOG(INFO) << tab_str;
    auto any_list = any.cast<AnyList>();
    for (auto &it : any_list) {
      DumpGraphOutput(it, recurse_level + 1);
    }
    (void)tab_str.append("}");
    MS_LOG(INFO) << tab_str;
  }
  (void)tab_str.append(any.ToString());
  MS_LOG(INFO) << tab_str;
}

bool ExistSummaryNode(const KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto ret = graph->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  auto all_nodes = DeepLinkedGraphSearch(ret);
  for (auto &n : all_nodes) {
    if (IsPrimitiveCNode(n, prim::kPrimScalarSummary) || IsPrimitiveCNode(n, prim::kPrimTensorSummary) ||
        IsPrimitiveCNode(n, prim::kPrimImageSummary) || IsPrimitiveCNode(n, prim::kPrimHistogramSummary)) {
      return true;
    }
  }
  return false;
}

bool IgnoreCreateParameterForMakeTuple(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfAlgo::CheckPrimitiveType(node, prim::kPrimMakeTuple)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &node_inputs = cnode->inputs();
  for (size_t i = 1; i < node_inputs.size(); ++i) {
    if (!AnfAlgo::CheckPrimitiveType(node_inputs[i], prim::kPrimControlDepend)) {
      return false;
    }
  }
  return true;
}
}  // namespace

GraphId SessionBasic::graph_sum_ = 0;

void SessionBasic::InitDevice(const std::string &device_name, uint32_t device_id) {
  device_id_ = device_id;
  context_ = std::make_shared<Context>(device_name, device_id);
  executor_ = ExecutorManager::Instance().GetExecutor(device_name, device_id);
}

KernelGraphPtr SessionBasic::GetGraph(mindspore::GraphId graph_id) const {
  auto it = graphs_.find(graph_id);
  if (it == graphs_.end()) {
    MS_LOG(WARNING) << "Can't find graph " << graph_id;
    return nullptr;
  }
  return it->second;
}

void SessionBasic::InitInternalOutputParameter(const AnfNodePtr &out_node, const AnfNodePtr &parameter) {
  auto graph_id = GetGraphIdByNode(out_node);
  if (graph_id == kInvalidGraphId) {
    return;
  }
  auto node_graph = GetGraph(graph_id);
  if (node_graph == nullptr) {
    return;
  }
  MS_LOG(INFO) << "Init parameter with pre graph output node: " << out_node->DebugString();
  auto ref_node = node_graph->GetInternalOutputByFrontNode(out_node);
  if (ref_node == nullptr) {
    MS_LOG(INFO) << "No corresponding internal output for output node";
    return;
  }
  size_t output_idx = 0;
  if (AnfAlgo::CheckPrimitiveType(out_node, prim::kPrimTupleGetItem)) {
    output_idx = AnfAlgo::GetTupleGetItemOutIndex(out_node->cast<CNodePtr>());
  }
  auto real_kernel = AnfAlgo::VisitKernel(ref_node, output_idx);
  auto ref_real_node = real_kernel.first;
  auto ref_real_node_index = real_kernel.second;
  if (ref_real_node->isa<CNode>() && node_graph->IsUniqueTargetInternalOutput(ref_real_node, ref_real_node_index)) {
    auto kernel_info = ref_real_node->kernel_info();
    if (kernel_info == nullptr || !kernel_info->has_build_info()) {
      MS_LOG(INFO) << "No kernel info";
      return;
    }
    if (!opt::IsNopNode(ref_real_node) && !AnfAlgo::OutputAddrExist(ref_real_node, ref_real_node_index)) {
      MS_LOG(INFO) << "No kernel address";
      return;
    }
    auto address = AnfAlgo::GetMutableOutputAddr(ref_real_node, ref_real_node_index);
    auto format = AnfAlgo::GetOutputFormat(ref_real_node, ref_real_node_index);
    auto type = AnfAlgo::GetOutputDeviceDataType(ref_real_node, ref_real_node_index);
    auto d_kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(d_kernel_info);
    parameter->set_kernel_info(d_kernel_info);
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    builder.SetOutputsDeviceType({type});
    builder.SetOutputsFormat({format});
    d_kernel_info->set_select_kernel_build_info(builder.Build());
    AnfAlgo::SetOutputAddr(address, 0, parameter.get());
    AnfAlgo::SetOutputInferTypeAndShape({type}, {AnfAlgo::GetOutputInferShape(parameter, 0)}, parameter.get());
  }
}

std::vector<AnfNodePtr> SessionBasic::CreateParameterFromTuple(const AnfNodePtr &node, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> parameters;
  std::vector<AnfNodePtr> pre_graph_out = {node};
  if (IgnoreCreateParameterForMakeTuple(node)) {
    pre_graph_out.clear();
  }
  // If a cnode is a call, it's input0 is a cnode too, so it doesn't have primitive
  if (!pre_graph_out.empty() && !AnfAlgo::IsRealKernel(node)) {
    pre_graph_out = AnfAlgo::GetAllOutput(node, {prim::kPrimTupleGetItem});
  }
  auto valid_inputs = graph->MutableValidInputs();
  MS_EXCEPTION_IF_NULL(valid_inputs);
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  auto create_parameter = [&](const AbstractBasePtr &abstract) -> void {
    auto parameter = graph->NewParameter();
    MS_EXCEPTION_IF_NULL(parameter);
    parameter->set_abstract(abstract);
    auto new_parameter = graph->NewParameter(parameter);
    parameters.push_back(new_parameter);
    valid_inputs->push_back(true);
    graph_inputs->push_back(new_parameter);
  };
  for (const auto &out_node : pre_graph_out) {
    MS_EXCEPTION_IF_NULL(out_node);
    auto abstract = out_node->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    // create multiple parameters if is a tuple output real kernel
    if (abstract->isa<abstract::AbstractTuple>() && !AnfAlgo::CheckPrimitiveType(out_node, prim::kPrimTupleGetItem)) {
      auto tuple_abstract = abstract->cast<abstract::AbstractTuplePtr>();
      MS_EXCEPTION_IF_NULL(tuple_abstract);
      MS_LOG(INFO) << "Tuple_size [" << tuple_abstract->size() << "]";
      for (size_t output_idx = 0; output_idx < tuple_abstract->size(); output_idx++) {
        create_parameter((*tuple_abstract)[output_idx]);
      }
      continue;
    }
    // create single parameter if is a abstract real kernel
    create_parameter(out_node->abstract());
    InitInternalOutputParameter(out_node, parameters[parameters.size() - 1]);
  }
  return parameters;
}

ParameterPtr SessionBasic::CreateNewParameterFromParameter(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  if (!anf->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "Anf[" << anf->DebugString() << "] is not a parameter";
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto param_value = GetParamDefaultValue(anf);
  auto valid_inputs = graph->MutableValidInputs();
  MS_EXCEPTION_IF_NULL(valid_inputs);
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  ParameterPtr new_parameter = nullptr;
  // if parameter's python parameter has been exist a backend parameter, reuse the exist parameter
  if (python_paras == nullptr) {
    python_paras = std::make_shared<std::map<ValuePtr, ParameterPtr>>();
  }
  auto iter = python_paras->find(param_value);
  if (iter != python_paras->end()) {
    new_parameter = iter->second;
  } else {
    TraceManager::DebugTrace(std::make_shared<TraceCopy>(anf->debug_info()));
    new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
    if (param_value != nullptr) {
      (*python_paras)[param_value] = new_parameter;
    }
    TraceManager::EndTrace();
  }
  new_parameter->IncreaseUsedGraphCount();
  graph_inputs->push_back(new_parameter);
  valid_inputs->push_back(true);
  return new_parameter;
}

AnfNodePtr SessionBasic::CreateNewParameterFromCNode(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Create a new parameter from cnode[" << anf->DebugString() << "]";
  auto parameters = CreateParameterFromTuple(anf, graph);
  if (parameters.empty()) {
    MS_LOG(INFO) << "Empty parameter from cnode";
    return nullptr;
  }
  if (parameters.size() == 1) {
    return parameters[0];
  }
  std::vector<AnfNodePtr> make_tuple_input = {NewValueNode(prim::kPrimMakeTuple)};
  (void)std::copy(parameters.begin(), parameters.end(), std::back_inserter(make_tuple_input));
  auto make_tuple = graph->NewCNode(make_tuple_input);
  MS_EXCEPTION_IF_NULL(make_tuple);
  MS_LOG(INFO) << "New make tuple [" << make_tuple->DebugString() << "] of parameters";
  return make_tuple;
}

void SessionBasic::GetCNodeInfo(const CNodePtr &cnode, std::vector<AnfNodePtr> *cnode_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  auto prim = AnfAlgo::GetCNodePrimitive(cnode);
  if (prim != nullptr) {
    // push attr to inputs[0] of new cnode
    cnode_inputs->push_back(std::make_shared<ValueNode>(std::make_shared<Primitive>(*prim)));
  } else {
    auto fg = AnfAlgo::GetCNodeFuncGraphPtr(cnode);
    MS_EXCEPTION_IF_NULL(fg);
    auto new_fg = BasicClone(fg);
    cnode_inputs->push_back(std::make_shared<ValueNode>(new_fg));
  }
}

void SessionBasic::GetNewCNodeInputs(const CNodePtr &cnode, KernelGraph *graph, std::vector<AnfNodePtr> *cnode_inputs,
                                     std::unordered_map<AnfNodePtr, AnfNodePtr> *other_graph_cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(other_graph_cnode);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  auto origin_inputs = cnode->inputs();
  bool optimize_depend = IsPrimitiveCNode(cnode, prim::kPrimDepend) && origin_inputs.size() == 3;
  bool optimize_control_depend = IsPrimitiveCNode(cnode, prim::kPrimControlDepend) && origin_inputs.size() == 3;
  // if has multiple depends,only select first depend as parameter
  for (size_t input_idx = 1; input_idx < origin_inputs.size(); input_idx++) {
    auto anf = origin_inputs[input_idx];
    MS_EXCEPTION_IF_NULL(anf);
    // anf has been created before
    if (graph->GetBackendAnfByFrontAnf(anf) != nullptr) {
      cnode_inputs->emplace_back(graph->GetBackendAnfByFrontAnf(anf));
      continue;
    } else if (optimize_depend && input_idx == kDependAttachNodeIndex) {
      cnode_inputs->push_back(NewValueNode(MakeValue(SizeToInt(input_idx))));
      continue;
    } else if (other_graph_cnode->find(anf) != other_graph_cnode->end()) {
      cnode_inputs->push_back((*other_graph_cnode)[anf]);
      continue;
    } else if (anf->isa<ValueNode>() && !IsValueNode<FuncGraph>(anf)) {
      // if input is a value node,
      auto new_value_node = CreateNewValueNode(anf, graph);
      if (new_value_node != nullptr) {
        cnode_inputs->emplace_back(new_value_node);
      }
      continue;
    } else if (anf->isa<Parameter>()) {
      auto new_parameter = CreateNewParameterFromParameter(anf, graph);
      cnode_inputs->push_back(new_parameter);
      if (GetGraphIdByNode(anf) == kInvalidGraphId) {
        graph->FrontBackendlMapAdd(anf, new_parameter);
      } else {
        (*other_graph_cnode)[anf] = new_parameter;
      }
      continue;
    } else if (optimize_control_depend) {
      cnode_inputs->push_back(NewValueNode(MakeValue(SizeToLong(input_idx))));
    } else {
      // the input node is a cnode from other graph
      auto parameter_from_cnode = CreateNewParameterFromCNode(anf, graph);
      if (parameter_from_cnode == nullptr) {
        parameter_from_cnode = NewValueNode(MakeValue(SizeToLong(input_idx)));
      }
      cnode_inputs->push_back(parameter_from_cnode);
      (*other_graph_cnode)[anf] = parameter_from_cnode;
    }
  }
}

CNodePtr SessionBasic::CreateNewCNode(const CNodePtr &cnode, KernelGraph *graph,
                                      std::unordered_map<AnfNodePtr, AnfNodePtr> *other_graph_cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(other_graph_cnode);
  // get primitive of old node
  std::vector<AnfNodePtr> cnode_inputs;
  GetCNodeInfo(cnode, &cnode_inputs);
  GetNewCNodeInputs(cnode, graph, &cnode_inputs, other_graph_cnode);
  TraceManager::DebugTrace(std::make_shared<TraceCopy>(cnode->debug_info()));
  auto new_cnode = graph->NewCNode(cnode_inputs);
  TraceManager::EndTrace();
  return new_cnode;
}

CNodePtr SessionBasic::CreateSwitchInput(const AnfNodePtr &node_input, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node_input);
  MS_EXCEPTION_IF_NULL(graph);
  // switch input generalizes partial
  std::vector<AnfNodePtr> partial_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimPartial->name()))};
  if (AnfAlgo::CheckPrimitiveType(node_input, prim::kPrimPartial)) {
    auto partial_node = graph->GetBackendAnfByFrontAnf(node_input);
    return partial_node->cast<CNodePtr>();
  } else if (node_input->isa<ValueNode>() && IsValueNode<FuncGraph>(node_input)) {
    partial_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(node_input));
  } else {
    KernelGraphPtr kernel_graph = NewKernelGraph();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto parameter = CreateNewParameterFromCNode(graph->GetBackendAnfByFrontAnf(node_input), kernel_graph.get());
    auto primitive = NewValueNode(std::make_shared<Primitive>(prim::kPrimReturn->name()));
    auto return_node = kernel_graph->NewCNode({primitive, parameter});
    kernel_graph->set_return(return_node);
    partial_inputs.emplace_back(std::make_shared<ValueNode>(kernel_graph));
    partial_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(node_input));
  }
  auto partial_node = graph->NewCNode(partial_inputs);
  return partial_node;
}

std::vector<AnfNodePtr> SessionBasic::CreateCallSwitchInputs(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto cnode_input = graph->GetBackendAnfByFrontAnf(attr_input);
  auto switch_cnode = cnode_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(switch_cnode);
  if (cnode->inputs().size() < 2) {
    cnode_inputs = switch_cnode->inputs();
    return cnode_inputs;
  }
  std::vector<AnfNodePtr> switch_inputs = {switch_cnode->input(kAnfPrimitiveIndex),
                                           switch_cnode->input(kFirstDataInputIndex)};
  for (size_t index = kFirstBranchInSwitch; index < switch_cnode->inputs().size(); index++) {
    auto node = switch_cnode->input(index);
    // there is real input in call, should put it to true and false branch in switch
    if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
      auto partial_node = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(partial_node);
      std::vector<AnfNodePtr> partial_inputs = partial_node->inputs();
      partial_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(cnode->input(kFirstDataInputIndex)));
      auto new_partial = graph->NewCNode(partial_inputs);
      switch_inputs.emplace_back(new_partial);
    }
  }
  if (switch_inputs.size() < kSwitchInputSize) {
    MS_LOG(EXCEPTION) << "Switch inputs size: " << switch_inputs.size() << "less than " << kSwitchInputSize;
  }
  auto switch_node = graph->NewCNode(switch_inputs);
  cnode_inputs.emplace_back(switch_node);
  return cnode_inputs;
}

std::vector<AnfNodePtr> SessionBasic::CreateSwitchOrPartialNode(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  // create primitive of cnode:call(partial or switch)
  std::vector<AnfNodePtr> cnode_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto cnode_input = graph->GetBackendAnfByFrontAnf(attr_input);
  if (cnode_input == nullptr) {
    MS_LOG(EXCEPTION) << "CNode input[0] is CNode:" << attr_input->DebugString()
                      << ", but input[0] has not been created.";
  }
  // if the node is partial, insert the inputs of partial to the call
  if (AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimPartial)) {
    auto partial_node = attr_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(partial_node);
    auto partial_inputs = partial_node->inputs();
    std::transform(partial_inputs.begin() + kFirstDataInputIndex, partial_inputs.end(),
                   std::back_inserter(cnode_inputs), [&graph](const AnfNodePtr &node) {
                     MS_EXCEPTION_IF_NULL(graph->GetBackendAnfByFrontAnf(node));
                     return graph->GetBackendAnfByFrontAnf(node);
                   });
    return cnode_inputs;
  } else if (AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimSwitch)) {
    return CreateCallSwitchInputs(cnode, graph);
  }
  MS_LOG(EXCEPTION) << "CNode input[0] must be partial or switch.";
}

std::vector<AnfNodePtr> SessionBasic::CreateValueNode(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs;
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  if (AnfAlgo::IsGraphKernel(cnode)) {
    auto fg = AnfAlgo::GetCNodeFuncGraphPtr(cnode);
    MS_EXCEPTION_IF_NULL(fg);
    auto new_fg = BasicClone(fg);
    cnode_inputs.push_back(std::make_shared<ValueNode>(new_fg));
  } else {
    // create primitive of cnode:call
    cnode_inputs = {graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
    // create a ValueNode<KernelGraph> as input of cnode:call
    if (graph->GetBackendAnfByFrontAnf(attr_input) != nullptr) {
      cnode_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(attr_input));
    } else {
      auto new_value_node = CreateValueNodeKernelGraph(attr_input, graph);
      if (new_value_node != nullptr) {
        cnode_inputs.emplace_back(new_value_node);
      }
    }
  }
  return cnode_inputs;
}

void SessionBasic::CreateCNodeInputs(const CNodePtr &cnode, KernelGraph *graph, std::vector<AnfNodePtr> *cnode_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch)) {
    cnode_inputs->emplace_back(graph->GetBackendAnfByFrontAnf(cnode->input(kFirstDataInputIndex)));
    for (size_t index = kFirstBranchInSwitch; index < cnode->inputs().size(); index++) {
      auto node_input = cnode->input(index);
      auto switch_input = CreateSwitchInput(node_input, graph);
      cnode_inputs->emplace_back(switch_input);
    }
  } else {
    for (size_t input_idx = kFirstDataInputIndex; input_idx < cnode->inputs().size(); input_idx++) {
      auto anf = cnode->input(input_idx);
      MS_EXCEPTION_IF_NULL(anf);
      // anf has been created before
      if (graph->GetBackendAnfByFrontAnf(anf) != nullptr) {
        cnode_inputs->emplace_back(graph->GetBackendAnfByFrontAnf(anf));
        continue;
      } else if (IsValueNode<None>(anf)) {
        continue;
      }
      MS_LOG(EXCEPTION) << "Unexpected input[" << anf->DebugString() << "]";
    }
  }
}

CNodePtr SessionBasic::CreateNewCNode(CNodePtr cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs;
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  if (IsValueNode<FuncGraph>(attr_input)) {
    // cnode is a graph or a call
    cnode_inputs = CreateValueNode(cnode, graph);
  } else if (attr_input->isa<CNode>()) {
    // cnode ia a call (partial/switch/switch_layer)
    // 1. take the args of call to the partial node, as the real_args to call switch's or switch_layer's child graph
    // 2. the call in frontend is map to the partial/switch/switch_layer in backend and haven't been created
    cnode_inputs = CreateSwitchOrPartialNode(cnode, graph);
  } else {
    // get primitive of old node
    auto prim = AnfAlgo::GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    // push attr to inputs[0] of new cnode
    cnode_inputs = {graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(*prim)))};
  }
  // handle inputs of cnode except primitive
  CreateCNodeInputs(cnode, graph, &cnode_inputs);

  TraceManager::DebugTrace(std::make_shared<TraceCopy>(cnode->debug_info()));
  auto new_cnode = graph->NewCNode(cnode_inputs);
  TraceManager::EndTrace();

  // if the cnode is call switch, remove call
  if (new_cnode->inputs().size() > 1) {
    auto first_input = new_cnode->input(kFirstDataInputIndex);
    if (AnfAlgo::CheckPrimitiveType(new_cnode, prim::kPrimCall) &&
        AnfAlgo::CheckPrimitiveType(first_input, prim::kPrimSwitch)) {
      new_cnode = first_input->cast<CNodePtr>();
    }
  }

  return new_cnode;
}

ValueNodePtr SessionBasic::CreateValueNodeKernelGraph(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  auto value_node = anf->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto sub_func_graph = AnfAlgo::GetValueNodeFuncGraph(anf);
  MS_EXCEPTION_IF_NULL(sub_func_graph);
  if (front_backend_graph_map_.find(sub_func_graph) == front_backend_graph_map_.end()) {
    MS_LOG(EXCEPTION) << "FuncGraph: " << sub_func_graph->ToString() << " has not been transformed to KernelGraph.";
  }
  auto sub_kernel_graph = front_backend_graph_map_[sub_func_graph];

  ValueNodePtr new_value_node = std::make_shared<ValueNode>(sub_kernel_graph);
  new_value_node->set_abstract(value_node->abstract());
  // create new kernel_info of new value_node
  auto kernel_info = std::make_shared<device::KernelInfo>();
  kernel_info->SetFeatureMapFlag(false);
  new_value_node->set_kernel_info(kernel_info);
  // create kernel_build_info for new value node
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), new_value_node.get());
  AnfAlgo::SetGraphId(graph->graph_id(), new_value_node.get());

  graph->FrontBackendlMapAdd(anf, new_value_node);

  return new_value_node;
}

ParameterPtr SessionBasic::CreateNewParameter(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  if (!anf->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "Anf[" << anf->DebugString() << "] is not a parameter";
  }

  auto param_value = GetParamDefaultValue(anf);
  ParameterPtr new_parameter = nullptr;
  if (python_paras == nullptr) {
    python_paras = std::make_shared<std::map<ValuePtr, ParameterPtr>>();
  }
  auto iter = python_paras->find(param_value);
  if (iter != python_paras->end()) {
    new_parameter = iter->second;
  } else {
    TraceManager::DebugTrace(std::make_shared<TraceCopy>(anf->debug_info()));
    new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
    if (param_value != nullptr) {
      (*python_paras)[param_value] = new_parameter;
    }
    TraceManager::EndTrace();
  }
  new_parameter->IncreaseUsedGraphCount();

  return new_parameter;
}

KernelGraphPtr SessionBasic::ConstructKernelGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  std::unordered_map<AnfNodePtr, AnfNodePtr> other_graph_cnode;
  auto graph = NewKernelGraph();
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Create graph: " << graph->graph_id();
  for (const auto &node : lst) {
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Start create new cnode, node = " << node->DebugString();
    if (!node->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "Node " << node->DebugString() << " is not CNode";
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    // create a new cnode object
    auto new_cnode = CreateNewCNode(cnode, graph.get(), &other_graph_cnode);
    MS_EXCEPTION_IF_NULL(new_cnode);
    new_cnode->set_abstract(cnode->abstract());
    new_cnode->set_scope(cnode->scope());
    // record map relations between anf from ME and new anf node used in backend
    graph->FrontBackendlMapAdd(node, new_cnode);
  }
  // add a make_tuple at the end of graph as output
  graph->set_output(ConstructOutput(outputs, graph));
  MS_EXCEPTION_IF_NULL(context_);
  FuncGraphManagerPtr manager = MakeManager({graph});
  if (manager) {
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
  }
  graph->SetExecOrderByDefault();
  if (ExistSummaryNode(graph.get())) {
    graph->set_summary_node_exist(true);
  }

  // Update Graph Dynamic Shape Attr
  UpdateGraphDynamicShapeAttr(NOT_NULL(graph));

  opt::BackendCommonOptimization(graph);
  graph->SetInputNodes();
  graph->SetOptimizerFlag();
  return graph;
}

void SessionBasic::CreateCNodeKernelGraph(const AnfNodePtr node, KernelGraphPtr graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // create a new cnode object
  auto new_cnode = CreateNewCNode(cnode, graph.get());
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(cnode->abstract());
  std::string fullname;
  if (cnode->input(kAnfPrimitiveIndex)->isa<CNode>()) {
    fullname = cnode->input(kAnfPrimitiveIndex)->fullname_with_scope();
  } else {
    fullname = cnode->fullname_with_scope();
  }
  new_cnode->set_fullname_with_scope(fullname);
  new_cnode->set_scope(cnode->scope());
  graph->FrontBackendlMapAdd(node, new_cnode);
  if (AnfAlgo::CheckPrimitiveType(new_cnode, prim::kPrimReturn)) {
    graph->set_return(new_cnode);
  }
}

std::shared_ptr<KernelGraph> SessionBasic::ConstructKernelGraph(const FuncGraphPtr &func_graph,
                                                                std::vector<KernelGraphPtr> *all_out_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(all_out_graph);
  auto node_list = TopoSort(func_graph->get_return());
  auto graph = NewKernelGraph();
  MS_EXCEPTION_IF_NULL(graph);
  front_backend_graph_map_[func_graph] = graph;
  MS_LOG(INFO) << "Create graph: " << graph->graph_id();

  bool is_trace_back = false;
  for (const auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Start create new cnode, node = " << node->DebugString();
    if (node->isa<Parameter>()) {
      auto graph_inputs = graph->MutableInputs();
      MS_EXCEPTION_IF_NULL(graph_inputs);
      auto new_parameter = CreateNewParameter(node, graph.get());
      graph_inputs->push_back(new_parameter);
      graph->FrontBackendlMapAdd(node, new_parameter);
      continue;
    } else if (node->isa<ValueNode>()) {
      if (!IsValueNode<FuncGraph>(node)) {
        // if input is a common value node,
        (void)CreateNewValueNode(node, graph.get());
      } else {
        // if input is a ValueNode<FuncGraph>
        FuncGraphPtr child_graph = AnfAlgo::GetValueNodeFuncGraph(node);
        if (front_backend_graph_map_.find(child_graph) != front_backend_graph_map_.end()) {
          is_trace_back = true;
        } else {
          (void)ConstructKernelGraph(child_graph, all_out_graph);
        }
        (void)CreateValueNodeKernelGraph(node, graph.get());
      }
      continue;
    } else {
      CreateCNodeKernelGraph(node, graph);
    }
  }
  // if a graph jump back unconditionally, return op of this graph will never be executed, so output is null.
  graph->set_output_null(is_trace_back);
  AddParameterToGraphInputs(func_graph->parameters(), graph.get());
  graph->SetExecOrderByDefault();
  if (ExistSummaryNode(graph.get())) {
    graph->set_summary_node_exist(true);
  }
  all_out_graph->push_back(graph);
  return graph;
}

void SessionBasic::AddParameterToGraphInputs(const std::vector<AnfNodePtr> &parameters, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  graph_inputs->clear();
  for (auto &parameter : parameters) {
    MS_EXCEPTION_IF_NULL(parameter);
    auto backend_parameter = graph->GetBackendAnfByFrontAnf(parameter);
    if (backend_parameter == nullptr) {
      // for example "def f(x,y,z) {return x + y}", parameter z in unused
      auto new_parameter = CreateNewParameter(parameter, graph);
      graph_inputs->push_back(new_parameter);
      MS_LOG(INFO) << "Can't find parameter:" << parameter->DebugString();
      continue;
    }
    MS_LOG(INFO) << "Graph[" << graph->graph_id() << "],parameter:" << parameter->DebugString();
    graph_inputs->push_back(backend_parameter);
  }
}

namespace {
bool TensorNeedSync(const AnfNodePtr &parameter, const tensor::TensorPtr &tensor) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_address = AnfAlgo::GetMutableOutputAddr(parameter, 0);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
    return tensor->device_address().get() == nullptr || tensor->device_address() != device_address;
  }
  if (tensor->NeedSyncHostToDevice()) {
    return true;
  }
  auto tensor_address = tensor->device_address();
  if (tensor_address != device_address) {
    tensor->data_sync(false);
    return true;
  }
  return false;
}
}  // namespace

// run graph steps
void SessionBasic::LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                                 const std::vector<tensor::TensorPtr> &inputs_const) const {
  std::vector<tensor::TensorPtr> inputs(inputs_const);
  size_t input_ctrl_size = 3;
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (kernel_graph->input_ctrl_tensors()) {
    input_ctrl_size = LoadCtrlInputTensor(kernel_graph, &inputs);
  }
  auto &input_nodes = kernel_graph->input_nodes();

  if ((inputs.size() + input_ctrl_size) - 3 != input_nodes.size()) {
    MS_LOG(EXCEPTION) << "Tensor input:" << inputs.size() << " is not equal graph inputs:" << input_nodes.size()
                      << ", input_ctrl_size:" << input_ctrl_size;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto tensor = inputs[i];
    MS_EXCEPTION_IF_NULL(tensor);
    auto input_node = input_nodes[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_node->isa<Parameter>() && AnfAlgo::OutputAddrExist(input_node, 0) && TensorNeedSync(input_node, tensor)) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(input_node, 0);
      MS_EXCEPTION_IF_NULL(device_address);
      if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(input_node, 0),
                                            LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                            tensor->data_c())) {
        MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
      }
      if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode ||
          AnfAlgo::IsParameterWeight(input_node->cast<ParameterPtr>())) {
        tensor->set_device_address(device_address);
      }
    }
    tensor->set_sync_status(kNoNeedSync);
  }
}

void SessionBasic::UpdateOutputs(const std::shared_ptr<KernelGraph> &kernel_graph, VectorRef *const outputs,
                                 const std::vector<tensor::TensorPtr> &input_tensors) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  std::map<tensor::TensorPtr, session::KernelWithIndex> tensor_to_node;
  auto anf_outputs = kernel_graph->outputs();
  for (auto &item : anf_outputs) {
    MS_EXCEPTION_IF_NULL(item);
    MS_LOG(INFO) << "Update output[" << item->DebugString() << "]";
    outputs->emplace_back(CreateNodeOutputTensors(item, kernel_graph, input_tensors, &tensor_to_node));
  }

  for (auto &item : tensor_to_node) {
    auto &tensor = item.first;
    auto &node = item.second.first;
    auto &output_index = item.second.second;
    auto address = AnfAlgo::GetMutableOutputAddr(node, output_index);
    tensor->set_device_address(address);
    tensor->SetNeedWait(false);
  }
}

std::vector<tensor::TensorPtr> SessionBasic::GetInputNeedLockTensors(const GraphId &graph_id,
                                                                     const std::vector<tensor::TensorPtr> &inputs) {
  auto graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(graph);
  if (!graph->has_optimizer()) {
    return {};
  }
  std::vector<tensor::TensorPtr> result;
  for (auto &tensor : inputs) {
    if (!tensor->IsGraphOutput()) {
      result.emplace_back(tensor);
    }
  }
  return result;
}

void SessionBasic::CreateOutputTensors(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &input_tensors,
                                       VectorRef *outputs,
                                       std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node) {
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(tensor_to_node);
  auto anf_outputs = kernel_graph->outputs();
  for (auto &item : anf_outputs) {
    MS_EXCEPTION_IF_NULL(item);
    MS_LOG(INFO) << "Create node output[" << item->DebugString() << "]";
    outputs->emplace_back(CreateNodeOutputTensors(item, kernel_graph, input_tensors, tensor_to_node));
  }
}

void SessionBasic::RegisterSummaryCallBackFunc(const CallBackFunc &callback) {
  MS_EXCEPTION_IF_NULL(callback);
  summary_callback_ = callback;
}

void SessionBasic::Reorder(std::vector<CNodePtr> *node_list) { AnfAlgo::ReorderExecList(NOT_NULL(node_list)); }

void SessionBasic::RunInfer(NotNull<FuncGraphPtr> func_graph, const std::vector<tensor::TensorPtr> &inputs) {
  auto node_list = TopoSort(func_graph->get_return());
  size_t tensor_index = 0;
  for (const auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<CNode>()) {
      AbstractBasePtrList input_abstracts;
      for (size_t index = 0; index < AnfAlgo::GetInputTensorNum(node); ++index) {
        auto input_node = AnfAlgo::GetInputNode(node->cast<CNodePtr>(), index);
        MS_EXCEPTION_IF_NULL(input_node);
        auto abstract = input_node->abstract();
        MS_EXCEPTION_IF_NULL(abstract);
        input_abstracts.emplace_back(abstract);
      }
      auto prim = AnfAlgo::GetCNodePrimitive(node);
      if (prim->isa<PrimitiveC>()) {
        auto prim_c = prim->cast<std::shared_ptr<PrimitiveC>>();
        MS_EXCEPTION_IF_NULL(prim_c);
        auto abstract = prim_c->Infer(input_abstracts);
        node->set_abstract(abstract);
      } else {
        node->set_abstract(
          std::make_shared<tensor::Tensor>(kNumberTypeFloat32, std::vector<int64_t>{32, 64, 218, 218})->ToAbstract());
      }
    } else if (node->isa<Parameter>()) {
      if (tensor_index > inputs.size()) {
        MS_EXCEPTION(IndexError) << "Index " << tensor_index << "is out of " << inputs.size() << "tensor's size";
      }
      node->set_abstract(inputs[tensor_index++]->ToAbstract());
    } else {
      auto value_node = node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto value = value_node->value();
      MS_EXCEPTION_IF_NULL(value);
      value_node->set_abstract(value->ToAbstract());
    }
  }
}

void SessionBasic::SetSummaryNodes(KernelGraph *graph) {
  MS_LOG(DEBUG) << "Update summary Start";
  MS_EXCEPTION_IF_NULL(graph);
  if (!graph->summary_node_exist()) {
    return;
  }
  auto summary = graph->summary_nodes();
  auto apply_list = TopoSort(graph->get_return());
  for (auto &n : apply_list) {
    MS_EXCEPTION_IF_NULL(n);
    if (IsPrimitiveCNode(n, prim::kPrimScalarSummary) || IsPrimitiveCNode(n, prim::kPrimTensorSummary) ||
        IsPrimitiveCNode(n, prim::kPrimImageSummary) || IsPrimitiveCNode(n, prim::kPrimHistogramSummary)) {
      auto cnode = n->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (cnode->inputs().size() <= kSummaryGetItem) {
        MS_LOG(EXCEPTION) << "The node Summary should have 2 inputs at least!";
      }
      auto node = cnode->input(kSummaryGetItem);
      MS_EXCEPTION_IF_NULL(node);
      auto item_with_index = AnfAlgo::VisitKernelWithReturnType(node, 0, true);
      MS_EXCEPTION_IF_NULL(item_with_index.first);
      if (!AnfAlgo::IsRealKernel(item_with_index.first)) {
        MS_LOG(EXCEPTION) << "Unexpected node:" << item_with_index.first->DebugString();
      }
      summary[n->fullname_with_scope()] = item_with_index;
    }
  }
  graph->set_summary_nodes(summary);
  MS_LOG(DEBUG) << "Update summary end size: " << summary.size();
}

void SessionBasic::Summary(KernelGraph *graph) {
  if (summary_callback_ == nullptr) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  bool exist_summary = graph->summary_node_exist();
  if (!exist_summary) {
    return;
  }
  SetSummaryNodes(graph);
  auto summary_outputs = graph->summary_nodes();
  std::map<std::string, tensor::TensorPtr> params_list;
  // fetch outputs apply kernel in session & run callback functions
  for (auto &output_item : summary_outputs) {
    auto node = output_item.second.first;
    size_t index = IntToSize(output_item.second.second);
    auto address = AnfAlgo::GetOutputAddr(node, index);
    auto shape = AnfAlgo::GetOutputInferShape(node, index);
    TypeId type_id = AnfAlgo::GetOutputInferDataType(node, index);
    std::vector<int64_t> temp_shape;
    (void)std::copy(shape.begin(), shape.end(), std::back_inserter(temp_shape));
    tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, temp_shape);
    MS_EXCEPTION_IF_NULL(address);
    if (!address->GetPtr()) {
      continue;
    }
    if (!address->SyncDeviceToHost(trans::GetRuntimePaddingShape(node, index), LongToSize(tensor->data().nbytes()),
                                   tensor->data_type(), tensor->data_c())) {
      MS_LOG(ERROR) << "Failed to sync output from device to host.";
    }
    tensor->set_sync_status(kNoNeedSync);
    params_list[output_item.first] = tensor;
  }
  // call callback function here
  summary_callback_(0, params_list);
}

namespace {
bool CNodeFirstInputIsPrimitive(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return false;
  }
  auto prim = cnode->input(kAnfPrimitiveIndex);
  if (prim == nullptr || !IsValueNode<Primitive>(prim)) {
    return false;
  }
  return true;
}

std::vector<AnfNodePtr> ExtendNodeUsers(const FuncGraphManagerPtr &front_func_graph_manager,
                                        const AnfNodePtr &front_node) {
  auto node_users = front_func_graph_manager->node_users();
  auto users = node_users[front_node];
  std::vector<AnfNodePtr> result;
  for (auto user : users) {
    if (IsPrimitiveCNode(user.first, prim::kPrimControlDepend)) {
      continue;
    }
    if (IsPrimitiveCNode(user.first, prim::kPrimDepend)) {
      auto res = ExtendNodeUsers(front_func_graph_manager, user.first);
      result.insert(result.end(), res.begin(), res.end());
      continue;
    }
    result.emplace_back(user.first);
  }
  return result;
}

void HandleInternalOutput(const AnfNodePtr &front_node, const AnfNodePtr &backend_node,
                          const FuncGraphManagerPtr &front_func_graph_manager,
                          const std::shared_ptr<KernelGraph> &backend_graph) {
  // When init parameter from cnode of other graphs, the cnode will not be real kernel except for tuple_getitem.
  if (!AnfAlgo::IsRealKernel(front_node) && !AnfAlgo::CheckPrimitiveType(front_node, prim::kPrimTupleGetItem)) {
    return;
  }
  auto front_real_kernel_pair = AnfAlgo::VisitKernel(front_node, 0);
  auto backend_real_kernel_pair = AnfAlgo::VisitKernel(backend_node, 0);

  auto front_real_kernel = front_real_kernel_pair.first;
  std::string kernel_target = GetCNodeTarget(front_real_kernel);
  bool internal_output = CNodeFirstInputIsPrimitive(front_real_kernel);
  bool unique_target = true;
  if (internal_output && opt::IsNopNode(front_real_kernel)) {
    auto pre_node_pair = AnfAlgo::GetPrevNodeOutput(front_real_kernel, 0);
    auto pre_node_target = GetCNodeTarget(pre_node_pair.first);
    if (pre_node_target != kernel_target) {
      unique_target = false;
    }
  }
  if (internal_output) {
    auto users = ExtendNodeUsers(front_func_graph_manager, front_node);
    for (auto user : users) {
      if (!CNodeFirstInputIsPrimitive(user)) {
        internal_output = false;
        break;
      }
      if (!AnfAlgo::IsRealKernel(user)) {
        internal_output = false;
        break;
      }
      if (kernel_target != GetCNodeTarget(user)) {
        unique_target = false;
      }
    }
  }
  if (internal_output) {
    MS_LOG(INFO) << "Internal output: " << front_node->DebugString() << " To "
                 << backend_real_kernel_pair.first->DebugString() << ", unique_target: " << unique_target;
    backend_graph->AddInternalOutput(front_node, backend_real_kernel_pair.first, backend_real_kernel_pair.second,
                                     unique_target);
  }
}
}  // namespace

CNodePtr SessionBasic::ConstructOutput(const AnfNodePtrList &outputs, const std::shared_ptr<KernelGraph> &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> output_args;
  for (const auto &output : outputs) {
    MS_EXCEPTION_IF_NULL(output);
    MS_LOG(INFO) << "Output:" << output->DebugString();
  }
  auto FindEqu = [graph, outputs](const AnfNodePtr &out) -> AnfNodePtr {
    auto backend_anf = graph->GetBackendAnfByFrontAnf(out);
    if (backend_anf != nullptr) {
      auto context_ptr = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context_ptr);
      if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
        return backend_anf;
      }

      MS_EXCEPTION_IF_NULL(out);
      auto out_func_graph = out->func_graph();
      MS_EXCEPTION_IF_NULL(out_func_graph);
      auto out_func_graph_manager = out_func_graph->manager();
      if (out_func_graph_manager == nullptr) {
        return backend_anf;
      }
      HandleInternalOutput(out, backend_anf, out_func_graph_manager, graph);
      return backend_anf;
    }
    MS_LOG(EXCEPTION) << "Can't find the node in the equiv map!";
  };
  output_args.push_back(NewValueNode(prim::kPrimMakeTuple));
  (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_args),
                       [&](const AnfNodePtr &out) -> AnfNodePtr { return FindEqu(out); });
  return graph->NewCNode(output_args);
}

void SessionBasic::CreateOutputNode(const CNodePtr &cnode, const std::shared_ptr<KernelGraph> &graph) {
  MS_LOG(INFO) << "Start!";
  std::vector<AnfNodePtr> make_tuple_inputs;
  make_tuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
  MS_EXCEPTION_IF_NULL(graph);
  if (AnfRuntimeAlgorithm::GetOutputTensorNum(cnode) > 1) {
    for (size_t output_index = 0; output_index < AnfRuntimeAlgorithm::GetOutputTensorNum(cnode); output_index++) {
      auto idx = NewValueNode(SizeToLong(output_index));
      MS_EXCEPTION_IF_NULL(idx);
      auto imm = std::make_shared<Int64Imm>(output_index);
      idx->set_abstract(std::make_shared<abstract::AbstractScalar>(imm));
      auto getitem = graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), cnode, idx});
      std::vector<TypeId> types = {AnfAlgo::GetOutputInferDataType(cnode, output_index)};
      std::vector<std::vector<size_t>> shapes = {AnfAlgo::GetOutputInferShape(cnode, output_index)};
      AnfAlgo::SetOutputInferTypeAndShape(types, shapes, getitem.get());
      make_tuple_inputs.push_back(getitem);
    }
  } else {
    make_tuple_inputs.push_back(cnode);
  }
  // create output
  auto g_output = graph->NewCNode(make_tuple_inputs);
  graph->set_output(g_output);
  MS_LOG(INFO) << "Finish!";
}

std::shared_ptr<KernelGraph> SessionBasic::ConstructSingleOpGraph(const OpRunInfo &op_run_info,
                                                                  const std::vector<tensor::TensorPtr> &input_tensors,
                                                                  const std::vector<int64_t> &tensors_mask) {
  auto graph = std::make_shared<KernelGraph>();
  graph->set_graph_id(graph_sum_);
  graph_sum_++;
  std::vector<AnfNodePtr> inputs;
  // set input[0]
  PrimitivePtr op_prim = op_run_info.primitive;
  MS_EXCEPTION_IF_NULL(op_prim);
  inputs.push_back(std::make_shared<ValueNode>(op_prim));
  // set input parameter
  MS_LOG(INFO) << "Input tensor size: " << input_tensors.size();
  if (input_tensors.size() != tensors_mask.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size() << " should be equal to tensors mask size "
                      << tensors_mask.size();
  }
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    if (tensors_mask[i] == kValueNodeTensorMask) {
      auto value_node = ConstructRunOpValueNode(graph, input_tensors[i]);
      inputs.push_back(value_node);
      continue;
    }
    auto parameter = ConstructRunOpParameter(graph, input_tensors[i], tensors_mask[i]);
    inputs.push_back(parameter);
    auto mutable_inputs = graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(mutable_inputs);
    mutable_inputs->push_back(parameter);
  }
  // set execution order
  auto cnode = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode);
  // set abstract,which include inferred shapes and types
  cnode->set_abstract(op_run_info.abstract);
  // get output dynamic shape info
  AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(op_run_info.is_dynamic_shape), cnode);
  // set execution order
  std::vector<CNodePtr> exe_order = {cnode};
  graph->set_execution_order(exe_order);
  graph->UpdateGraphDynamicAttr();
  // set output
  CreateOutputNode(cnode, graph);
  graph->SetInputNodes();
  return graph;
}

KernelGraphPtr SessionBasic::NewKernelGraph() {
  auto graph = std::make_shared<KernelGraph>();
  graph->set_graph_id(graph_sum_);
  graphs_[graph_sum_++] = graph;
  return graph;
}

AnfNodePtr SessionBasic::FindPullNode(const AnfNodePtr &push_node, const std::vector<AnfNodePtr> &node_list) {
  MS_EXCEPTION_IF_NULL(push_node);
  for (auto &node : node_list) {
    if (node != nullptr && node->isa<CNode>()) {
      for (auto input : node->cast<CNodePtr>()->inputs()) {
        if (push_node == AnfAlgo::VisitKernel(input, 0).first) {
          if (AnfAlgo::GetCNodeName(node) != kPullOpName) {
            MS_LOG(EXCEPTION) << "The edge between Push and Pull node is invalid.";
          }
          return node;
        }
      }
    }
  }
  return nullptr;
}

GraphId SessionBasic::CompileGraph(const GraphSegmentPtr &segment, const AnfNodePtrList &outputs) {
  MS_EXCEPTION_IF_NULL(executor_);
  return executor_->CompileGraph(shared_from_this(), segment, outputs);
}

GraphId SessionBasic::CompileGraph(NotNull<FuncGraphPtr> func_graph) {
  MS_EXCEPTION_IF_NULL(executor_);
  return executor_->CompileGraph(shared_from_this(), func_graph);
}

void SessionBasic::BuildGraph(GraphId graph_id) {
  MS_EXCEPTION_IF_NULL(executor_);
  executor_->BuildGraph(shared_from_this(), graph_id);
}

void SessionBasic::BuildOp(OpRunInfo *op_run_info, const GraphInfo &graph_info,
                           const std::vector<tensor::TensorPtr> &input_tensors,
                           const std::vector<int64_t> &tensors_mask) {
  MS_EXCEPTION_IF_NULL(executor_);
  executor_->BuildOp(shared_from_this(), op_run_info, graph_info, input_tensors, tensors_mask);
}

void SessionBasic::RunOp(OpRunInfo *op_run_info, const GraphInfo &graph_info,
                         const std::vector<tensor::TensorPtr> &input_tensors, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(executor_);
  executor_->RunOp(shared_from_this(), op_run_info, graph_info, input_tensors, outputs);
}

void SessionBasic::RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(executor_);
  executor_->RunGraph(shared_from_this(), graph_id, inputs, outputs);
}

void SessionBasic::RunGraphAsync(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                                 VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(executor_);
  executor_->RunGraphAsync(shared_from_this(), graph_id, inputs, outputs);
}

bool IsDynamicShape(const NotNull<abstract::ShapePtr> &shape) {
  return std::any_of(shape->shape().begin(), shape->shape().end(), [](int64_t s) { return s < 0; });
}

bool IsNodeOutputDynamicShape(const CNodePtr &anf_node_ptr) {
  MS_EXCEPTION_IF_NULL(anf_node_ptr);
  return AnfAlgo::IsNodeDynamicShape(anf_node_ptr);
}

bool IsNodeInputDynamicShape(const CNodePtr &anf_node_ptr) {
  MS_EXCEPTION_IF_NULL(anf_node_ptr);
  auto input_num = AnfAlgo::GetInputTensorNum(anf_node_ptr);
  for (size_t i = 0; i < input_num; ++i) {
    auto input_with_index = AnfAlgo::GetPrevNodeOutput(anf_node_ptr, i);
    auto input = input_with_index.first;
    auto index = input_with_index.second;
    MS_EXCEPTION_IF_NULL(input);

    auto base_shape = input->Shape();
    if (base_shape == nullptr) {
      MS_LOG(INFO) << "Invalid shape ptr, node:" << input->fullname_with_scope();
      continue;
    }
    if (base_shape->isa<abstract::Shape>()) {
      if (IsDynamicShape(NOT_NULL(base_shape->cast<abstract::ShapePtr>()))) {
        return true;
      }
    } else if (base_shape->isa<abstract::TupleShape>()) {
      auto tuple_shape = base_shape->cast<abstract::TupleShapePtr>();
      MS_EXCEPTION_IF_NULL(tuple_shape);

      if (index >= tuple_shape->size()) {
        MS_LOG(INFO) << "Node:" << anf_node_ptr->fullname_with_scope() << "Invalid index:" << index
                     << " and tuple_shape size:" << tuple_shape->size();
        continue;
      }

      auto b_shp = (*tuple_shape)[index];
      if (!b_shp->isa<abstract::Shape>()) {
        continue;
      }
      if (IsDynamicShape(NOT_NULL(b_shp->cast<abstract::ShapePtr>()))) {
        return true;
      }
    }
  }
  return false;
}

void SessionBasic::UpdateGraphDynamicShapeAttr(const NotNull<KernelGraphPtr> &root_graph) {
  for (const auto &cnode : root_graph->execution_order()) {
    auto output_dynamic = IsNodeOutputDynamicShape(NOT_NULL(cnode));
    auto input_dynamic = IsNodeInputDynamicShape(NOT_NULL(cnode));
    if (output_dynamic || input_dynamic) {
      AnfAlgo::SetNodeAttr(kAttrIsDynamicShape, MakeValue(true), cnode);
      MS_LOG(INFO) << "Set Dynamic Shape Attr to Node:" << cnode->fullname_with_scope();
    }
    if (output_dynamic) {
      AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), cnode);
      MS_LOG(INFO) << "Set Output Dynamic Shape Attr to Node:" << cnode->fullname_with_scope();
    }
    if (input_dynamic) {
      AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), cnode);
      MS_LOG(INFO) << "Set Input Dynamic Shape Attr to Node:" << cnode->fullname_with_scope();
    }
  }
  root_graph->UpdateGraphDynamicAttr();
}

#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
void SessionBasic::CheckPSModeConsistence(const KernelGraphPtr &kernel_graph) {
  auto input_nodes = kernel_graph->inputs();
  for (const auto &input_node : input_nodes) {
    if (!input_node->isa<Parameter>()) {
      continue;
    }
    auto pk_node = input_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(pk_node);
    auto param_info_ptr = pk_node->param_info();
    if (param_info_ptr != nullptr && param_info_ptr->init_in_server()) {
      const std::string &param_name = pk_node->fullname_with_scope();
      MS_LOG(EXCEPTION) << "Can not initialize the parameter[" << param_name
                        << "] in server, this parameter is used by kernel which executes in device";
    }
  }
}

void SessionBasic::AssignParamKey(const KernelGraphPtr &kernel_graph) {
  if (!ps::Util::IsRoleOfWorker()) {
    MS_LOG(INFO) << "Not parameter server mode.";
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(kernel_graph->get_return());
  for (auto &node : node_list) {
    if (node != nullptr && node->isa<CNode>()) {
      // Assign key for forward kernel EmbeddingLookup.
      // The key will be assigned to embedding table ande Push kernel as well.
      if (AnfAlgo::GetCNodeName(node) == kEmbeddingLookupOpName) {
        size_t embedding_table_idx = 0;
        auto embedding_table = AnfAlgo::GetInputNode(node->cast<CNodePtr>(), embedding_table_idx);
        size_t key = ps::worker.SetParamKey(embedding_table->fullname_with_scope());
        AnfAlgo::SetNodeAttr(kAttrPsKey, MakeValue(key), node);
      } else if (AnfAlgo::GetCNodeName(node) == kPushOpName) {
        auto pull_node = FindPullNode(node, node_list);
        if (!pull_node) {
          MS_LOG(EXCEPTION) << "Assigning parameter key failed: can't find Pull node of the Push node.";
        }

        // Second input of Pull node is the trainable parameter.
        size_t parameter_index = 1;
        auto parameter_node = AnfAlgo::GetInputNode(pull_node->cast<CNodePtr>(), parameter_index);
        size_t key = ps::worker.SetParamKey(parameter_node->fullname_with_scope());
        AnfAlgo::SetNodeAttr(kAttrPsKey, MakeValue(key), node);
        AnfAlgo::SetNodeAttr(kAttrPsKey, MakeValue(key), pull_node);

        std::string optimizer_name = AnfAlgo::GetNodeAttr<std::string>(node, kAttrOptimizerType);
        ps::worker.SetKeyOptimId(key, optimizer_name);
      }
    }
  }
}

void SessionBasic::InitPSParamAndOptim(const KernelGraphPtr &kernel_graph,
                                       const std::vector<tensor::TensorPtr> &inputs_const) {
  if (!ps::Util::IsRoleOfWorker()) {
    return;
  }
  std::vector<tensor::TensorPtr> inputs(inputs_const);
  size_t input_ctrl_size = 1;
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (kernel_graph->input_ctrl_tensors()) {
    input_ctrl_size = LoadCtrlInputTensor(kernel_graph, &inputs);
  }
  auto input_nodes = kernel_graph->inputs();
  if ((inputs.size() + input_ctrl_size) - 1 != input_nodes.size()) {
    MS_LOG(EXCEPTION) << "Tensor input:" << inputs.size() << " is not equal graph inputs:" << input_nodes.size()
                      << ", input_ctrl_size:" << input_ctrl_size;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto tensor = inputs[i];
    MS_EXCEPTION_IF_NULL(tensor);
    auto input_node = input_nodes[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_node->isa<Parameter>() && AnfAlgo::OutputAddrExist(input_node, 0)) {
      ps::worker.InitPSParamAndOptim(input_node, tensor);
    }
  }
}
#endif
}  // namespace session
}  // namespace mindspore
