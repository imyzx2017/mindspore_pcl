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

#include <dirent.h>
#include <stdio.h>
#include <fstream>
#include <tuple>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <utility>
#include <map>
#include <regex>
#include "debug/debugger/debugger.h"
#include "debug/data_dump/dump_json_parser.h"
#include "pipeline/jit/pipeline.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/device/kernel_runtime.h"
#include "debug/data_dump/e2e_dump_util.h"
#include "utils/config_manager.h"

using debugger::Chunk;
using debugger::EventReply;
using debugger::GraphProto;
using debugger::ModelProto;
using debugger::TensorProto;
using debugger::WatchCondition;
using debugger::WatchCondition_Condition_inf;
using debugger::WatchCondition_Condition_nan;
using debugger::WatchCondition_Parameter;
using debugger::WatchNode;
using debugger::WatchpointHit;

#define CHUNK_SIZE 1024 * 1024 * 3

namespace mindspore {

DebuggerPtr Debugger::debugger_ = nullptr;
std::mutex Debugger::instance_lock_;
static const size_t PARAMETER_OUTPUT_INDEX = 0;
static const size_t VALUE_NODE_OUTPUT_INDEX = 0;

Debugger::Debugger()
    : grpc_client_(nullptr),
      debug_services_(nullptr),
      device_id_(0),
      device_target_(""),
      num_step_(0),
      debugger_enabled_(false),
      run_level_(""),
      node_name_(""),
      cur_name_(""),
      training_done_(false),
      is_dataset_graph_(false),
      partial_memory_(false),
      last_overflow_bin_(0),
      overflow_bin_path_(""),
      initial_suspend_(true),
      not_dataset_graph_sum_(0) {
  if (CheckDebuggerEnabled()) {
    // configure partial memory reuse
    partial_memory_ = CheckDebuggerPartialMemoryEnabled();

    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    // switch memory reuse on or off
    context_ptr->set_param<bool>(MS_CTX_ENABLE_MEM_REUSE, partial_memory_);
    // print some message about memory reuse to user
    if (partial_memory_) {
      MS_LOG(WARNING)
        << "Partial Memory Reuse is enabled. Note: 1. Please only set watchpoints before running the first "
           "step. 2. Tensor values are only available for nodes that are watched by any watchpoint.";
    } else {
      MS_LOG(INFO) << "Memory Reuse is disabled. Set environment variable MS_DEBUGGER_PARTIAL_MEM=1 to reduce memory "
                      "usage for large models.";
    }
  }
}

void Debugger::Init(const uint32_t device_id, const std::string device_target) {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  // save device_id
  MS_LOG(INFO) << "Debugger got device_id: " << device_id;
  device_id_ = device_id;
  MS_LOG(INFO) << "Debugger got device_target: " << device_target;
  device_target_ = device_target;
}

void Debugger::EnableDebugger() {
  // reset some of the class members
  num_step_ = 0;
  debugger_enabled_ = false;
  partial_memory_ = false;
  grpc_client_ = nullptr;
  debug_services_ = nullptr;

  // see if dump using debugger backend is enabled
  bool dump_enabled = CheckDebuggerDumpEnabled();
  MS_LOG(INFO) << "dump using debugger backend = " << dump_enabled;

  // check if debugger enabled
  debugger_enabled_ = CheckDebuggerEnabled();
  MS_LOG(INFO) << "debugger_enabled_ = " << debugger_enabled_;

  if (!debugger_enabled_ && !dump_enabled) {
    MS_LOG(INFO) << "Not enabling debugger. Set environment variable ENABLE_MS_DEBUGGER=1 to enable debugger.";
    return;
  }

  // configure grpc host
  const char *env_host_str = std::getenv("MS_DEBUGGER_HOST");
  std::string host;
  if (env_host_str != nullptr) {
    std::regex reg_ip(
      "(25[0-4]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[1-9])"
      "[.](25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])"
      "[.](25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])"
      "[.](25[0-4]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[1-9])");
    std::smatch smat;
    std::string host_str = std::string(env_host_str);
    if (std::regex_match(host_str, smat, reg_ip)) {
      MS_LOG(INFO) << "Getenv MS_DEBUGGER_HOST: " << env_host_str;
      host = std::string(env_host_str);
    } else {
      MS_LOG(ERROR) << "Environment variable MS_DEBUGGER_HOST isn't a valid IP address. "
                       "Please set environment variable MS_DEBUGGER_HOST=x.x.x.x to a valid IP";
      debugger_enabled_ = false;
    }
  } else {
    MS_LOG(INFO) << "Environment variable MS_DEBUGGER_HOST doesn't exist. Using default debugger host: localhost";
    host = "localhost";
  }
  // configure grpc port
  const char *env_port_str = std::getenv("MS_DEBUGGER_PORT");
  std::string port;
  if (env_port_str != nullptr) {
    if (CheckPort(env_port_str)) {
      MS_LOG(INFO) << "Getenv MS_DEBUGGER_PORT: " << env_port_str;
      port = std::string(env_port_str);
    } else {
      MS_LOG(ERROR) << "Environment variable MS_DEBUGGER_PORT is not valid. Custom port ranging from 1 to 65535";
      debugger_enabled_ = false;
    }
  } else {
    MS_LOG(INFO) << "Environment variable MS_DEBUGGER_PORT doesn't exist. Using default debugger port: 50051";
    port = "50051";
  }
#ifdef ENABLE_D
  // set operation overflow info
  overflow_bin_path_ = DumpJsonParser::GetInstance().GetOpOverflowBinPath(graph_ptr_->graph_id(), device_id_);
  // new overflow dump files will have a timestamp greater than last_overflow_bin_
  last_overflow_bin_ = 0;
  DIR *d;
  d = opendir(overflow_bin_path_.c_str());
  if (d != nullptr) {
    struct dirent *dir;
    while ((dir = readdir(d)) != NULL) {
      if (dir->d_type == DT_REG) {
        std::string file_path = overflow_bin_path_;
        file_path.append(dir->d_name);
        std::size_t found = file_path.find_last_of(".");
        if (found == std::string::npos) {
          continue;
        }
        std::string overflow_time = file_path.substr(found + 1);
        if (stod(overflow_time) <= last_overflow_bin_) {
          MS_LOG(INFO) << "Old op overflow bin folder" << file_path;
          continue;
        }
        last_overflow_bin_ = stod(overflow_time);
      }
    }
    MS_LOG(INFO) << "last op overflow bin folder" << last_overflow_bin_;
    closedir(d);
  }
#endif

  // initialize grpc client
  if (debugger_enabled_) {
    grpc_client_ = std::make_unique<GrpcClient>(host, port);
  }

  debug_services_ = std::make_unique<DebugServices>();
}

void Debugger::CheckDatasetSinkMode() {
  if (CheckDebuggerDumpEnabled() && ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE) {
    MS_EXCEPTION(NotSupportError)
      << "e2e_dump not supported on GPU with dataset_sink_mode=True. Please set dataset_sink_mode=False";
  }

  if (CheckDebuggerEnabled() && ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE) {
    MS_EXCEPTION(NotSupportError)
      << "Debugger is not supported with dataset_sink_mode=True. Please set dataset_sink_mode=False";
  }
}

bool Debugger::CheckDebuggerDumpEnabled() {
  // see if dump is enabled
  if (device_target_ == kGPUDevice) {
    return device::KernelRuntime::DumpDataEnabled();
  }
  return false;
}

bool Debugger::CheckDebuggerEnabled() {
  // get env variables to configure debugger
  const char *env_enable_str = std::getenv("ENABLE_MS_DEBUGGER");
  if (env_enable_str != nullptr) {
    if (std::strcmp(env_enable_str, "1") == 0) {
      return true;
    }
  }
  return false;
}

bool Debugger::CheckDebuggerPartialMemoryEnabled() {
  const char *env_partial_mem_str = std::getenv("MS_DEBUGGER_PARTIAL_MEM");
  if (env_partial_mem_str != nullptr) {
    MS_LOG(INFO) << "Getenv MS_DEBUGGER_PARTIAL_MEM: " << env_partial_mem_str;
    if (std::strcmp(env_partial_mem_str, "1") == 0) {
      return true;
    }
  }
  return false;
}

bool Debugger::DebuggerBackendEnabled() { return CheckDebuggerDumpEnabled() || CheckDebuggerEnabled(); }

void Debugger::Reset() {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  // reset components
  device_id_ = 0;
  device_target_ = "";
  num_step_ = 0;
  debugger_enabled_ = false;
  is_dataset_graph_ = false;
  partial_memory_ = false;
  graph_ptr_ = nullptr;
  grpc_client_ = nullptr;
  debug_services_ = nullptr;
  last_overflow_bin_ = 0;
  overflow_bin_path_ = "";
  stream_task_to_opname_.clear();
}

void Debugger::PreExecute(const KernelGraphPtr &graph_ptr, uint32_t graph_sum) {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  CheckDatasetSinkMode();
  auto graph_id = graph_ptr->graph_id();
  // collect rungrap_ids to update step number in multigraph case
  if (!rungraph_id_list_.size()) {
    rungraph_id_list_.push_back(graph_id);

  } else {
    if (std::find(rungraph_id_list_.begin(), rungraph_id_list_.end(), graph_id) == rungraph_id_list_.end()) {
      rungraph_id_list_.push_back(graph_id);
    }
  }
  // check and save graph_ptr, suspend if graph is new
  MS_LOG(INFO) << "total number graph: " << graph_sum;
  // multiple graphs
  if (graph_sum > 1) {
    // there are more than one graphs are not dataset_graph
    if (not_dataset_graph_sum_ > 0) {
      // only try to enable debugger if they are not all dataset graphs
      if (!debugger_enabled_) {
        EnableDebugger();
      }

      if (debugger_enabled_) {
        if (graph_proto_list_.size()) {
          // only send compiled graphs once.
          SendMultiGraphsAndSuspend(graph_proto_list_, graph_sum);
          graph_proto_list_.clear();
        } else if (graph_id == rungraph_id_list_.front() && device_target_ == kGPUDevice) {
          // stop only when receive the first sub run graph for each step
          CommandLoop();
        }
      }
    }
  } else if (graph_proto_list_.size() == 1) {
    // In single graph case, reset graph_ptr_ to be nullptr for the initial step
    if (num_step_ == 0) {
      graph_ptr_ = nullptr;
    }
    CheckGraphPtr(graph_ptr);
  }
}

void Debugger::PostExecute() {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  if (pipeline::ExecutorPy::GetDebugTerminate()) {
    return;
  }
  if (debugger_->DebuggerBackendEnabled()) {
    // analyze tensor data and send the watchpoints been hit
    if (run_level_ == "node") {
      MS_LOG(INFO) << "Debugger is in node level mode ";
      return;
    }
    if (debugger_enabled_ && !is_dataset_graph_) {
      if (device_target_ != kGPUDevice) {
        num_step_++;
        MS_LOG(INFO) << "Debugger suspend at end of step; number of steps executed: " << num_step_;
        SendWatchpoints(CheckWatchpoints());
        CommandLoop();
      } else {
        CommandLoop();
      }
    }
  }
}

bool Debugger::ReadNodeDataRequired(const CNodePtr &kernel) {
  if (debugger_enabled_ && !is_dataset_graph_) {
    auto is_watchpoint = debug_services_->IsWatchPoint(cur_name_, kernel);
    // if node has a watchpoint on it, is next_to node, or continue_to node then read the kernel tensor data
    if (is_watchpoint || (run_level_ == "node" && (node_name_ == "" || node_name_ == cur_name_))) {
      return true;
    }
  }
  return false;
}

void Debugger::PostExecuteNode(const CNodePtr &kernel) {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  if (pipeline::ExecutorPy::GetDebugTerminate()) {
    return;
  }
  if (debugger_enabled_ && !is_dataset_graph_) {
    auto is_watchpoint = debug_services_->IsWatchPoint(cur_name_, kernel);

    // if kernel is watchpoint,and get hit. suspend.
    bool hit_empty_flag = true;
    if (is_watchpoint) {
      auto hits = CheckWatchpoints(cur_name_, kernel);
      if (!hits.empty()) {
        SendWatchpoints(hits);
        CommandLoop();
        hit_empty_flag = false;
      }
    }
    if (hit_empty_flag && run_level_ == "node" && (node_name_ == "" || node_name_ == cur_name_)) {
      // if kernel is not watchpoint and is next_to or continue_to node, suspend
      CommandLoop();
    }
    return;
  }
}

void Debugger::PostDebugOp() {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  // suspend if debugger is enabled
  if (debugger_enabled_ && !is_dataset_graph_) {
    MS_LOG(INFO) << "Debugger suspend at debug_op";
    CommandLoop();
  }
}

void Debugger::SetStreamTaskToOpnameMap(const std::map<std::pair<uint32_t, uint32_t>, std::string> &mapping) {
  stream_task_to_opname_ = mapping;
}

void Debugger::LoadGraphs(const KernelGraphPtr &graph_ptr) {
  if (graph_ptr_ != graph_ptr) {
    MS_LOG(INFO) << "LoadGraphs Debugger got new graph: " << graph_ptr->graph_id();
    // save new graph_ptr
    graph_ptr_ = graph_ptr;
    CheckDatasetGraph();
    if (!is_dataset_graph_) {
      // get proto for new graph_ptr
      auto graph_proto = GetGraphProto(graph_ptr);
      // add new graph proto to graph_proto_list_
      graph_proto_list_.push_back(graph_proto);
      graph_ptr_list_.push_back(graph_ptr);
      not_dataset_graph_sum_++;
    }
    // reset is_dataset_graph to be false
    is_dataset_graph_ = false;
  }
}

// In single graph cases, check single graph ptr
void Debugger::CheckGraphPtr(const KernelGraphPtr &graph_ptr) {
  if (graph_ptr_ != graph_ptr) {
    MS_LOG(INFO) << "CheckGraphPtr Debugger got new graph: " << graph_ptr->graph_id();
    // save new graph_ptr
    graph_ptr_ = graph_ptr;
    if (!is_dataset_graph_) {
      // only try to enable debugger if it is not a dataset graph
      EnableDebugger();
      if (debugger_enabled_) {
        LoadParametersAndConst();
        // get graph proto and send to mindinsight
        auto graph_proto = graph_proto_list_.front();
        SendGraphAndSuspend(graph_proto);
      }
    }
  }
}

void Debugger::CheckDatasetGraph() {
  // print parameter node names
  const auto &params = graph_ptr_->inputs();
  for (const auto &param : params) {
    MS_LOG(INFO) << "param: " << param->fullname_with_scope();
  }
  // check if there is GetNext or InitDataSetQueue node
  const auto &nodes = graph_ptr_->execution_order();
  for (const auto &node : nodes) {
    auto node_name = AnfAlgo::GetCNodeName(node);
    MS_LOG(INFO) << "node: " << node->fullname_with_scope();
    if (node_name == "GetNext" || node_name == "InitDataSetQueue") {
      MS_LOG(INFO) << "Not enabling debugger for graph " << graph_ptr_->graph_id() << ": found dataset graph node "
                   << node_name;
      is_dataset_graph_ = true;
      return;
    }
  }
  is_dataset_graph_ = false;
}

GraphProto Debugger::GetGraphProto(const KernelGraphPtr &graph_ptr) const {
  // convert kernel graph to debugger modelproto
  ModelProto model = GetDebuggerFuncGraphProto(graph_ptr_);
  return model.graph();
}

void Debugger::SendGraphAndSuspend(const GraphProto &graph_proto) {
  SendMetadata();
  // send graph to mindinght server
  EventReply reply = grpc_client_->SendGraph(graph_proto);
  if (reply.status() != reply.OK) {
    MS_LOG(ERROR) << "Error: SendGraph failed";
  }
  // enter command loop, wait and process commands
  CommandLoop();
}

void Debugger::SendMetadata() {
  // prepare metadata
  std::string device_name = std::to_string(device_id_) + ":" + std::to_string(graph_ptr_->graph_id());
  Metadata metadata;
  metadata.set_device_name(device_name);
  metadata.set_cur_step(num_step_);
  metadata.set_backend(device_target_);
  metadata.set_cur_node(cur_name_);
  metadata.set_training_done(training_done_);
  MS_LOG(INFO) << "Is training done?" << training_done_;
  // set graph munber to not_dataset_graph_sum_
  metadata.set_graph_num(not_dataset_graph_sum_);
  EventReply reply_metadata = grpc_client_->SendMetadata(metadata);
  if (reply_metadata.status() != reply_metadata.OK) {
    MS_LOG(ERROR) << "Error: SendMetadata failed";
  }
}

void Debugger::SendMultiGraphsAndSuspend(const std::list<GraphProto> &graph_proto_list, uint32_t graph_sum) {
  SendMetadata();
  // send multiple graphs to mindinght server
  // split graph into chunks if one graph is larger than chunk size
  std::list<Chunk> chunked_graph_proto_list;
  Chunk chunk;
  for (auto graph : graph_proto_list) {
    std::string str = graph.SerializeAsString();
    auto graph_size = graph.ByteSize();
    if (graph_size > CHUNK_SIZE) {
      auto sub_graph_str = grpc_client_->ChunkString(str, graph_size);
      for (unsigned int i = 0; i < sub_graph_str.size(); i++) {
        chunk.set_buffer(sub_graph_str[i]);
        chunked_graph_proto_list.push_back(chunk);
        if (i < sub_graph_str.size() - 1) {
          chunk.set_finished(false);
        } else {
          chunk.set_finished(true);
          chunked_graph_proto_list.push_back(chunk);
        }
      }
    } else {
      chunk.set_buffer(str);
      chunk.set_finished(true);
      chunked_graph_proto_list.push_back(chunk);
    }
  }
  EventReply reply = grpc_client_->SendMultiGraphs(chunked_graph_proto_list);
  if (reply.status() != reply.OK) {
    MS_LOG(ERROR) << "Error: SendGraph failed";
  }
  // enter command loop, wait and process commands
  CommandLoop();
}

void Debugger::CommandLoop() {
  // prepare metadata
  std::string device_name = std::to_string(device_id_) + ":" + std::to_string(graph_ptr_->graph_id());
  Metadata metadata;

  metadata.set_device_name(device_name);
  metadata.set_cur_step(num_step_);
  metadata.set_backend(device_target_);
  metadata.set_cur_node(cur_name_);
  metadata.set_training_done(training_done_);

  // loop exit flag
  bool run = false;
  int num_wait_fail = 0;
  const int max_num_wait_fail = 5;

  while (!run) {
    // wait for command
    EventReply reply = grpc_client_->WaitForCommand(metadata);
    if (reply.status() != reply.OK) {
      MS_LOG(ERROR) << "Error: WaitForCommand failed";
      num_wait_fail++;
      if (num_wait_fail > max_num_wait_fail) {
        MS_LOG(ERROR) << "Maximum number of WaitForCommand retry reached: exiting training session.";
        MS_LOG(ERROR) << "Failed to connect to MindInsight debugger server. Please check the config "
                         "of debugger host and port.";
        Exit();
        run = true;
      } else {
        MS_LOG(ERROR) << "Number of consecutive WaitForCommand fail:" << num_wait_fail << "; Retry after "
                      << num_wait_fail << "s";
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 * num_wait_fail));
      }
      continue;
    }

    // get type of the command in reply
    DebuggerCommand cmd = GetCommand(reply);
    if (cmd == DebuggerCommand::kUnknownCMD) {
      MS_LOG(DEBUG) << "Debug: debugger received unknown command";
      continue;
    }

    MS_LOG(INFO) << "received command: ";
    switch (cmd) {
      case DebuggerCommand::kUnknownCMD:
        MS_LOG(INFO) << "UnknownCMD";
        break;
      case DebuggerCommand::kExitCMD:
        MS_LOG(INFO) << "ExitCMD";
        Exit();
        // Used for debugger termination
        run = true;
        break;
      case DebuggerCommand::kRunCMD:
        MS_LOG(INFO) << "RunCMD";
        if (GetRunLevel(reply) == "recheck") {
          MS_LOG(INFO) << "rechecking all watchpoints";
          SendWatchpoints(CheckWatchpoints());
        } else {
          // no longer the initial suspension.
          initial_suspend_ = false;
          // print run cmd content
          // get run_level and node_name
          run_level_ = GetRunLevel(reply);
          node_name_ = GetNodeName(reply);

          MS_LOG(INFO) << "run_level: " << run_level_;
          MS_LOG(INFO) << "node_name_: " << node_name_;

          // exit loop
          run = true;
        }
        break;
      case DebuggerCommand::kSetCMD:
        MS_LOG(INFO) << "SetCMD";
        MS_LOG(INFO) << "id: " << GetWatchpointID(reply);
        MS_LOG(INFO) << "delete: " << GetWatchpointDelete(reply);
        if (GetWatchpointDelete(reply)) {
          MS_LOG(INFO) << "Deleting watchpoint";
          RemoveWatchpoint(GetWatchpointID(reply));
        } else {
          MS_LOG(INFO) << "Setting watchpoint";
          MS_LOG(INFO) << "condition: " << GetWatchcondition(reply).condition();
          ProtoVector<WatchNode> recieved_nodes = GetWatchnodes(reply);
          for (const auto &node : recieved_nodes) {
            MS_LOG(INFO) << "node name: " << node.node_name();
            MS_LOG(INFO) << "node type: " << node.node_type();
          }
          ProtoVector<WatchCondition_Parameter> parameters = GetParameters(reply);
          for (const auto &parameter : parameters) {
            MS_LOG(INFO) << "parameter name: " << parameter.name();
            MS_LOG(INFO) << "parameter is disabled: " << parameter.disabled();
            MS_LOG(INFO) << "parameter value: " << parameter.value();
          }
          SetWatchpoint(GetWatchnodes(reply), GetWatchcondition(reply), GetWatchpointID(reply), GetParameters(reply));
        }
        break;
      case DebuggerCommand::kViewCMD:
        MS_LOG(INFO) << "ViewCMD";
        {
          // print view cmd content
          ProtoVector<TensorProto> received_tensors = GetTensors(reply);
          for (auto tensor : received_tensors) {
            MS_LOG(INFO) << "tensor node name: " << tensor.node_name();
            MS_LOG(INFO) << "tensor slot: " << tensor.slot();
            MS_LOG(INFO) << "tensor finished: " << std::boolalpha << tensor.finished() << std::noboolalpha;
            MS_LOG(INFO) << "tensor iter: " << tensor.iter();
            MS_LOG(INFO) << "tensor truncate: " << std::boolalpha << tensor.truncate() << std::noboolalpha;
          }
        }
        MS_LOG(INFO) << "Sending tensors";
        std::list<TensorProto> tensors = LoadTensors(GetTensors(reply));
        {
          // print view cmd reply
          for (auto tensor : tensors) {
            MS_LOG(INFO) << "tensor node name: " << tensor.node_name();
            MS_LOG(INFO) << "tensor slot: " << tensor.slot();
            MS_LOG(INFO) << "tensor finished: " << std::boolalpha << tensor.finished() << std::noboolalpha;
            MS_LOG(INFO) << "tensor iter: " << tensor.iter();
            MS_LOG(INFO) << "tensor truncate: " << std::boolalpha << tensor.truncate() << std::noboolalpha;
            MS_LOG(INFO) << "tensor dims: ";
            for (auto dim : tensor.dims()) {
              MS_LOG(INFO) << dim << ",";
            }
            MS_LOG(INFO) << "tensor dtype: " << tensor.data_type();
          }
        }
        EventReply send_tensors_reply = grpc_client_->SendTensors(tensors);
        if (send_tensors_reply.status() != send_tensors_reply.OK) {
          MS_LOG(ERROR) << "Error: SendTensors failed";
        }
        break;
    }
  }
}

void AddTensorProtoInfo(TensorProto *tensor_item, TensorProto tensor) {
  tensor_item->set_node_name(tensor.node_name());
  tensor_item->set_slot(tensor.slot());
  tensor_item->set_iter(tensor.iter());
  tensor_item->set_truncate(tensor.truncate());
  tensor_item->clear_tensor_content();
  tensor_item->clear_data_type();
  tensor_item->clear_dims();
}

void Debugger::SetWatchpoint(const ProtoVector<WatchNode> &nodes, const WatchCondition &condition, const int32_t id,
                             const ProtoVector<WatchCondition_Parameter> &parameters) {
  std::vector<std::tuple<std::string, bool>> check_node_list;
  std::vector<DebugServices::parameter_t> parameter_list;

  std::transform(nodes.begin(), nodes.end(), std::back_inserter(check_node_list),
                 [](const WatchNode &node) -> std::tuple<std::string, bool> {
                   return make_tuple(node.node_name(), node.node_type() == "scope");
                 });

  std::transform(
    parameters.begin(), parameters.end(), std::back_inserter(parameter_list),
    [](const WatchCondition_Parameter &parameter) -> DebugServices::parameter_t {
      return DebugServices::parameter_t{parameter.name(), parameter.disabled(), parameter.value(), parameter.hit()};
    });
  debug_services_->AddWatchpoint(id, condition.condition(), condition.value(), check_node_list, parameter_list);
  if (initial_suspend_ &&
      static_cast<DebugServices::CONDITION_TYPE>(condition.condition()) == DebugServices::CONDITION_TYPE::INIT)
    SendWatchpoints(CheckWatchpoints());
}

void Debugger::RemoveWatchpoint(const int32_t id) { debug_services_->RemoveWatchpoint(id); }

std::list<TensorProto> Debugger::LoadTensors(const ProtoVector<TensorProto> &tensors) const {
  std::vector<std::string> name;
  std::vector<std::string> ret_name;
  std::vector<char *> data_ptr;
  std::vector<unsigned int> data_size;
  std::vector<TypePtr> dtype;
  std::vector<std::vector<int64_t>> shape;

  std::transform(tensors.begin(), tensors.end(), std::back_inserter(name), GetTensorFullName);

  // ret_name will contain tensor names that are found in TensorLoader
  // items in ret_name will be in the same order with tensors if found
  debug_services_->ReadNodesTensors(name, &ret_name, &data_ptr, &data_size, &dtype, &shape);
  std::list<TensorProto> tensor_list;
  unsigned int result_index = 0;

  for (auto tensor : tensors) {
    int size_iter = 0;
    if (result_index >= ret_name.size() || ret_name[result_index] != GetTensorFullName(tensor)) {
      TensorProto tensor_item;
      tensor_item.set_finished(true);
      AddTensorProtoInfo(&tensor_item, tensor);
      tensor_list.push_back(tensor_item);
      continue;
    }
    int tensor_size = data_size[result_index];
    while (size_iter < tensor_size) {
      int chunk_size = CHUNK_SIZE;
      TensorProto tensor_item;
      tensor_item.set_finished(false);
      if (tensor_size - size_iter <= CHUNK_SIZE) {
        chunk_size = tensor_size - size_iter;
        tensor_item.set_finished(true);
      }
      AddTensorProtoInfo(&tensor_item, tensor);
      // return empty tensor if didn't find the requested tensor

      tensor_item.set_tensor_content(data_ptr[result_index] + size_iter, chunk_size);

      tensor_item.set_data_type(GetDebuggerNumberDataType(dtype[result_index]));
      for (auto &elem : shape[result_index]) {
        tensor_item.add_dims(elem);
      }
      // add tensor to result list and increment result_index to check next item in ret_name
      tensor_list.push_back(tensor_item);
      size_iter += CHUNK_SIZE;
    }
    result_index++;
  }
  return tensor_list;
}

void Debugger::Exit() {
  // clear resource before exit
  // For node level, debugger has to exit itself because main thread can only exit in step bundary;
  // For step level, debugger will notify main thread to exit;
  if (run_level_ == "node") {
    pipeline::ClearResAtexit();
    exit(1);
  } else if (run_level_ == "step" || device_target_ == kAscendDevice) {
    // Notify main thread to terminate
    pipeline::ExecutorPy::DebugTerminate(true);
  } else {
    pipeline::ClearResAtexit();
    exit(1);
  }
}

std::list<WatchpointHit> Debugger::CheckWatchpoints(const std::string &watchnode, const CNodePtr &kernel) {
  std::vector<std::string> name;
  std::vector<std::string> slot;
  std::vector<int> condition;
  std::vector<unsigned int> watchpoint_id;
  std::vector<std::string> overflow_ops;
  std::vector<std::vector<DebugServices::parameter_t>> parameters;
#ifdef ENABLE_D
  overflow_ops = CheckOpOverflow();
#endif
  auto tensor_loader = debug_services_->tensor_loader();
  std::vector<std::shared_ptr<TensorData>> tensor_list;
  if (watchnode.empty()) {
    tensor_list = tensor_loader->GetTensor();
  } else {
    tensor_list = tensor_loader->GetNodeTensorMap(watchnode);
    debug_services_->AddWeightsBiasInputs(&tensor_list, kernel);
  }
  debug_services_->CheckWatchpoints(&name, &slot, &condition, &watchpoint_id, &parameters, overflow_ops, tensor_list,
                                    initial_suspend_);
  std::list<WatchpointHit> hits;
  for (unsigned int i = 0; i < name.size(); i++) {
    WatchpointHit hit;
    std::vector<DebugServices::parameter_t> &parameter = parameters[i];
    hit.set_id(watchpoint_id[i]);

    // here TensorProto act as a tensor indicator, not sending tensor content
    TensorProto *tensor_item = hit.mutable_tensor();
    tensor_item->set_node_name(name[i]);
    tensor_item->set_slot(slot[i]);
    tensor_item->set_finished(true);

    WatchCondition *condition_item = hit.mutable_watch_condition();
    condition_item->set_condition(debugger::WatchCondition_Condition(condition[i]));
    for (const auto &p : parameter) {
      auto x = condition_item->mutable_params()->Add();
      x->set_name(p.name);
      x->set_disabled(p.disabled);
      x->set_value(p.value);
      x->set_hit(p.hit);
    }
    hits.push_back(hit);
  }
  return hits;
}

void Debugger::SendWatchpoints(const std::list<WatchpointHit> &points) {
  // send info about watchpoint
  if (!points.empty()) {
    EventReply reply = grpc_client_->SendWatchpointHits(points);
    if (reply.status() != reply.OK) {
      MS_LOG(ERROR) << "Error: SendWatchpointHits failed";
    }
  }
}

DebugServices *Debugger::debug_services() const { return debug_services_.get(); }

bool Debugger::debugger_enabled() const { return debugger_enabled_; }

DebuggerCommand GetCommand(const EventReply &reply) {
  DebuggerCommand cmd = DebuggerCommand::kUnknownCMD;
  switch (reply.cmd_case()) {
    case debugger::EventReply::CmdCase::kExit:
      cmd = DebuggerCommand::kExitCMD;
      break;
    case debugger::EventReply::CmdCase::kRunCmd:
      cmd = DebuggerCommand::kRunCMD;
      break;
    case debugger::EventReply::CmdCase::kSetCmd:
      cmd = DebuggerCommand::kSetCMD;
      break;
    case debugger::EventReply::CmdCase::kViewCmd:
      cmd = DebuggerCommand::kViewCMD;
      break;
    default:
      MS_LOG(DEBUG) << "Debug: UnknownCMD";
      break;
  }
  return cmd;
}

ProtoVector<WatchCondition_Parameter> GetParameters(const EventReply &reply) {
  if (!reply.has_set_cmd() || !reply.set_cmd().has_watch_condition()) {
    MS_LOG(ERROR) << "Error: Can not get Parameters from command. Returning default value: ProtoVector<Parameter>().";
    return ProtoVector<WatchCondition_Parameter>();
  }
  return reply.set_cmd().watch_condition().params();
}

ProtoVector<WatchNode> GetWatchnodes(const EventReply &reply) {
  if (!reply.has_set_cmd()) {
    MS_LOG(ERROR) << "Error: Not SetCMD, can not get WatchNodes. Returning default value: ProtoVector<WatchNode>().";
    return ProtoVector<WatchNode>();
  }
  return reply.set_cmd().watch_nodes();
}

std::string GetRunLevel(const EventReply &reply) {
  if (!reply.has_run_cmd()) {
    MS_LOG(ERROR) << "Error: Not RunCMD, can not get RunLevel. Returning default value: "
                     "";
    return "";
  }
  return reply.run_cmd().run_level();
}

std::string GetNodeName(const EventReply &reply) {
  if (!reply.has_run_cmd()) {
    MS_LOG(ERROR) << "Error: Not RunCMD, can not get NodeName. Returning default value: "
                     "";
    return "";
  }
  return reply.run_cmd().node_name();
}

WatchCondition GetWatchcondition(const EventReply &reply) {
  if (!reply.has_set_cmd() || !reply.set_cmd().has_watch_condition()) {
    MS_LOG(ERROR) << "Error: Can not get WatchCondition from command. Returning default value: WatchCondition().";
    return WatchCondition();
  }
  return reply.set_cmd().watch_condition();
}

int32_t GetWatchpointID(const EventReply &reply) {
  if (!reply.has_set_cmd()) {
    MS_LOG(ERROR) << "Error: Not SetCMD, can not get Watchpoint ID. Returning default value: 0.";
    return 0;
  }
  return reply.set_cmd().id();
}

bool GetWatchpointDelete(const EventReply &reply) {
  if (!reply.has_set_cmd()) {
    MS_LOG(ERROR) << "Error: Not SetCMD, can not get Watchpoint delete flag. Returning default value: false.";
    return false;
  }
  return reply.set_cmd().delete_();
}

ProtoVector<TensorProto> GetTensors(const EventReply &reply) {
  if (!reply.has_view_cmd()) {
    MS_LOG(ERROR) << "Error: Not ViewCMD, can not get Tensors. Returning default value: ProtoVector<TensorProto>().";
    return ProtoVector<TensorProto>();
  }
  return reply.view_cmd().tensors();
}

std::string GetTensorFullName(const TensorProto &tensor) {
  string node_name = tensor.node_name();
  if (tensor.truncate()) {
    // scopes in node name are seperated by '/'
    // use the name without scope if truncate is true
    std::size_t found = node_name.find_last_of("/");
    node_name = node_name.substr(found + 1);
  }
  return node_name + ":" + tensor.slot() + (tensor.iter() == "" ? "" : ":" + tensor.iter());
}

bool Debugger::partial_memory() { return partial_memory_; }

void Debugger::SetCurNode(std::string cur_name) {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  cur_name_ = cur_name;
}

std::string Debugger::run_level() const { return run_level_; }

void Debugger::SetStepNum(int32_t cur_num_step) {
  // access lock for public method
  std::lock_guard<std::mutex> a_lock(access_lock_);
  num_step_ = cur_num_step;
}

int32_t Debugger::step_num() const { return num_step_; }

uint64_t BytestoInt64(const std::vector<char> &buffer) {
  uint64_t ret;

  ret = ((uint64_t)buffer[7] << 56) | ((uint64_t)buffer[6] << 48) | ((uint64_t)buffer[5] << 40) |
        ((uint64_t)buffer[4] << 32) | ((uint64_t)buffer[3] << 24) | ((uint64_t)buffer[2] << 16) |
        ((uint64_t)buffer[1] << 8) | ((uint64_t)buffer[0]);

  return ret;
}

#define BUF_SIZ 256
std::vector<std::string> Debugger::CheckOpOverflow() {
  std::vector<double> bin_list;
  std::vector<std::string> op_names;
  DIR *d;
  struct dirent *dir = nullptr;
  d = opendir(overflow_bin_path_.c_str());
  if (d != nullptr) {
    while ((dir = readdir(d)) != NULL) {
      if (dir->d_type == DT_REG) {
        std::string file_path = overflow_bin_path_;
        file_path.append(dir->d_name);
        std::string file_name = dir->d_name;
        std::size_t found = file_name.find_last_of(".");
        if (found == std::string::npos) {
          continue;
        }
        std::string overflow_time = file_name.substr(found + 1);
        if (stod(overflow_time) <= last_overflow_bin_) {
          MS_LOG(INFO) << "File already processed " << file_name;
          continue;
        }
        bin_list.push_back(stod(overflow_time));
        std::fstream infile;
        infile.open(file_path.c_str(), std::ios::binary | std::ios::in);
        if (!infile.is_open()) {
          MS_LOG(ERROR) << "Failed to open overflow bin file " << file_name;
          continue;
        }
        infile.seekg(313, std::ios::beg);
        std::vector<char> buffer;
        buffer.resize(BUF_SIZ);
        infile.read(buffer.data(), BUF_SIZ);
        uint64_t stream_id = BytestoInt64(std::vector<char>(buffer.begin() + 8, buffer.end()));
        uint64_t task_id = BytestoInt64(std::vector<char>(buffer.begin() + 16, buffer.end()));
        MS_LOG(INFO) << "Overflow stream_id " << stream_id << ", task_id " << task_id << ".";
        auto op = debugger_->stream_task_to_opname_.find(std::make_pair(stream_id, task_id));
        if (op != debugger_->stream_task_to_opname_.end()) {
          MS_LOG(ERROR) << "Overflow detected on node " << op->second << std::endl;
          op_names.push_back(op->second);
        } else {
          MS_LOG(INFO) << "No overflow is detected " << std::endl;
        }
        infile.close();
      }
    }
  } else {
    MS_LOG(INFO) << "OverFlow bin directory does not exist!";
  }
  closedir(d);

  if (op_names.size()) {
    MS_LOG(ERROR) << "These operation overflows are detected " << op_names;
  }

  for (auto &i : bin_list) {
    if (i > last_overflow_bin_) {
      last_overflow_bin_ = i;
    }
  }

  return op_names;
}

void Debugger::SetTrainingDone(bool training_done) { training_done_ = training_done; }

bool Debugger::CheckPort(const char *port) {
  char *p = const_cast<char *>(port);
  int num = 0;
  if (*p == '0' && *(p + 1) != '\0') return false;
  while (*p != '\0') {
    if (*p < '0' || *p > '9') return false;
    num = num * 10 + (*p) - '0';
    if (num < 1 || num > 65535) return false;
    p++;
  }
  return true;
}

uint32_t Debugger::GetFirstRunGraphId() { return rungraph_id_list_.front(); }

void Debugger::LoadSingleAnfnode(const AnfNodePtr &anf_node, const size_t output_index) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (!anf_node->isa<Parameter>() && !anf_node->isa<ValueNode>()) {
    return;
  }
  bool keep_prev;
  if (anf_node->isa<Parameter>()) {
    keep_prev = true;
  } else {
    keep_prev = false;
  }
  // for parameters and value nodes, set its execution order to be 0;
  int exec_order = 0;
  std::string node_name = anf_node->fullname_with_scope();
  E2eDumpUtil::GetFileKernelName(NOT_NULL(&node_name));
  // check if output adde exists, if not, return;
  if (!AnfAlgo::OutputAddrExist(anf_node, output_index)) {
    return;
  }
  auto addr = AnfAlgo::GetOutputAddr(anf_node, output_index);
  MS_EXCEPTION_IF_NULL(addr);
  auto type = AnfAlgo::GetOutputInferDataType(anf_node, output_index);
  auto format = kOpFormat_DEFAULT;
  string tensor_name = node_name + ':' + "0";
  ShapeVector int_shapes;
  auto shape = AnfAlgo::GetOutputDeviceShape(anf_node, output_index);
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(int_shapes),
                       [](size_t inner_item) { return SizeToInt(inner_item); });
  bool ret = addr->LoadMemToHost(tensor_name, exec_order, format, int_shapes, type, 0, keep_prev);
  if (!ret) {
    MS_LOG(ERROR) << "LoadMemToHost:"
                  << ", tensor_name:" << tensor_name << ", host_format:" << format << ".!";
  }
}

void Debugger::LoadParametersAndConst() {
  if (!(debugger_enabled_ || CheckDebuggerDumpEnabled())) return;
  if (!(num_step_ == 0 || device_target_ == kAscendDevice ||
        (device_target_ == kGPUDevice && device::KernelRuntime::DumpDataEnabledIteration())))
    return;
  MS_EXCEPTION_IF_NULL(graph_ptr_);
  // load parameters
  MS_LOG(INFO) << "Start to load Parameters!";
  const auto &parameters = graph_ptr_->inputs();
  for (auto &item : parameters) {
    LoadSingleAnfnode(item, PARAMETER_OUTPUT_INDEX);
  }
  // load value nodes
  // get all constant avlues from the graph
  MS_LOG(INFO) << "Start to load value nodes!";
  const auto value_nodes = graph_ptr_->graph_value_nodes();
  for (auto &item : value_nodes) {
    LoadSingleAnfnode(item, VALUE_NODE_OUTPUT_INDEX);
  }
}

void Debugger::LoadGraphOutputs() {
  if (!(debugger_enabled() && device_target_ == kAscendDevice)) return;
  MS_EXCEPTION_IF_NULL(graph_ptr_);
  const auto &apply_kernels = graph_ptr_->execution_order();
  // for kernels, execution order starts from 1
  int exec_order = 1;
  for (const auto &node : apply_kernels) {
    MS_EXCEPTION_IF_NULL(node);
    auto node_name = AnfAlgo::GetCNodeName(node);
    std::string kernel_name = node->fullname_with_scope();
    auto output_size = AnfAlgo::GetOutputTensorNum(node);
    if (partial_memory_) {
      if (!debug_services_->IsWatchPoint(kernel_name, node)) {
        continue;
      }
    }
    for (size_t j = 0; j < output_size; ++j) {
      auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
      MS_EXCEPTION_IF_NULL(kernel_info);
      auto addr_test = kernel_info->GetOutputAddr(j);
      if (addr_test == nullptr) {
        MS_LOG(INFO) << "Cannot find output addr for slot " << j << " for " << kernel_name;
        continue;
      }
      auto addr = AnfAlgo::GetOutputAddr(node, j);
      MS_EXCEPTION_IF_NULL(addr);
      auto type = AnfAlgo::GetOutputInferDataType(node, j);
      auto format = kOpFormat_DEFAULT;
      string tensor_name = kernel_name + ':' + std::to_string(j);
      ShapeVector int_shapes;
      auto shape = AnfAlgo::GetOutputDeviceShape(node, j);
      (void)std::transform(shape.begin(), shape.end(), std::back_inserter(int_shapes),
                           [](size_t inner_item) { return SizeToInt(inner_item); });
      auto ret = addr->LoadMemToHost(tensor_name, exec_order, format, int_shapes, type, j, false);
      if (!ret) {
        MS_LOG(ERROR) << "LoadMemToHost:"
                      << ", tensor_name:" << tensor_name << ", host_format:" << format << ".!";
      }
    }
    exec_order = exec_order + 1;
  }
}

void Debugger::UpdateStepNum(const session::KernelGraph *graph) {
  // update step number if we are processing the first graph (to support multigraph)
  if (device_target_ == kGPUDevice && (debugger_enabled_ || device::KernelRuntime::DumpDataEnabledIteration()) &&
      (graph->graph_id() == debugger_->GetFirstRunGraphId())) {
    // access lock for public method
    std::lock_guard<std::mutex> a_lock(access_lock_);
    ++num_step_;
  }
}

void Debugger::ClearCurrentData() {
  if (device_target_ == kGPUDevice && (debugger_enabled_ || device::KernelRuntime::DumpDataEnabledIteration()))
    debug_services_->tensor_loader()->EmptyCurrentTensor();
}

}  // namespace mindspore
