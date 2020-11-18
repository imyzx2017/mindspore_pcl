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
#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_EXECUTOR_H
#define MINDSPORE_CCSRC_BACKEND_SESSION_EXECUTOR_H

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <list>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "backend/session/session_basic.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "utils/any.h"
#include "utils/contract.h"
#include "utils/comm_manager.h"

namespace mindspore {
namespace session {
enum TaskType {
  kUnKnown,
  kExit,
  kCompileNodes,
  kCompileGraph,
  kBuildGraph,
  kBuildOp,
  kRunGraph,
  kRunOp,
  kCreateCommGroup,
  kDestroyCommGroup
};

class Task {
 public:
  Task() = default;
  virtual ~Task() = default;
  SessionPtr session_{nullptr};
  TaskType type_{kUnKnown};
  bool sync_run_{false};
  virtual void Run() {}
};

class CompileNodesTask : public Task {
 public:
  CompileNodesTask() { type_ = kCompileNodes; }
  ~CompileNodesTask() override = default;
  void Run() override;
  GraphSegmentPtr segment_;
  AnfNodePtrList output_nodes_;
  GraphId graph_id_{0};
};

class CompileGraphTask : public Task {
 public:
  CompileGraphTask() { type_ = kCompileGraph; }
  ~CompileGraphTask() override = default;
  void Run() override;
  FuncGraphPtr func_graph_{nullptr};
  GraphId graph_id_{0};
};

class BuildGraphTask : public Task {
 public:
  BuildGraphTask() { type_ = kBuildGraph; }
  ~BuildGraphTask() override = default;
  void Run() override;
  GraphId graph_id_{0};
};

class RunGraphTask : public Task {
 public:
  RunGraphTask() { type_ = kRunGraph; }
  ~RunGraphTask() override = default;
  void Run() override;
  std::vector<tensor::TensorPtr> input_tensors_;
  std::vector<tensor::TensorPtr> input_need_wait_tensors_;
  std::vector<tensor::TensorPtr> input_need_lock_tensors_;
  VectorRef outputs_;
  GraphId graph_id_{0};
  std::map<tensor::TensorPtr, session::KernelWithIndex> tensor_to_node_;
};

class BuildOpTask : public Task {
 public:
  BuildOpTask() { type_ = kBuildOp; }
  ~BuildOpTask() override = default;
  void Run() override;
  OpRunInfo *op_run_info_{nullptr};
  GraphInfo graph_info_;
  std::vector<tensor::TensorPtr> input_tensors_;
  std::vector<int64_t> tensors_mask_;
};

class RunOpTask : public Task {
 public:
  RunOpTask() { type_ = kRunOp; }
  ~RunOpTask() override = default;
  void Run() override;
  OpRunInfo *op_run_info_{nullptr};
  GraphInfo graph_info_;
  std::vector<tensor::TensorPtr> input_tensors_;
  VectorRef outputs_;
};

class CreateCommGroupTask : public Task {
 public:
  CreateCommGroupTask() { type_ = kCreateCommGroup; }
  ~CreateCommGroupTask() override = default;
  void Run() override;
  std::string group_name_;
  std::vector<uint32_t> ranks_;
  bool result_{false};
};

class DestroyCommGroupTask : public Task {
 public:
  DestroyCommGroupTask() { type_ = kDestroyCommGroup; }
  ~DestroyCommGroupTask() override = default;
  void Run() override;
  std::string group_name_;
  bool result_{false};
};

class ExitTask : public Task {
 public:
  ExitTask() { type_ = kExit; }
  ~ExitTask() override = default;
};

class Executor {
 public:
  Executor(const std::string &device_name, uint32_t device_id);
  ~Executor();
  void WorkerLoop();
  void WorkerJoin();
  GraphId CompileGraph(const SessionPtr &session, const GraphSegmentPtr &segment, const AnfNodePtrList &outputs);
  GraphId CompileGraph(const SessionPtr &session, NotNull<FuncGraphPtr> func_graph);
  void BuildGraph(const SessionPtr &session, GraphId graphId);
  void RunGraph(const SessionPtr &session, const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                VectorRef *outputs);
  void RunGraphAsync(const SessionPtr &session, const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                     VectorRef *outputs);
  void BuildOp(const SessionPtr &session, OpRunInfo *op_run_info, const GraphInfo &graph_info,
               const std::vector<tensor::TensorPtr> &input_tensors, const std::vector<int64_t> &tensors_mask);
  void RunOp(const SessionPtr &session, OpRunInfo *op_run_info, const GraphInfo &graph_info,
             const std::vector<tensor::TensorPtr> &input_tensors, VectorRef *outputs);
  void OnRunGraphFinished();
  bool CreateCommGroup(const std::string &group_name, std::vector<uint32_t> ranks);
  bool DestroyCommGroup(const std::string &group_name);

 private:
  void SyncRunTask(const std::shared_ptr<Task> &task);
  void UpdateOutputTensors(VectorRef *outputs,
                           const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node);
  std::vector<std::shared_ptr<RunGraphTask>> GetNewReadyTasks();
  bool IsTaskReady(const std::shared_ptr<RunGraphTask> &task);
  void CheckException();
  void OnWorkerExit();

  uint32_t device_id_;
  std::string device_name_;
  std::mutex task_mutex_;
  std::mutex pending_task_mutex_;
  std::condition_variable task_cond_var_;
  std::condition_variable sync_cond_var_;
  std::queue<std::shared_ptr<Task>> ready_tasks_;
  std::list<std::shared_ptr<RunGraphTask>> pending_tasks_;
  std::vector<std::shared_ptr<Task>> done_tasks_;
  std::shared_ptr<std::thread> worker_;
};
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_EXECUTOR_H
