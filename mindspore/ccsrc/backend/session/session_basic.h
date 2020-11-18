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
#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_SESSION_BASIC_H
#define MINDSPORE_CCSRC_BACKEND_SESSION_SESSION_BASIC_H

#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <memory>
#include <map>
#include "backend/session/session_context.h"
#include "backend/session/kernel_graph.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "utils/any.h"
#include "utils/contract.h"
#include "runtime/device/kernel_info.h"
#include "utils/ms_context.h"
#if !defined(_WIN32) && !defined(_WIN64)
#include "debug/debugger/debugger.h"
#endif

namespace mindspore {
using GraphId = uint32_t;
using GraphInfo = std::string;
namespace session {
void ClearPythonParasMap();
using CallBackFunc = uint32_t (*)(uint32_t graph_id,
                                  const std::map<std::string, mindspore::tensor::TensorPtr> &params_list);
using AnyList = std::vector<Any>;
using AnyListPtr = std::shared_ptr<AnyList>;

struct OpRunInfo {
  std::string op_name;
  PrimitivePtr primitive;
  AbstractBasePtr abstract;
  ValuePtr value = nullptr;
  bool is_dynamic_shape = false;
};
using OpRunInfoPtr = std::shared_ptr<OpRunInfo>;
class Executor;
class SessionBasic : public std::enable_shared_from_this<SessionBasic> {
 public:
  SessionBasic() : context_(nullptr), summary_callback_(nullptr), device_id_(0) {
#if !defined(_WIN32) && !defined(_WIN64)
    debugger_ = nullptr;
#endif
  }

  virtual void Init(uint32_t device_id) { device_id_ = device_id; }

  void InitDevice(const std::string &device_name, uint32_t device_id);

  virtual ~SessionBasic() { summary_callback_ = nullptr; }

  GraphId CompileGraph(const GraphSegmentPtr &segment, const AnfNodePtrList &outputs);
  GraphId CompileGraph(NotNull<FuncGraphPtr> func_graph);
  void BuildGraph(GraphId graphId);
  void RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs);
  void RunGraphAsync(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs);
  void BuildOp(OpRunInfo *, const GraphInfo &, const std::vector<tensor::TensorPtr> &input_tensors,
               const std::vector<int64_t> &tensors_mask);
  void RunOp(OpRunInfo *, const GraphInfo &, const std::vector<tensor::TensorPtr> &input_tensors, VectorRef *outputs);

  virtual void RegisterSummaryCallBackFunc(const CallBackFunc &callback);

  void CreateCNodeKernelGraph(const AnfNodePtr node, KernelGraphPtr graph);

  std::shared_ptr<KernelGraph> ConstructKernelGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs);
  std::shared_ptr<KernelGraph> ConstructKernelGraph(const FuncGraphPtr &func_graph,
                                                    std::vector<KernelGraphPtr> *all_out_graph);

  CNodePtr CreateNewCNode(const CNodePtr &cnode, KernelGraph *graph,
                          std::unordered_map<AnfNodePtr, AnfNodePtr> *other_graph_cnode);
  CNodePtr CreateNewCNode(CNodePtr cnode, KernelGraph *graph);

  // get graph id in child graphs by ME front anf node pointer
  virtual GraphId GetGraphIdByNode(const AnfNodePtr &) const { return kInvalidGraphId; }
  virtual GraphId GetFinalRunGraph() const { return kInvalidGraphId; }
  void CheckPSModeConsistence(const KernelGraphPtr &Kernel_graph);
  void AssignParamKey(const KernelGraphPtr &kernel_graph);
  void InitPSParamAndOptim(const KernelGraphPtr &kernel_graph, const std::vector<tensor::TensorPtr> &inputs_const);
  virtual bool CheckModelInputs(uint32_t graph_id, const std::vector<tensor::TensorPtr> &inputs,
                                std::string *error_msg) const {
    return true;
  }
  virtual void GetModelInputsInfo(uint32_t graph_id, std::vector<tensor::TensorPtr> *inputs) const {}
  std::vector<tensor::TensorPtr> GetInputNeedLockTensors(const GraphId &graph_id,
                                                         const std::vector<tensor::TensorPtr> &inputs);
  // Get graph by graph id, if not exist return null ptr
  KernelGraphPtr GetGraph(GraphId graph_id) const;
#ifdef ENABLE_DEBUGGER
  // set debugger
  void SetDebugger() {
    debugger_ = Debugger::GetInstance();
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    debugger_->Init(device_id_, ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET));
  }
#endif

 private:
  CNodePtr CreateSwitchInput(const AnfNodePtr &node_input, KernelGraph *graph);
  std::vector<AnfNodePtr> CreateSwitchOrPartialNode(const CNodePtr &cnode, KernelGraph *graph);
  std::vector<AnfNodePtr> CreateValueNode(const CNodePtr &cnode, KernelGraph *graph);
  void CreateCNodeInputs(const CNodePtr &cnode, KernelGraph *graph, std::vector<AnfNodePtr> *cnode_inputs);
  std::vector<AnfNodePtr> CreateCallSwitchInputs(const CNodePtr &cnode, KernelGraph *graph);
  void GetCNodeInfo(const CNodePtr &cnode, std::vector<AnfNodePtr> *cnode_inputs);
  void GetNewCNodeInputs(const CNodePtr &cnode, KernelGraph *graph, std::vector<AnfNodePtr> *cnode_inputs,
                         std::unordered_map<AnfNodePtr, AnfNodePtr> *other_graph_cnode);

 protected:
  friend class Executor;
  friend class CompileNodesTask;
  friend class CompileGraphTask;
  friend class BuildGraphTask;
  friend class RunGraphTask;
  friend class BuildOpTask;
  friend class RunOpTask;
  virtual void CreateOutputTensors(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &input_tensors,
                                   VectorRef *outputs,
                                   std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node);
  virtual GraphId CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) = 0;
  virtual GraphId CompileGraphImpl(NotNull<FuncGraphPtr> func_graph) { return kInvalidGraphId; }
  virtual GraphId CompileGraphImpl(NotNull<FuncGraphPtr> func_graph, const std::vector<tensor::TensorPtr> &inputs) {
    MS_EXCEPTION(NotExistsError) << "Call an empty function";
  }
  virtual void BuildGraphImpl(GraphId) {}
  virtual void RunGraphImpl(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                            VectorRef *outputs) = 0;
  virtual void BuildOpImpl(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                           const std::vector<tensor::TensorPtr> &input_tensors,
                           const std::vector<int64_t> &tensors_mask) {}
  virtual void RunOpImpl(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                         const std::vector<tensor::TensorPtr> &input_tensors, VectorRef *outputs) {}
  void RunInfer(NotNull<FuncGraphPtr> func_graph, const std::vector<tensor::TensorPtr> &inputs);

  virtual void SetSummaryNodes(KernelGraph *graph);

  virtual void LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                             const std::vector<tensor::TensorPtr> &inputs_const) const;
  void UpdateOutputs(const std::shared_ptr<KernelGraph> &kernel_graph, VectorRef *const outputs,
                     const std::vector<tensor::TensorPtr> &input_tensors) const;
  void Reorder(std::vector<CNodePtr> *node_list);
  void Summary(KernelGraph *graph);
  // create graph output for RunOp
  void CreateOutputNode(const CNodePtr &cnode, const std::shared_ptr<KernelGraph> &graph);
  CNodePtr ConstructOutput(const AnfNodePtrList &outputs, const std::shared_ptr<KernelGraph> &graph);
  // create a single run op graph
  std::shared_ptr<KernelGraph> ConstructSingleOpGraph(const OpRunInfo &op_run_info,
                                                      const std::vector<tensor::TensorPtr> &input_tensors,
                                                      const std::vector<int64_t> &tensors_mask);
  // create a new kernel graph and update the graph sum
  KernelGraphPtr NewKernelGraph();
  std::vector<AnfNodePtr> CreateParameterFromTuple(const AnfNodePtr &node, KernelGraph *graph);
  virtual ParameterPtr CreateNewParameterFromParameter(const AnfNodePtr &anf, KernelGraph *graph);
  ValueNodePtr CreateValueNodeKernelGraph(const AnfNodePtr &anf, KernelGraph *graph);
  ParameterPtr CreateNewParameter(const AnfNodePtr &anf, KernelGraph *graph);
  AnfNodePtr CreateNewParameterFromCNode(const AnfNodePtr &anf, KernelGraph *graph);
  void AddParameterToGraphInputs(const std::vector<AnfNodePtr> &parameters, KernelGraph *graph);
  void InitInternalOutputParameter(const AnfNodePtr &out_node, const AnfNodePtr &parameter);
  AnfNodePtr FindPullNode(const AnfNodePtr &push_node, const std::vector<AnfNodePtr> &node_list);
  void UpdateGraphDynamicShapeAttr(const NotNull<KernelGraphPtr> &root_graph);

  std::unordered_map<GraphId, std::shared_ptr<KernelGraph>> graphs_;
  std::unordered_map<GraphInfo, std::shared_ptr<KernelGraph>> run_op_graphs_;
  std::unordered_map<FuncGraphPtr, KernelGraphPtr> front_backend_graph_map_;
  std::shared_ptr<Context> context_;
  CallBackFunc summary_callback_;
  static GraphId graph_sum_;
  uint32_t device_id_;
  std::shared_ptr<Executor> executor_;
#if !defined(_WIN32) && !defined(_WIN64)
  std::shared_ptr<Debugger> debugger_;
#endif
};

using SessionPtr = std::shared_ptr<session::SessionBasic>;
using NamedSummaryOutputs = std::map<std::string, std::pair<AnfNodePtr, int>>;
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_SESSION_BASIC_H
