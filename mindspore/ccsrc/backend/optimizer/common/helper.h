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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_HELPER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_HELPER_H_

#include <vector>
#include <memory>
#include <utility>
#include <string>
#include <set>
#include <unordered_set>
#include "ir/func_graph.h"
#include "backend/session/kernel_graph.h"
#include "utils/ms_utils.h"
#include "backend/optimizer/common/pattern_engine.h"

namespace mindspore {
namespace opt {
constexpr size_t kTransOpInputNum = 2;
constexpr size_t kCastInputNum = 2;
constexpr size_t kDependInputNum = 3;
constexpr size_t kReluInputNum = 2;
constexpr size_t kReluGradInputNum = 3;
constexpr size_t kAddInputNum = 3;
constexpr size_t kAddNInputNum = 3;
constexpr size_t kTupleGetitemInputNum = 3;
constexpr size_t kConvInputNum = 3;
constexpr size_t kRealDivInputNum = 3;
constexpr size_t kSqrtInputNum = 2;
constexpr size_t kMulInputNum = 3;
constexpr size_t kRsqrtInputNum = 2;
constexpr size_t kSubInputNum = 3;
constexpr size_t kAssignSubInputNum = 3;

constexpr size_t kConvBn1OutputNum = 3;
constexpr size_t kBn2ReluOutputNum = 4;

constexpr size_t kBnInputNum = 6;
constexpr size_t kBnOutputNum = 5;
constexpr size_t kBatchNormInputNum = 5;
constexpr size_t kBatchNormOutputNum = 5;

constexpr size_t kBN1OutputNum = 2;
constexpr size_t kBN2OutputNum = 3;
constexpr size_t kBN3OutputNum = 1;

constexpr size_t kBNGradInputNum = 6;
constexpr size_t kBNGradOutputNum = 3;

constexpr size_t kBNGrad1OutputNum = 3;
constexpr size_t kBNGrad2OutputNum = 5;
constexpr size_t kBNGrad3OutputNum = 1;

constexpr size_t kBNTrainingReduceOutputNum = 2;
constexpr size_t kBNTrainingUpdateOutputNum = 5;
constexpr size_t kBNTrainingUpdateV2OutputNum = 3;
constexpr size_t kBNTrainingUpdateV3OutputNum = 5;
constexpr size_t kBNTrainingUpdateGradOutputNum = 2;

constexpr size_t kSingleOutputNum = 1;
constexpr size_t kSumNodeInputNum = 2;
constexpr size_t kSquareNodeInputNum = 2;
constexpr size_t kSquareSumv2OutputNum = 2;
constexpr size_t kMinimumInputNum = 3;

constexpr size_t kLambNextMVWithDecayInputNum = 7;
constexpr size_t kLambNextMVWithDecayConstantMulInputNum = 5;
constexpr size_t kLambNextMVWithDecayOutputNum = 4;
constexpr size_t kLambNextMVWithDecayV1OutputNum = 4;
constexpr size_t kLambNextRightOutputNum = 2;
constexpr size_t kLambUpdateWithLrV2InputNum = 8;
constexpr size_t kLambNextMVRuleInputNum = 14;
constexpr size_t kLambNextMVRuleOutputNum = 4;
constexpr size_t kBackendReshapeInputNum = 2;
constexpr size_t kBackendTransposeInputNum = 2;
constexpr size_t kAdamApplyOneWithDecayOutputNum = 3;
constexpr size_t kLayerNormBetaGammaBackpropInputNum = 5;
constexpr size_t kLayerNormBetaGammaBackpropOutputNum = 2;
constexpr size_t kLayerNormGradInputNum = 6;
constexpr size_t kAdamApplyOneOutputNum = 3;
constexpr size_t kBackendTransDataInputNum = 2;
constexpr size_t kApplyMomentumInputNum = 6;
constexpr size_t kBiasAddInputNum = 3;
constexpr size_t kTopkInputNum = 3;
constexpr size_t kLarsV2InputNum = 5;
constexpr size_t kFusedMulApplyMomentumOutputNum = 2;
constexpr size_t kSplitInputNum = 2;
constexpr size_t kUnsortedSegmentSumInputNum = 2;

enum FusedBatchNormInput {
  kX = 1,
  kVariance = 5,
};
enum FusedBatchNormOutput {
  kY = 0,
  kRunningMean,
  kRunningVariance,
  kSaveMean,
  kSaveInvVariance,
};
enum ConvBn1Output {
  kData = 0,
  kVarPart,
  kMean,
};

std::vector<int64_t> Convert2Int(const std::vector<size_t> &v);

std::vector<int64_t> Convert2Long(const std::vector<size_t> &v);

// check whether node depends on either of nodes or not
bool IsDepend(const FuncGraph &graph, const AnfNodePtr &node, const std::vector<AnfNodePtr> &nodes);

bool UnVisited(const BaseRef &n);

bool Visited(const BaseRef &n);

// check if the input node is CNode, then check it's input_size, return CNodePtr if check success.
CNodePtr CheckAnfNodeIfCNodeAndInputSize(const AnfNodePtr &node, size_t input_size);

void CheckCNodeInputSize(const CNodePtr &cnode, size_t input_size);

bool HasSymmetricalKernelInfo(const AnfNodePtr &node_x, const AnfNodePtr &node_y);

const AnfNodePtr EliminateDependTransop(const FuncGraphPtr &func_graph, const AnfNodePtr &node);

void CreateOutputsOfConvBn1(const FuncGraphPtr &func_graph, const CNodePtr &conv_cnode, const CNodePtr &bn_cnode,
                            std::vector<AnfNodePtr> *conv_bn1_outputs);

void CreateOutputsOfFusedBn2(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &fused_bn1_outputs,
                             const CNodePtr &bn_node, std::vector<AnfNodePtr> *fused_bn2_outputs);
void CreateOutputsOfFusedBn3(const FuncGraphPtr &graph, const AnfNodePtr &data_input,
                             const std::vector<AnfNodePtr> &fused_bn1_outputs,
                             const std::vector<AnfNodePtr> &fused_bn2_outputs, const CNodePtr &bn_node,
                             std::vector<AnfNodePtr> *fused_bn3_outputs);

void CreateMultipleOutputsOfAnfNode(const FuncGraphPtr &kernel_graph, const AnfNodePtr &anf_node_ptr, size_t output_num,
                                    std::vector<AnfNodePtr> *outputs);

tensor::TensorPtr CreateTensorWithValueTuple(const ValueTuplePtr &value_tuple_ptr, const TypePtr &type_ptr,
                                             size_t data_length);

tensor::TensorPtr CreateTupleTensor(const ValueTuplePtr &value_tuple);

bool IsAllNopNode(const session::KernelGraph *const graph);

bool IsNopNode(const AnfNodePtr &node);

void HideNopNode(session::KernelGraph *const graph);

void RemoveNopNode(session::KernelGraph *const graph);

AnfNodePtr CreatTupleGetItemNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, size_t output_idx);

bool IsUsedByOthers(const FuncGraphPtr &graph, const AnfNodePtr &node);

std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> GetRealNodeUsedList(const FuncGraphPtr &graph,
                                                                             const AnfNodePtr &node);

size_t GetRealNodeNum(const FuncGraphPtr &graph, const AnfNodePtr &node);

std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> GetRealNodeUsedListByOutputIdx(const FuncGraphPtr &graph,
                                                                                        const AnfNodePtr &node,
                                                                                        size_t output_index);
bool IsNotRealUsedByOthers(const FuncGraphPtr &graph, const AnfNodePtr &node);

void ConstInputToAttr(const CNodePtr &cnode, const std::unordered_set<size_t> &input_attrs);

bool AnfEqual(const BaseRef &a, const BaseRef &b);

bool CNodeTypeEqual(const BaseRef &a, const BaseRef &b);

AnfNodePtr SexpToNode(const BaseRef &sexp, const BaseRef &graph, PrimitiveVarMap *primitive_vars,
                      bool multigraph = false);

// Check var_node in two equivs is the same node
bool IsSameNode(const EquivPtr &equiv1, const EquivPtr &equiv2, const VarPtr &var_node);

// Get anf_node from equiv by var_node
AnfNodePtr GetAnfNodeByVar(const EquivPtr &equiv, const VarPtr &var_node);

// Compare tuple getitem's index, return bool[n1's index < n2's index]
bool CompareTupleGetitem(const AnfNodePtr &n1, const AnfNodePtr &n2);

// Get attr which is bool from cnode
bool GetBoolAttr(const AnfNodePtr &node, const std::string &attr_name);

// Check node's data type is in supported data type set
bool CheckSupportDataType(const AnfNodePtr &node, const std::set<TypeId> &supported_data_type_set);

// Create a new value node of func graph,not kernel graph
ValueNodePtr MakeValueNode(const ValueNodePtr &value_node);

// Transfer depend or control_depend to the new node
void TransferDepend(const CNodePtr &old_node, const FuncGraphPtr &graph, const CNodePtr &new_node);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_HELPER_H_
