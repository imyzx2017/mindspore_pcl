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

#include "backend/optimizer/ascend/format_type/merge_cast_to_op.h"

#include <memory>
#include <vector>
#include <algorithm>
#include <string>

#include "backend/session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "base/core_ops.h"

namespace mindspore {
namespace opt {
namespace {
const size_t kCastInputNum = 2;
const size_t kTupleGetitemInputNum = 3;
bool AlternativeKernelInfoForInput(const CNodePtr &node, const TypeId dst_type, const size_t change_idx,
                                   const std::shared_ptr<kernel::KernelBuildInfo> &candidate_kernel_info) {
  if (node == nullptr || node->kernel_info() == nullptr || candidate_kernel_info == nullptr) {
    return false;
  }

  // checkout inputs' fmt and dtype except index equal change_idx
  for (size_t i = 0; i < candidate_kernel_info->GetInputNum(); i++) {
    if (i == change_idx) {
      if (candidate_kernel_info->GetInputDeviceType(i) != dst_type ||
          candidate_kernel_info->GetInputFormat(i) != AnfAlgo::GetInputFormat(node, i)) {
        return false;
      }
    } else if (candidate_kernel_info->GetInputDeviceType(i) != AnfAlgo::GetInputDeviceDataType(node, i) ||
               candidate_kernel_info->GetInputFormat(i) != AnfAlgo::GetInputFormat(node, i)) {
      return false;
    }
  }

  // check outputs's fmt and dtype
  for (size_t i = 0; i < candidate_kernel_info->GetOutputNum(); i++) {
    if (candidate_kernel_info->GetOutputDeviceType(i) != AnfAlgo::GetOutputDeviceDataType(node, i) ||
        candidate_kernel_info->GetOutputFormat(i) != AnfAlgo::GetOutputFormat(node, i)) {
      return false;
    }
  }
  return true;
}

bool GetNextNodeAndCastIndex(const FuncGraphPtr &graph, const AnfNodePtr &node, AnfNodePtr *next_node,
                             size_t *cast_index) {
  auto output_node_list = GetRealNodeUsedList(graph, node);
  MS_EXCEPTION_IF_NULL(output_node_list);
  if (output_node_list->size() != 1) {
    return false;
  }
  auto node_pair = output_node_list->at(0);
  *next_node = node_pair.first;
  *cast_index = node_pair.second - 1;
  return true;
}

bool CheckInputs(const CNodePtr &node, const std::shared_ptr<kernel::KernelBuildInfo> &kernel_info) {
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (AnfAlgo::GetInputTensorNum(node) != kernel_info->GetInputNum()) {
    return false;
  }

  for (size_t index = 0; index < kernel_info->GetInputNum(); ++index) {
    if (AnfAlgo::GetInputFormat(node, index) != kernel_info->GetInputFormat(index) ||
        AnfAlgo::GetInputDeviceDataType(node, index) != kernel_info->GetInputDeviceType(index)) {
      return false;
    }
  }
  return true;
}

bool CheckOtherOutputs(const CNodePtr &node, const std::shared_ptr<kernel::KernelBuildInfo> &kernel_info,
                       const size_t idx) {
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (AnfAlgo::GetOutputTensorNum(node) != kernel_info->GetOutputNum()) {
    return false;
  }
  for (size_t index = 0; index < kernel_info->GetOutputNum(); ++index) {
    if (idx == index) {
      continue;
    }
    if (AnfAlgo::GetOutputFormat(node, index) != kernel_info->GetOutputFormat(index) ||
        AnfAlgo::GetOutputDeviceDataType(node, index) != kernel_info->GetOutputDeviceType(index)) {
      return false;
    }
  }
  return true;
}

bool CheckIndexOutput(const CNodePtr &node, const std::shared_ptr<kernel::KernelBuildInfo> &kernel_info, size_t index) {
  if (kernel_info == nullptr) {
    return false;
  }

  if (AnfAlgo::GetOutputDeviceDataType(node, 0) != kernel_info->GetOutputDeviceType(index)) {
    return false;
  }
  if (AnfAlgo::GetOutputInferShape(node, 0).size() == 4 && AnfAlgo::GetOutputFormat(node, 0) == kOpFormat_NCHW &&
      kernel_info->GetOutputFormat(index) == kOpFormat_DEFAULT) {
    return true;
  }
  return AnfAlgo::GetOutputFormat(node, 0) == kernel_info->GetOutputFormat(index);
}

void ChangeNodeInferInfo(const CNodePtr &cnode, const CNodePtr &cast, const size_t cast_index) {
  using Shape = std::vector<size_t>;
  auto cast_dtype = AnfAlgo::GetOutputInferDataType(cast, 0);
  auto cast_shape = AnfAlgo::GetOutputInferShape(cast, 0);
  std::vector<Shape> shapes;
  std::vector<TypeId> types;
  for (size_t index = 0; index < AnfAlgo::GetOutputTensorNum(cnode); ++index) {
    if (cast_index == index) {
      shapes.emplace_back(cast_shape);
      types.emplace_back(cast_dtype);
      continue;
    }
    shapes.emplace_back(AnfAlgo::GetOutputInferShape(cnode, index));
    types.emplace_back(AnfAlgo::GetOutputInferDataType(cnode, index));
  }
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, cnode.get());
}

AnfNodePtr MergeCastToNextOp(const FuncGraphPtr &graph, const CNodePtr &node, const KernelQueryPtr kernel_query) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(kernel_query);
  AnfNodePtr next_node = nullptr;
  size_t cast_index = 0;
  if (!GetNextNodeAndCastIndex(graph, node, &next_node, &cast_index)) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(next_node);
  if (!next_node->isa<CNode>() || !AnfAlgo::IsRealKernel(next_node)) {
    return nullptr;
  }
  auto next_cnode = next_node->cast<CNodePtr>();
  if (AnfAlgo::IsGraphKernel(next_node)) {
    return nullptr;
  }
  auto next_op_name = AnfAlgo::GetCNodeName(next_node);
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
  kernel_query->Query(next_cnode, &kernel_info_list);

  auto dst_type_id = AnfAlgo::GetInputDeviceDataType(node, 0);
  auto alternative_kernel_info = std::find_if(
    kernel_info_list.begin(), kernel_info_list.end(),
    [&next_cnode, &dst_type_id, &cast_index](const std::shared_ptr<kernel::KernelBuildInfo> &candidate_kernel_info) {
      return AlternativeKernelInfoForInput(next_cnode, dst_type_id, cast_index, candidate_kernel_info);
    });
  if (alternative_kernel_info == kernel_info_list.end()) {
    return nullptr;
  }
  auto ori_kernel_info = AnfAlgo::GetSelectKernelBuildInfo(next_node);
  MS_LOG(INFO) << "Found alternative kernel info for current anf kernel " << next_cnode->DebugString()
               << "ori kernel info" << ori_kernel_info->ToString() << "alternative kernel info"
               << (*alternative_kernel_info)->ToString();
  AnfAlgo::SetSelectKernelBuildInfo(*alternative_kernel_info, next_cnode.get());
  if (node->inputs().size() < kCastInputNum) {
    MS_LOG(EXCEPTION) << "Op[" << node->DebugString() << "] has wrong input num:";
  }
  return node->input(1);
}

bool GetPriorOp(const AnfNodePtr &x_node, CNodePtr *prior_op, bool *single_output, size_t *output_idx) {
  MS_EXCEPTION_IF_NULL(x_node);
  if (x_node->isa<CNode>()) {
    auto x_cnode = x_node->cast<CNodePtr>();
    *prior_op = x_cnode;
    // when x_node is tuple_getitem
    if (AnfAlgo::GetCNodeName(x_node) == prim::kPrimTupleGetItem->name()) {
      if (x_cnode->inputs().size() < kTupleGetitemInputNum) {
        MS_LOG(EXCEPTION) << "tuple getitem node has wrong input num" << x_cnode->inputs().size();
      }
      MS_EXCEPTION_IF_NULL(output_idx);
      AnfNodePtr input1 = x_cnode->input(1);
      MS_EXCEPTION_IF_NULL(input1);
      if (!input1->isa<CNode>()) {
        return false;
      }
      *prior_op = input1->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(*prior_op);
      AnfNodePtr input2 = x_cnode->input(2);
      MS_EXCEPTION_IF_NULL(input2);
      auto value_ptr = input2->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_ptr);
      *output_idx = LongToSize(GetValue<int64_t>(value_ptr->value()));
      *single_output = false;
    }
    return AnfAlgo::IsRealKernel(*prior_op);
  }
  return false;
}

AnfNodePtr MergeCastToPriorOp(const FuncGraphPtr &graph, const CNodePtr &cur_node, const KernelQueryPtr kernel_query) {
  MS_EXCEPTION_IF_NULL(cur_node);
  MS_EXCEPTION_IF_NULL(kernel_query);
  if (cur_node->inputs().size() < kCastInputNum) {
    MS_LOG(EXCEPTION) << "op[Cast] has wrong input num:";
  }
  AnfNodePtr x_node = cur_node->input(1);
  if (IsUsedByOthers(graph, x_node)) {
    return nullptr;
  }

  CNodePtr prior_op = nullptr;
  bool single_output = true;
  size_t output_idx = 0;
  if (!GetPriorOp(x_node, &prior_op, &single_output, &output_idx)) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(prior_op);
  if (AnfAlgo::IsGraphKernel(prior_op)) {
    return nullptr;
  }

  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
  kernel_query->Query(prior_op, &kernel_info_list);
  auto kernel_info_it = std::find_if(
    kernel_info_list.begin(), kernel_info_list.end(),
    [&prior_op, &cur_node, &output_idx](const std::shared_ptr<kernel::KernelBuildInfo> &item_kernel_info) {
      return CheckInputs(prior_op, item_kernel_info) && CheckOtherOutputs(prior_op, item_kernel_info, output_idx) &&
             CheckIndexOutput(cur_node, item_kernel_info, output_idx);
    });
  if (kernel_info_it == kernel_info_list.end()) {
    return nullptr;
  }
  auto ori_kernel_info = AnfAlgo::GetSelectKernelBuildInfo(prior_op);
  MS_LOG(INFO) << "Found alternative kernel info for current anf kernel " << prior_op->DebugString()
               << "ori kernel info" << ori_kernel_info->ToString() << "alternative kernel info"
               << (*kernel_info_it)->ToString();
  AnfAlgo::SetSelectKernelBuildInfo(*kernel_info_it, prior_op.get());
  ChangeNodeInferInfo(prior_op, cur_node, output_idx);
  if (!single_output) {
    MS_EXCEPTION_IF_NULL(x_node);
    ChangeNodeInferInfo(x_node->cast<CNodePtr>(), cur_node, 0);
  }
  auto prior_name = AnfAlgo::GetCNodeName(prior_op);
  if (prior_name == kFive2FourOpName) {
    AnfAlgo::CopyNodeAttr("dst_type", "dstType", cur_node, prior_op);
  } else if (prior_name == kFour2FiveOpName) {
    AnfAlgo::CopyNodeAttr("dst_type", cur_node, prior_op);
  }
  return single_output ? prior_op : x_node;
}
}  // namespace

const BaseRef MergeCastToOp::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  return VectorRef({prim::kPrimCast, X});
}

const AnfNodePtr MergeCastToOp::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  auto new_node = MergeCastToNextOp(graph, cnode, kernel_query_);
  if (new_node == nullptr) {
    new_node = MergeCastToPriorOp(graph, cnode, kernel_query_);
  }
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
