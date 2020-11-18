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

#include "minddata/dataset/engine/ir/datasetops/map_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/map_op/map_op.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

MapNode::MapNode(std::shared_ptr<DatasetNode> child, std::vector<std::shared_ptr<TensorOperation>> operations,
                 std::vector<std::string> input_columns, std::vector<std::string> output_columns,
                 const std::vector<std::string> &project_columns, std::shared_ptr<DatasetCache> cache,
                 std::vector<std::shared_ptr<DSCallback>> callbacks)
    : operations_(operations),
      input_columns_(input_columns),
      output_columns_(output_columns),
      project_columns_(project_columns),
      DatasetNode(std::move(cache)),
      callbacks_(callbacks) {
  this->children.push_back(child);
}

std::vector<std::shared_ptr<DatasetOp>> MapNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  std::vector<std::shared_ptr<TensorOp>> tensor_ops;

  // Build tensorOp from tensorOperation vector
  // This is to ensure each iterator hold its own copy of the tensorOp objects.
  (void)std::transform(
    operations_.begin(), operations_.end(), std::back_inserter(tensor_ops),
    [](std::shared_ptr<TensorOperation> operation) -> std::shared_ptr<TensorOp> { return operation->Build(); });

  // This parameter will be removed with next rebase
  std::vector<std::string> col_orders;
  auto map_op = std::make_shared<MapOp>(input_columns_, output_columns_, tensor_ops, num_workers_, connector_que_size_);

  if (!callbacks_.empty()) {
    map_op->AddCallbacks(callbacks_);
  }

  if (!project_columns_.empty()) {
    auto project_op = std::make_shared<ProjectOp>(project_columns_);
    node_ops.push_back(project_op);
  }
  RETURN_EMPTY_IF_ERROR(AddCacheOp(&node_ops));

  node_ops.push_back(map_op);
  return node_ops;
}

Status MapNode::ValidateParams() {
  if (operations_.empty()) {
    std::string err_msg = "MapNode: No operation is specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (!input_columns_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("MapNode", "input_columns", input_columns_));
  }

  if (!output_columns_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("MapNode", "output_columns", output_columns_));
  }

  if (!project_columns_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("MapNode", "project_columns", project_columns_));
  }

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
