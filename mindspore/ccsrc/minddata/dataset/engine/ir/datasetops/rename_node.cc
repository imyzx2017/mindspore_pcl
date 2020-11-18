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

#include "minddata/dataset/engine/ir/datasetops/rename_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/rename_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

// Function to build RenameOp
RenameNode::RenameNode(std::shared_ptr<DatasetNode> child, const std::vector<std::string> &input_columns,
                       const std::vector<std::string> &output_columns)
    : input_columns_(input_columns), output_columns_(output_columns) {
  this->children.push_back(child);
}

Status RenameNode::ValidateParams() {
  if (input_columns_.size() != output_columns_.size()) {
    std::string err_msg = "RenameNode: input and output columns must be the same size";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetColumnParam("RenameNode", "input_columns", input_columns_));

  RETURN_IF_NOT_OK(ValidateDatasetColumnParam("RenameNode", "output_columns", output_columns_));

  return Status::OK();
}

std::vector<std::shared_ptr<DatasetOp>> RenameNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<RenameOp>(input_columns_, output_columns_, connector_que_size_));
  return node_ops;
}

}  // namespace dataset
}  // namespace mindspore
