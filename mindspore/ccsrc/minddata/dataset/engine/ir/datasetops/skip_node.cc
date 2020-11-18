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

#include "minddata/dataset/engine/ir/datasetops/skip_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/skip_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

// Constructor for SkipNode
SkipNode::SkipNode(std::shared_ptr<DatasetNode> child, int32_t count) : skip_count_(count) {
  this->children.push_back(child);
}

// Function to build the SkipOp
std::vector<std::shared_ptr<DatasetOp>> SkipNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<SkipOp>(skip_count_, connector_que_size_));
  return node_ops;
}

// Function to validate the parameters for SkipNode
Status SkipNode::ValidateParams() {
  if (skip_count_ <= -1) {
    std::string err_msg = "SkipNode: skip_count should not be negative, skip_count: " + std::to_string(skip_count_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
