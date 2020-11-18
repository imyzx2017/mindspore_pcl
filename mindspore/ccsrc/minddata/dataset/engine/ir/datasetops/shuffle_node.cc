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

#include "minddata/dataset/engine/ir/datasetops/shuffle_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/shuffle_op.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

// Constructor for ShuffleNode
ShuffleNode::ShuffleNode(std::shared_ptr<DatasetNode> child, int32_t shuffle_size, bool reset_every_epoch)
    : shuffle_size_(shuffle_size), shuffle_seed_(GetSeed()), reset_every_epoch_(reset_every_epoch) {
  this->children.push_back(child);
}

// Function to build the ShuffleOp
std::vector<std::shared_ptr<DatasetOp>> ShuffleNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<ShuffleOp>(shuffle_size_, shuffle_seed_, connector_que_size_, reset_every_epoch_,
                                                 rows_per_buffer_));
  return node_ops;
}

// Function to validate the parameters for ShuffleNode
Status ShuffleNode::ValidateParams() {
  if (shuffle_size_ <= 1) {
    std::string err_msg = "ShuffleNode: Invalid input, shuffle_size: " + std::to_string(shuffle_size_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
