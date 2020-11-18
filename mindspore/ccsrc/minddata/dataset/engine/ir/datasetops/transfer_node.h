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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_TRANSFER_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_TRANSFER_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {

class TransferNode : public DatasetNode {
 public:
  /// \brief Constructor
  TransferNode(std::shared_ptr<DatasetNode> child, bool send_epoch_end);

  /// \brief Destructor
  ~TransferNode() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return shared pointer to the list of newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams() override;

  static Status get_distribution(std::shared_ptr<DatasetNode> ds, int32_t *device_id);

 private:
  std::string queue_name_;
  int32_t device_id_;
  std::string device_type_;
  int32_t prefetch_size_;
  bool send_epoch_end_;
  int32_t total_batch_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_TRANSFER_NODE_H_
