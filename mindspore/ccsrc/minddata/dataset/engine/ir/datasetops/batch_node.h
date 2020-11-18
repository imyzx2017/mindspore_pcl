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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_BATCH_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_BATCH_NODE_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {

class BatchNode : public DatasetNode {
 public:
#ifdef ENABLE_PYTHON
  /// \brief Constructor #1, for Python API to create a BatchNode
  BatchNode(std::shared_ptr<DatasetNode> child, int32_t batch_size, bool drop_remainder, bool pad,
            const std::vector<std::string> &in_col_names, const std::vector<std::string> &out_col_names,
            const std::vector<std::string> &col_order, py::function batch_size_func, py::function batch_map_func,
            std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> pad_map);
#endif

  /// \brief Constructor #2 for C++ API to create a BatchNode
  BatchNode(std::shared_ptr<DatasetNode> child, int32_t batch_size, bool drop_remainder);

  /// \brief Destructor
  ~BatchNode() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams() override;

 private:
  int32_t batch_size_;
  bool drop_remainder_;
  bool pad_;
  std::vector<std::string> in_col_names_;
  std::vector<std::string> out_col_names_;
  std::vector<std::string> col_order_;
#ifdef ENABLE_PYTHON
  py::function batch_size_func_;
  py::function batch_map_func_;
#endif
  std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> pad_map_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_BATCH_NODE_H_
