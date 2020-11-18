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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_RANDOM_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_RANDOM_NODE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {

class RandomNode : public DatasetNode {
 public:
  // Some constants to provide limits to random generation.
  static constexpr int32_t kMaxNumColumns = 4;
  static constexpr int32_t kMaxRank = 4;
  static constexpr int32_t kMaxDimValue = 32;

  /// \brief Constructor
  RandomNode(const int32_t &total_rows, std::shared_ptr<SchemaObj> schema, const std::vector<std::string> &columns_list,
             std::shared_ptr<DatasetCache> cache)
      : DatasetNode(std::move(cache)),
        total_rows_(total_rows),
        schema_path_(""),
        schema_(std::move(schema)),
        columns_list_(columns_list) {}

  /// \brief Constructor
  RandomNode(const int32_t &total_rows, std::string schema_path, const std::vector<std::string> &columns_list,
             std::shared_ptr<DatasetCache> cache)
      : DatasetNode(std::move(cache)),
        total_rows_(total_rows),
        schema_path_(schema_path),
        columns_list_(columns_list) {}

  /// \brief Destructor
  ~RandomNode() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams() override;

  /// \brief Get the shard id of node
  /// \return Status Status::OK() if get shard id successfully
  Status GetShardId(int32_t *shard_id) override;

 private:
  /// \brief A quick inline for producing a random number between (and including) min/max
  /// \param[in] min minimum number that can be generated.
  /// \param[in] max maximum number that can be generated.
  /// \return The generated random number
  int32_t GenRandomInt(int32_t min, int32_t max);

  int32_t total_rows_;
  std::string schema_path_;
  std::shared_ptr<SchemaObj> schema_;
  std::vector<std::string> columns_list_;
  std::shared_ptr<SamplerObj> sampler_;
  std::mt19937 rand_gen_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_RANDOM_NODE_H_
