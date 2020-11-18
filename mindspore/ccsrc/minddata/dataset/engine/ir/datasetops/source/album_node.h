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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_ALBUM_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_ALBUM_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {

class AlbumNode : public DatasetNode {
 public:
  /// \brief Constructor
  AlbumNode(const std::string &dataset_dir, const std::string &data_schema,
            const std::vector<std::string> &column_names, bool decode, const std::shared_ptr<SamplerObj> &sampler,
            const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor
  ~AlbumNode() = default;

  /// \brief a base class override function to create a runtime dataset op object from this class
  /// \return shared pointer to the newly created DatasetOp
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams() override;

  /// \brief Get the shard id of node
  /// \return Status Status::OK() if get shard id successfully
  Status GetShardId(int32_t *shard_id) override;

 private:
  std::string dataset_dir_;
  std::string schema_path_;
  std::vector<std::string> column_names_;
  bool decode_;
  std::shared_ptr<SamplerObj> sampler_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_ALBUM_NODE_H_
