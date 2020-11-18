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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_CACHE_DATASET_CACHE_IMPL_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_CACHE_DATASET_CACHE_IMPL_H_

#include <memory>
#include <string>
#include <optional>
#include <utility>
#include "minddata/dataset/engine/cache/cache_client.h"
#include "minddata/dataset/engine/datasetops/cache_op.h"
#include "minddata/dataset/engine/ir/cache/dataset_cache.h"

namespace mindspore::dataset {

/// DatasetCache is the IR of CacheClient
class DatasetCacheImpl : public DatasetCache {
 public:
  ///
  /// \brief Constructor
  /// \param id A user assigned session id for the current pipeline
  /// \param mem_sz Size of the memory set aside for the row caching. 0 for unlimited
  /// \param spill Spill to disk if out of memory
  /// \param hostname optional host name
  /// \param port optional port
  /// \param num_connections optional number of connections
  /// \param prefetch_sz optional prefetch size
  DatasetCacheImpl(session_id_type id, uint64_t mem_sz, bool spill, std::optional<std::string> hostname,
                   std::optional<int32_t> port, std::optional<int32_t> num_connections,
                   std::optional<int32_t> prefetch_sz)
      : session_id_(id),
        cache_mem_sz_(mem_sz),
        spill_(spill),
        hostname_(std::move(hostname)),
        port_(std::move(port)),
        num_connections_(std::move(num_connections)),
        prefetch_sz_(std::move(prefetch_sz)) {}

  /// Method to initialize the DatasetCache by creating an instance of a CacheClient
  /// \return Status Error code
  Status Build() override;

  Status CreateCacheOp(int32_t num_workers, std::shared_ptr<DatasetOp> *ds) override;

  Status ValidateParams() override { return Status::OK(); }

 private:
  std::shared_ptr<CacheClient> cache_client_;
  session_id_type session_id_;
  uint64_t cache_mem_sz_;
  bool spill_;
  std::optional<std::string> hostname_;
  std::optional<int32_t> port_;
  std::optional<int32_t> num_connections_;
  std::optional<int32_t> prefetch_sz_;
};
}  // namespace mindspore::dataset

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_CACHE_DATASET_CACHE_IMPL_H_
