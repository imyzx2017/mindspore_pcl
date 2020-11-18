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

#include <memory>
#include "minddata/dataset/engine/datasetops/cache_op.h"
#include "minddata/dataset/engine/datasetops/zip_op.h"
#include "minddata/dataset/engine/datasetops/map_op/map_op.h"
#include "minddata/dataset/engine/opt/pre/cache_error_pass.h"

namespace mindspore {
namespace dataset {

// Constructor
CacheErrorPass::CacheErrorPass() : is_cached_(false), is_mappable_(false) {}

// Identifies the subtree below this node as being cached
Status CacheErrorPass::PreRunOnNode(std::shared_ptr<CacheOp> node, bool *modified) {
  // Turn on the flag that we're under a merge op
  is_cached_ = true;
  return Status::OK();
}

// Returns an error if ZipOp exists under a cache
Status CacheErrorPass::PreRunOnNode(std::shared_ptr<ZipOp> node, bool *modified) {
  if (is_cached_) {
    RETURN_STATUS_UNEXPECTED("ZipOp is currently not supported as a descendant operator under a cache.");
  }

  return Status::OK();
}

// Returns an error if MapOp with non-deterministic TensorOps exists under a cache
Status CacheErrorPass::PreRunOnNode(std::shared_ptr<MapOp> node, bool *modified) {
  if (is_cached_) {
    auto tfuncs = node->TFuncs();
    for (size_t i = 0; i < tfuncs.size(); i++) {
      if (!tfuncs[i]->Deterministic()) {
        RETURN_STATUS_UNEXPECTED(
          "MapOp with non-deterministic TensorOps is currently not supported as a descendant of cache.");
      }
    }
  }
  return Status::OK();
}

// Returns an error if ConcatOp exists under a cache
Status CacheErrorPass::PreRunOnNode(std::shared_ptr<ConcatOp> node, bool *modified) {
  if (is_cached_) {
    RETURN_STATUS_UNEXPECTED("ConcatOp is currently not supported as a descendant operator under a cache.");
  }

  return Status::OK();
}

#ifdef ENABLE_PYTHON
// Returns an error if FilterOp exists under a cache
Status CacheErrorPass::PreRunOnNode(std::shared_ptr<FilterOp> node, bool *modified) {
  if (is_cached_) {
    RETURN_STATUS_UNEXPECTED("FilterOp is currently not supported as a descendant operator under a cache.");
  }

  return Status::OK();
}
#endif

Status CacheErrorPass::RunOnNode(std::shared_ptr<ImageFolderOp> node, bool *modified) {
  // Turn on the flag that this is a tree with mappable leaf dataset
  is_mappable_ = true;
  return Status::OK();
}

Status CacheErrorPass::RunOnNode(std::shared_ptr<AlbumOp> node, bool *modified) {
  // Turn on the flag that this is a tree with mappable leaf dataset
  is_mappable_ = true;
  return Status::OK();
}

Status CacheErrorPass::RunOnNode(std::shared_ptr<MnistOp> node, bool *modified) {
  // Turn on the flag that this is a tree with mappable leaf dataset
  is_mappable_ = true;
  return Status::OK();
}

Status CacheErrorPass::RunOnNode(std::shared_ptr<CifarOp> node, bool *modified) {
  // Turn on the flag that this is a tree with mappable leaf dataset
  is_mappable_ = true;
  return Status::OK();
}

Status CacheErrorPass::RunOnNode(std::shared_ptr<CocoOp> node, bool *modified) {
  // Turn on the flag that this is a tree with mappable leaf dataset
  is_mappable_ = true;
  return Status::OK();
}

Status CacheErrorPass::RunOnNode(std::shared_ptr<CelebAOp> node, bool *modified) {
  // Turn on the flag that this is a tree with mappable leaf dataset
  is_mappable_ = true;
  return Status::OK();
}

Status CacheErrorPass::RunOnNode(std::shared_ptr<ManifestOp> node, bool *modified) {
  // Turn on the flag that this is a tree with mappable leaf dataset
  is_mappable_ = true;
  return Status::OK();
}

Status CacheErrorPass::RunOnNode(std::shared_ptr<VOCOp> node, bool *modified) {
  // Turn on the flag that this is a tree with mappable leaf dataset
  is_mappable_ = true;
  return Status::OK();
}

Status CacheErrorPass::RunOnNode(std::shared_ptr<MindRecordOp> node, bool *modified) {
  // Turn on the flag that this is a tree with mappable leaf dataset
  is_mappable_ = true;
  return Status::OK();
}

Status CacheErrorPass::RunOnNode(std::shared_ptr<GeneratorOp> node, bool *modified) {
  // Turn on the flag that this is a tree with mappable leaf dataset
  is_mappable_ = true;
  return Status::OK();
}

Status CacheErrorPass::RunOnNode(std::shared_ptr<CacheOp> node, bool *modified) {
  // Turn off the flag that we're under a merge op
  is_cached_ = false;
  return Status::OK();
}

// Currently, returns an error if RepeatOp exists under a cache
// Because there is no operator in the cache hit stream to consume eoes, caching above repeat causes problem.
Status CacheErrorPass::RunOnNode(std::shared_ptr<RepeatOp> node, bool *modified) {
  if (is_cached_ && is_mappable_) {
    RETURN_STATUS_UNEXPECTED("Repeat is not supported as a descendant operator under a mappable cache.");
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
