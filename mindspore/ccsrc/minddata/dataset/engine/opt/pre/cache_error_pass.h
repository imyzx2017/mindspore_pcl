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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_CACHE_ERROR_PASS_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_CACHE_ERROR_PASS_

#include <memory>
#include <stack>
#include <utility>
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

/// \class CacheErrorPass cache_error_pass.h
/// \brief This is a NodePass who's job is to catch invalid tree configurations related to cache and generate failures.
class CacheErrorPass : public NodePass {
 public:
  /// \brief Constructor
  CacheErrorPass();

  /// \brief Destructor
  ~CacheErrorPass() = default;

  /// \brief Identifies the subtree below this node as being cached
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status PreRunOnNode(std::shared_ptr<CacheOp> node, bool *modified) override;

  /// \brief Returns an error if ZipOp exists under a cache
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status PreRunOnNode(std::shared_ptr<ZipOp> node, bool *modified) override;

  /// \brief Returns an error if MapOp with non-deterministic TensorOps exists under a cache
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status PreRunOnNode(std::shared_ptr<MapOp> node, bool *modified) override;

  /// \brief Returns an error if ConcatOp exists under a cache
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status PreRunOnNode(std::shared_ptr<ConcatOp> node, bool *modified) override;

#ifdef ENABLE_PYTHON
  /// \brief Returns an error if FilterOp exists under a cache
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status PreRunOnNode(std::shared_ptr<FilterOp> node, bool *modified) override;
#endif

  /// \brief Identifies the leaf dataset as being mappable
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<ImageFolderOp> node, bool *modified) override;

  /// \brief Identifies the leaf dataset as being mappable
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<AlbumOp> node, bool *modified) override;

  /// \brief Identifies the leaf dataset as being mappable
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<MnistOp> node, bool *modified) override;

  /// \brief Identifies the leaf dataset as being mappable
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<CifarOp> node, bool *modified) override;

  /// \brief Identifies the leaf dataset as being mappable
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<CocoOp> node, bool *modified) override;

  /// \brief Identifies the leaf dataset as being mappable
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<CelebAOp> node, bool *modified) override;

  /// \brief Identifies the leaf dataset as being mappable
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<ManifestOp> node, bool *modified) override;

  /// \brief Identifies the leaf dataset as being mappable
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<VOCOp> node, bool *modified) override;

  /// \brief Identifies the leaf dataset as being mappable
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<MindRecordOp> node, bool *modified) override;

  /// \brief Identifies the leaf dataset as being mappable
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<GeneratorOp> node, bool *modified) override;

  /// \brief Identifies the subtree above this node as not being cached
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<CacheOp> node, bool *modified) override;

  /// \brief Identifies and block repeat under cache scenarios
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<RepeatOp> node, bool *modified) override;

 private:
  bool is_cached_;
  bool is_mappable_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_POST_CACHE_ERROR_PASS_
