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

#ifndef DATASET_ENGINE_OPT_PASS_PRE_EPOCH_INJECTION_PASS_H_
#define DATASET_ENGINE_OPT_PASS_PRE_EPOCH_INJECTION_PASS_H_

#include <memory>
#include <vector>
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

class DatasetOp;

/// \class EpochInjectionPass epoch_injection_pass.h
/// \brief This is a pre pass that drives the injection of any nodes that could not be directly injected from the api
///     parsing.
class EpochInjectionPass : public TreePass {
  /// \class InjectionFinder
  /// \brief This is a nested node pass class who's job is to parse the tree and perform any identification logic for
  ///     operators that need to be injected.  It is run first by the main injection pass to find out what operators
  ///     it may need to inject.
  class InjectionFinder : public NodePass {
   public:
    /// \brief Constructor
    explicit InjectionFinder(std::shared_ptr<DatasetOp> node);

    /// \brief Destructor
    ~InjectionFinder() = default;

#ifndef ENABLE_ANDROID
    /// \brief Performs finder work for BuildVocabOp that has special rules about epoch control injection.
    /// \param[in] node The node being visited
    /// \param[inout] modified Indicator if the node was changed at all
    /// \return Status The error code return
    Status PreRunOnNode(std::shared_ptr<BuildVocabOp> node, bool *modified) override;

    /// \brief Performs finder work for BuildSentencePieceVocabOp that has special rules about epoch control injection.
    /// \param[in] node The node being visited
    /// \param[inout] modified Indicator if the node was changed at all
    /// \return Status The error code return
    Status PreRunOnNode(std::shared_ptr<BuildSentencePieceVocabOp> node, bool *modified) override;
#endif

    /// \brief Register the DeviceQueueOp for further action.
    /// \param[in] node The node being visited
    /// \param[inout] modified Indicator if the node was changed at all
    /// \return Status The error code return
    Status RunOnNode(std::shared_ptr<DeviceQueueOp> node, bool *modified) override;

    /// \brief Getter
    std::shared_ptr<DatasetOp> injection_point() { return injection_point_; }

   private:
    std::shared_ptr<DatasetOp> injection_point_;
  };

 public:
  /// \brief Constructor
  EpochInjectionPass();

  /// \brief Destructor
  ~EpochInjectionPass() = default;

  /// \brief Runs an injection pass to inject in operators needed at the pre pass stage
  /// \param[inout] tree The tree to operate on.
  /// \param[inout] Indicate of the tree was modified.
  /// \return Status The error code return
  Status RunOnTree(ExecutionTree *tree, bool *modified) override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_OPT_PASS_PRE_EPOCH_INJECTION_PASS_H_
