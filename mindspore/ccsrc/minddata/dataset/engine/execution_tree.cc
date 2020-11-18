/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/execution_tree.h"
#include <iostream>
#include <string>
#include "minddata/dataset/engine/datasetops/dataset_op.h"
#include "minddata/dataset/engine/datasetops/shuffle_op.h"
#include "minddata/dataset/engine/datasetops/device_queue_op.h"
#include "minddata/dataset/util/task_manager.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/engine/opt/pre/removal_pass.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/opt/pre/cache_transform_pass.h"
#include "minddata/dataset/engine/opt/post/repeat_pass.h"
#include "minddata/dataset/engine/opt/pre/cache_error_pass.h"
#include "mindspore/ccsrc/minddata/dataset/engine/opt/optional/tensor_op_fusion_pass.h"
#endif
#include "minddata/dataset/engine/opt/pre/epoch_injection_pass.h"
#include "minddata/dataset/engine/perf/profiling.h"
#include "minddata/dataset/engine/perf/monitor.h"

namespace mindspore {
namespace dataset {
// Constructor
ExecutionTree::ExecutionTree() : id_count_(0) {
  tg_ = std::make_unique<TaskGroup>();
  tree_state_ = kDeTStateInit;
  prepare_flags_ = kDePrepNone;
  profiling_manager_ = std::make_unique<ProfilingManager>(this);
  optimize_ = common::GetEnv("OPTIMIZE") == "true" ? true : false;
}

// Destructor
ExecutionTree::~ExecutionTree() {
#ifdef ENABLE_TDTQUE
  DeviceQueueOp *op = dynamic_cast<DeviceQueueOp *>(root_.get());
  if (op != nullptr) {
    op->StopWaiting();
  }
#endif
  (void)tg_->ServiceStop();
}

// Associates a DatasetOp with this tree. This assigns a valid node id to the operator and
// provides it with a link to the tree. A node cannot form any relationships (parent/child) with
// other nodes unless they are associated with the same tree.
Status ExecutionTree::AssociateNode(const std::shared_ptr<DatasetOp> &op) {
  // If we are already a part of the tree, no-op
  if (op->tree_ == this) {
    return Status::OK();
  }
  if (tree_state_ != kDeTStateInit && tree_state_ != kDeTStateBuilding && tree_state_ != kDeTStatePrepare) {
    std::string err_msg =
      "Invalid tree state for adding a node. Current state: " + std::to_string(static_cast<int>(tree_state_)) +
      " Expected states: " + std::to_string(static_cast<int>(kDeTStateInit)) + " or " +
      std::to_string(static_cast<int>(kDeTStateBuilding)) + " or " + std::to_string(static_cast<int>(kDeTStatePrepare));
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // Enter the building state if we were not already there
  tree_state_ = kDeTStateBuilding;

  // Assign an id to the operator
  op->set_id(id_count_);
  id_count_++;

  // Assign our tree into the op so that each op has a link back to the tree
  op->set_tree(this);
  return Status::OK();
}

// Sets the root node of the tree
Status ExecutionTree::AssignRoot(const std::shared_ptr<DatasetOp> &op) {
  // Tree must be in building state before we can assign root to it
  if (tree_state_ != kDeTStateBuilding) {
    std::string err_msg =
      "Invalid tree state for assigning a root node. Current state: " + std::to_string(static_cast<int>(tree_state_)) +
      " Expected state: " + std::to_string(static_cast<int>(kDeTStateBuilding));
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // If they didn't already call AssociateNode for this node before calling AssignRoot,
  // then do so now.
  if (op->operator_id_ == DatasetOp::kInvalidOperatorId) {
    RETURN_IF_NOT_OK(this->AssociateNode(op));
  }

  // Then add it as the root.
  root_ = op;

  return Status::OK();
}

// A print method typically used for debugging
void ExecutionTree::Print(std::ostream &out, const std::shared_ptr<DatasetOp> &op) const {
  out << "Execution tree summary:\n"
      << "-----------------------\n";
  this->PrintNode(out, op == nullptr ? root_ : op, "", true, false);
  out << "\nExecution tree operator details:\n"
      << "--------------------------------\n";
  this->PrintNode(out, op == nullptr ? root_ : op, "", true, true);
}

// A helper functions for doing the recursive printing
void ExecutionTree::PrintNode(std::ostream &out, const std::shared_ptr<DatasetOp> &dataset_op, std::string indent,
                              bool last, bool detailed) const {
  // Decide which printer to use based on detailed arg.
  if (!detailed) {
    out << indent << "+- " << *dataset_op;
    indent += (last ? "    " : "|   ");
  } else {
    dataset_op->Print(out, detailed);
  }

  // Descend to children
  for (int32_t i = 0; i < dataset_op->child_.size(); ++i) {
    this->PrintNode(out, dataset_op->child_[i], indent, (i == (dataset_op->child_.size() - 1)), detailed);
  }
}

// Start the execution of the tree
Status ExecutionTree::Launch() {
  // Tree must be built and prepared before it can be launched!
  if (tree_state_ != kDeTStateReady) {
    std::string err_msg =
      "Invalid tree state for launching tree. Current state: " + std::to_string(static_cast<int>(tree_state_)) +
      " Expected state: " + std::to_string(static_cast<int>(kDeTStateReady));
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::ostringstream ss;
  ss << *this;

  // Profiling infrastructures need to be initialized before Op launching
  if (profiling_manager_->IsProfilingEnable()) {
    // Setup profiling manager
    RETURN_IF_NOT_OK(profiling_manager_->Initialize());
    // Launch Monitor Thread
    RETURN_IF_NOT_OK(profiling_manager_->LaunchMonitor());
  }

  MS_LOG(DEBUG) << "Printing the tree before launch tasks:\n" << ss.str();
  for (auto itr = this->begin(); itr != this->end(); ++itr) {
    // An inlined operator is one that has an output connector size of 0, and it does not
    // require a thread to execute.  Instead, the work of this operator is executed inlined
    // from the tree node directly above it (or in the case of a root node, it runs from within
    // the launching tree/user thread.  Do not exec any thread for an inlined op.
    itr->state_ = DatasetOp::OpState::kDeOpRunning;
    if (!itr->inlined()) {
      RETURN_IF_NOT_OK(tg_->CreateAsyncTask("Op launched, OperatorId:" + std::to_string(itr->id()), std::ref(*itr)));
      // Set the state of the Operator as running. This only matters in Leaf ops, CacheOp and TakeOp
    }
  }

  tree_state_ = kDeTStateExecuting;

  return Status::OK();
}

// A function that traverse the tree in postorder then save the results in nodes
void ExecutionTree::Iterator::PostOrderTraverse(const std::shared_ptr<DatasetOp> &node) {
  if (node == nullptr) {
    return;
  }
  for (int32_t i = 0; i < node->child_.size(); ++i) {
    PostOrderTraverse(node->child_[i]);
  }
  nodes_.push_back(node);
}

ExecutionTree::Iterator::Iterator(const std::shared_ptr<DatasetOp> &root) : ind_(0) {
  // post-order traverse the tree, if root is null, it return
  PostOrderTraverse(root);
  nodes_.emplace_back(nullptr);
}

// Given the number of workers, launches the worker entry function for each. Essentially a
// wrapper for the TaskGroup handling that is stored inside the execution tree.
Status ExecutionTree::LaunchWorkers(int32_t num_workers, std::function<Status(uint32_t)> func) {
  // Launch the workers
  for (int32_t i = 0; i < num_workers; ++i) {
    RETURN_IF_NOT_OK(tg_->CreateAsyncTask("Parallel Op Worker", std::bind(func, i)));
  }
  return Status::OK();
}

// The driver of the prepare phase of the execution tree.
// Prepare phase consists of three sub phases
//
// 1. PrepareTreePreAction()
//    Compulsory transformation/action pre optimization.
//    For example, CacheOp Insertion
//
// 2. Optimize()
//    Optimization transformation/action, optional
//    For example, MapOp Fusion
//
// 3. PrepareTreePostAction()
//    Compulsory transformation/action post optimization.
//    For example, repeatOp inlining
//
// @return Status - The error code return
Status ExecutionTree::Prepare(int32_t num_epochs) {
  num_epochs_ = num_epochs;

  // Pre optimization compulsory transformation
  RETURN_IF_NOT_OK(this->PrepareTreePreAction());

  // If optional optimizations are enabled
  if (optimize_) {
    RETURN_IF_NOT_OK(this->Optimize());
  }
  // Post optimization compulsory transformation
  RETURN_IF_NOT_OK(this->PrepareTreePostAction());

  // Existing transformation implementation, will be removed later
  RETURN_IF_NOT_OK(this->PrepareDeprecated());
  return Status::OK();
}

Status ExecutionTree::PrepareTreePreAction() {
  bool modified = false;
  std::vector<std::unique_ptr<Pass>> pre_actions;
  // Construct pre actions
  MS_LOG(INFO) << "Running pre pass loops.";
#ifndef ENABLE_ANDROID
  pre_actions.push_back(std::make_unique<CacheErrorPass>());
#endif
  pre_actions.push_back(std::make_unique<EpochInjectionPass>());
  pre_actions.push_back(std::make_unique<RemovalPass>());
#ifndef ENABLE_ANDROID
  pre_actions.push_back(std::make_unique<CacheTransformPass>());
#endif
  // Apply pre action passes
  for (auto &pass : pre_actions) {
    RETURN_IF_NOT_OK(pass->Run(this, &modified));
  }
  MS_LOG(INFO) << "Pre passes complete.";
  return Status::OK();
}

Status ExecutionTree::PrepareTreePostAction() {
  // The tree is ready to be prepared.
  tree_state_ = kDeTStatePrepare;

  bool modified = false;
  std::vector<std::unique_ptr<Pass>> post_actions;
  // Construct pre actions
  MS_LOG(INFO) << "Running post pass loops.";
#ifndef ENABLE_ANDROID
  post_actions.push_back(std::make_unique<RepeatPass>());
#endif

  // Apply post action passes
  for (auto &pass : post_actions) {
    RETURN_IF_NOT_OK(pass->Run(this, &modified));
  }
  MS_LOG(INFO) << "Post passes complete.";

  return Status::OK();
}

Status ExecutionTree::Optimize() {
  // Vector of optimizations, currently only 1, add more as necessary
  std::vector<std::unique_ptr<NodePass>> optimizations;
#ifndef ENABLE_ANDROID
  optimizations.push_back(std::make_unique<TensorOpFusionPass>());
#endif
  // vector of flags for each optimization
  std::vector<bool> modified(optimizations.size(), false);
  for (auto i = 0; i < optimizations.size(); i++) {
    auto m = false;
    optimizations[i]->Run(this, &m);
    modified[i] = m;
  }
  return Status::OK();
}

// The driver of the prepare phase of the execution tree. The prepare phase will recursively
// walk the tree to perform modifications to the tree or specific nodes within the tree to get
// it ready for execution.
//
// This driver is deprecated.
Status ExecutionTree::PrepareDeprecated() {
  // Tree must be in pending prepare state before we can assign root to it
  if (tree_state_ != kDeTStatePrepare) {
    std::string err_msg =
      "Invalid tree state for preparing the tree. Current state: " + std::to_string(static_cast<int>(tree_state_)) +
      " Expected state: " + std::to_string(static_cast<int>(kDeTStatePrepare));
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  if (root_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Please assign one operator as the root of this tree.");
  }

  // Start the recursive prepare
  RETURN_IF_NOT_OK(this->PrepareNode(root_));
  tree_state_ = kDeTStateReady;
  return Status::OK();
}

// Recursive function used during prepare phase to visit a node and drive any pre- and post-
// node actions during a tree walk.
Status ExecutionTree::PrepareNode(const std::shared_ptr<DatasetOp> &dataset_op) {
  // execute PreAction
  RETURN_IF_NOT_OK(dataset_op->PrepareNodePreAction());

  // Before going down into children, make any prepare flags updates based on this operator.
  uint32_t op_prep_flags = dataset_op->PrepareFlags();
  BitSet(&prepare_flags_, op_prep_flags);

  // Now, descend to children
  for (const auto &i : dataset_op->child_) {
    RETURN_IF_NOT_OK(this->PrepareNode(i));
  }

  // No more children, now we execute any prepare actions before going back up the
  // the tree on recursive function
  RETURN_IF_NOT_OK(dataset_op->PrepareNodePostAction());

  // Then clear the flags from this op now that we have prepared it.
  BitClear(&prepare_flags_, op_prep_flags);

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
