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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_DATASET_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_DATASET_OP_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

#include "minddata/dataset/callback/callback_manager.h"
#include "minddata/dataset/core/constants.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

constexpr char kBarrierOp[] = "BarrierOp";
constexpr char kBatchOp[] = "BatchOp";
constexpr char kBucketBatchByLengthOp[] = "BucketBatchByLengthOp";
constexpr char kBuildSentencePieceVocabOp[] = "BuildSentencePieceVocabOp";
constexpr char kBuildVocabOp[] = "BuildVocabOp";
constexpr char kCacheBase[] = "CacheBase";
constexpr char kCacheLookupOp[] = "CacheLookupOp";
constexpr char kCacheMergeOp[] = "CacheMergeOp";
constexpr char kCacheOp[] = "CacheOp";
constexpr char kConcatOp[] = "ConcatOp";
constexpr char kDatasetOp[] = "DatasetOp";
constexpr char kDeviceQueueOp[] = "DeviceQueueOp";
constexpr char kEpochCtrlOp[] = "EpochCtrlOp";
constexpr char kFilterOp[] = "FilterOp";
constexpr char kMapOp[] = "MapOp";
constexpr char kParallelOp[] = "ParallelOp";
constexpr char kPipelineOp[] = "PipelineOp";
constexpr char kProjectOp[] = "ProjectOp";
constexpr char kRenameOp[] = "RenameOp";
constexpr char kRepeatOp[] = "RepeatOp";
constexpr char kShuffleOp[] = "ShuffleOp";
constexpr char kSkipOp[] = "SkipOp";
constexpr char kTakeOp[] = "TakeOp";
constexpr char kZipOp[] = "ZipOp";

// Forward declare
class ExecutionTree;

class DataBuffer;

class NodePass;

class SamplerRT;

/// \brief The base class DatasetOp is the main tree node.  It is an abstract class, so
/// the actual implementation of the operators will be derived from here.
class DatasetOp : public std::enable_shared_from_this<DatasetOp> {
  // Allow execution tree to access internal members
  friend class ExecutionTree;

 public:
  static constexpr int32_t kInvalidOperatorId = -1;
  static constexpr int32_t kInfiniteRepeat = -1;

  // Flags that control operator runtime behaviours
  enum OpState { kDeOpRunning = 0, kDeOpIdle = 1, kDeOpTerminated };

  /// Constructor
  /// \param op_connector_size - The size for the output connector of this operator.
  /// \param sampler - The sampler for the op
  explicit DatasetOp(int32_t op_connector_size, std::shared_ptr<SamplerRT> sampler);

  /// Destructor
  virtual ~DatasetOp() { tree_ = nullptr; }

  /// Adds a operator to become our child.
  /// \param child - shared pointer to the child to add.
  Status AddChild(std::shared_ptr<DatasetOp> child);

  /// Remove a operator from our children.
  /// \param child - shared pointer to the child to remove.
  Status RemoveChild(std::shared_ptr<DatasetOp> child);

  /// \brief Removes this node from the tree and connects it's parent/child together
  /// \return Status eerror code returned
  Status Remove();

  // Removes child operator in this operator.
  Status RemoveChildren();

  /// \brief Getter function to get a shared pointer to our child
  /// \param[in] child_index An operator can have n children. Indicates which child to return.
  /// \return The shared pointer to the child.  If there are no children, it returns null regardless of the given index
  std::shared_ptr<DatasetOp> child(int32_t child_index) const;

  /// \brief Getter function to get the pointer to our parent
  ///     If there are no parents, it returns null regardless of the given index
  /// \param[in] parent_index An operator can have n parents. Indicates which parent to return.
  void Parent(DatasetOp **parent, int32_t parent_index) const;

  // Getter function to get all of our children.
  std::vector<std::shared_ptr<DatasetOp>> children() const;

  // Getter function to get all of our parents.
  std::vector<DatasetOp *> parents() const;

  // Inserts a operator as the parent current op.
  // Inserted op will become the sole parent of the current op.
  // The existing parent of the current op will be transferred to the inserted op.
  Status InsertAsParent(std::shared_ptr<DatasetOp> to_add);

  /// \brief Creates the connector within this operator
  /// \param num_producers - number of threads that write into this connector
  /// \param num_consumers - number of threads that read from this connector
  void CreateConnector(int32_t num_producers, int32_t num_consumers);

  /// \brief A print method typically used for debugging
  /// \param out - The output stream to write output to
  /// \param show_all - A bool to control if you want to show all info or just a summary
  virtual void Print(std::ostream &out, bool show_all) const;

  /// \brief << Stream output operator overload
  /// \notes This allows you to write the debug print info using stream operators
  /// \param out - reference to the output stream being overloaded
  /// \param dO - reference to the DatasetOp to display
  /// \return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const DatasetOp &dO) {
    dO.Print(out, false);
    return out;
  }

  /// \brief Class functor operator ().
  /// DatasetOps operate by launching a thread (see ExecutionTree).
  /// This pure virtual version makes the requirement that derived classes must provide a functor
  /// that will execute their main runtime loop code.
  /// \return Status - The error code return
  virtual Status operator()() = 0;

  /// \brief Gets the next buffer from the given child
  /// \notes See GetNextInput for similar function that has built-in message handling
  /// \param p_buffer - The shared pointer for the fetched buffer to return (by reference)
  /// \param worker_id - The worker id
  /// \return Status - The error code return
  virtual Status GetNextBuffer(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id) {
    return GetNextBuffer(p_buffer, worker_id, false);
  }

  /// \brief Gets the next buffer from the given child
  /// \notes See GetNextInput for similar function that has built-in message handling
  /// \param p_buffer - The shared pointer for the fetched buffer to return (by reference)
  /// \return Status - The error code return
  virtual Status GetNextBuffer(std::unique_ptr<DataBuffer> *p_buffer) { return GetNextBuffer(p_buffer, 0, false); }

  /// \brief Gets the next buffer from the given child
  /// \notes See GetNextInput for similar function that has built-in message handling
  /// \param p_buffer - The shared pointer for the fetched buffer to return (by reference)
  /// \param worker_id - The worker id
  /// \param retry_if_eoe Set this flag to true to allow calling pop() again after the first pop() returns EOE.
  /// \return Status - The error code return
  virtual Status GetNextBuffer(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id, bool retry_if_eoe);

  /// \brief Gets the next buffer from the given child .  This function also has built-in eoe and eof
  /// message handling so that child classes don't have to manually code pass-through logic when
  /// those messages are received.
  /// \param p_buffer - The shared pointer for the fetched buffer to return (by reference)
  /// \param worker_id - The worker id
  /// \return Status - The error code return
  Status GetNextInput(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id = 0, int32_t child_index = 0);

  /// \brief Gets the dataset size
  /// \return Status - The status code return
  virtual Status GetDatasetSize(int64_t *dataset_size);

  /// \brief Gets the batch size
  /// \return Status - The status code return
  virtual int64_t GetTreeBatchSize();

  /// \brief Gets the repeat count
  /// \return Status - The status code return
  virtual int64_t GetTreeRepeatCount();

  /// \brief Gets the number of classes
  /// \return Status - The status code return
  virtual Status GetNumClasses(int64_t *num_classes);

  /// \brief Gets the class indexing
  /// \return Status - The status code return
  virtual Status GetClassIndexing(std::vector<std::pair<std::string, std::vector<int32_t>>> *output_class_indexing);

  /// \brief Performs handling for when an eoe message is received.
  /// The base class implementation simply flows the eoe message to output. Derived classes
  /// may override if they need to perform special eoe handling.
  /// \param worker_id - The worker id
  /// \return Status - The error code return
  virtual Status EoeReceived(int32_t worker_id);

  /// \brief Performs handling for when an eof message is received.
  /// The base class implementation simply flows the eof message to output. Derived classes
  /// may override if they need to perform special eof handling.
  /// \param worker_id - The worker id
  /// \return Status - The error code return
  virtual Status EofReceived(int32_t worker_id);

  /// \brief Derived classes may implement the reset function if the operator is stateful and needs
  /// specific reset handling that is not contained in this common code version of the reset
  /// \return Status - The error code return
  virtual Status Reset();

  /// \brief During tree prepare phase, operators may have specific pre-operations to perform depending on
  /// their role.
  /// \notes Derived versions of this function should always call it's superclass version first
  /// before providing their own implementations.
  virtual Status PrepareNodePreAction();

  /// \brief During tree prepare phase, operators may have specific post-operations to perform depending on
  /// their role.
  /// \notes Derived versions of this function should always call it's superclass version first
  /// before providing their own implementations.
  virtual Status PrepareNodePostAction();

  /// \brief Getter function
  /// \return The operator id
  int32_t id() const { return operator_id_; }

  /// \brief Getter function
  /// \return The prepare flags
  virtual uint32_t PrepareFlags() const;

  /// \brief Getter function
  /// \return The number of workers in this op
  virtual int32_t num_workers() const = 0;

  /// \brief Getter function
  /// \return The number of threads consuming from previous op.
  virtual int32_t num_consumers() const = 0;

  /// \brief Getter function
  /// \return The number of threads producing to the output connector.
  virtual int32_t num_producers() const = 0;

  /// \brief Getter function
  /// \return T/F if this is an inlined operator
  bool inlined() const { return (oc_queue_size_ == 0); }

  /// \brief Setter function, set the number of total repeats for the operator
  void set_total_repeats(int32_t total_repeats) { op_total_repeats_ = total_repeats; }

  /// \brief Setter function, set the number of repeats per epoch for the operator
  void set_num_repeats_per_epoch(int32_t num_repeats_per_epoch) { op_num_repeats_per_epoch_ = num_repeats_per_epoch; }

  /// \brief Getter function
  /// \return The number of required repeats for the operator
  int32_t op_total_repeats() { return op_total_repeats_; }

  /// \brief Getter function
  /// \return The number of required epochs for the operator
  int32_t op_total_epochs() { return op_total_repeats_ / op_num_repeats_per_epoch_; }

  /// \brief Getter function
  /// \return The number of repeats per epoch for the operator
  int32_t op_num_repeats_per_epoch() { return op_num_repeats_per_epoch_; }

  /// \brief Register the internal worker connectors. No op unless it is a parallel op
  /// \return Status
  virtual Status RegisterWorkerConnectors() { return Status::OK(); }

  /// \brief Getter for the column name mapping
  /// \return The returned map
  std::unordered_map<std::string, int32_t> column_name_id_map() const { return column_name_id_map_; }

  /// \brief Checks if the column name map has been set up yet for this op
  /// \return - T/F if the operator has the map set up
  bool HasColumnNameMap() const { return (column_name_id_map_.empty()); }

  /// \brief gives a string output for the column map for handy debug printing
  /// \return - the column name map as a string
  std::string ColumnNameMapAsString() const;

  /// \brief Getter function
  /// \return connector size of current op
  int32_t ConnectorSize() const {
    if (!inlined()) {
      return out_connector_->size();
    }
    // Return child connector size for inlined op
    return ChildOpConnectorSize();
  }

  /// \brief Counting number of buffer sent out by a connector
  int64_t ConnectorOutBufferCount() const {
    return out_connector_ == nullptr ? int64_t(-1) : static_cast<int64_t>(out_connector_->out_buffers_count());
  }

  /// \brief Getter function
  /// \return connector size of current op
  int32_t ConnectorCapacity() const {
    if (!inlined()) {
      return out_connector_->capacity();
    }
    // Return child connector capacity for inlined op
    return ChildOpConnectorCapacity();
  }

  /// \brief Getter function
  /// \return connector size of child op
  int32_t ChildOpConnectorSize(int32_t child_index = 0) const { return child_[child_index]->ConnectorSize(); }

  /// \brief Getter function
  /// \return connector capacity of child op
  int32_t ChildOpConnectorCapacity(int32_t child_index = 0) const { return child_[child_index]->ConnectorCapacity(); }

  /// \brief Children Getter
  /// \return Vector of Children
  std::vector<std::shared_ptr<DatasetOp>> Children() const { return child_; }

  /// \brief Base method for NodePass pre-visit.  A tree walk consists of walking down the tree and also walking back up
  ///     in a depth-first order.  PreAccept is the node visit on the way down, whereas the regular Accept is the main
  ///     visit on the way back up the tree during a post-order traversal. Subclass needs to override this if it
  ///     requires special node visit access. Check "dataset/engine/opt/pass.h" for more details.
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  virtual Status PreAccept(NodePass *p, bool *modified);

  /// \brief Base method for NodePass visit. Subclass needs to override this if it requires special node visit access.
  ///     Check "dataset/engine/opt/pass.h" for more details.
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  virtual Status Accept(NodePass *p, bool *modified);

  /// Op name getter
  /// \return Name of the current Op
  virtual std::string Name() const = 0;

  /// Execution Tree getter
  /// \return Pointer to the ExecutionTree the current op belongs to, no ownership
  ExecutionTree *Tree() { return tree_; }

  /// Getter for the sampler
  /// \return Shared pointer to the sampler (may return nullptr)
  std::shared_ptr<SamplerRT> sampler() { return sampler_; }

  /// \brief Getter for the sampler, and it also removes the sampler from the op
  /// \param[out] sampler A pointer to the output sampler that was removed
  /// \return Status error code
  Status FetchRemoveSampler(std::shared_ptr<SamplerRT> *sampler);

#ifndef ENABLE_ANDROID
  // Computes a CRC value for the operator
  static uint32_t GenerateCRC(const std::shared_ptr<DatasetOp> &op);
#endif

  /// \brief A helper templated function for casting "this" pointer to shared_ptr<derived>
  ///     Similar to shared_from_this, except this one will give you the derived class as shared_ptr
  /// \return A shared_ptr casted to the derived class
  template <typename Derived>
  std::shared_ptr<Derived> shared_from_base() {
    return std::static_pointer_cast<Derived>(shared_from_this());
  }

  /// \brief Setter for the sampler.  Allows you to overwrite a previous sampler with a new one.
  void SetSampler(std::shared_ptr<SamplerRT> sampler) { sampler_ = sampler; }

  /// \brief Checks if this is a leaf node (0 children)
  /// \return boolean returns true if it's a leaf
  bool IsLeaf() { return (child_.empty()); }

  /// Checks if an operator has reached its last iteration
  /// \return boolean returns true if it's last iteration
  bool IsLastIteration() { return op_total_repeats_ == op_current_repeats_ + 1; }

  /// This function is only intended to be called by CallbackManager within the master thread of ParallelOp
  /// The expected behavior is this, when this function is invoked, this function will block until all the workers
  /// have finished their remaining work and go to sleep. Since all ParallelOps use a QueueList to sync with master.
  /// They would automatically wait on the QueueList when they are done.
  /// \return Status
  virtual Status WaitForWorkers() { return Status::OK(); }

  /// \brief Add callback to DatasetOp, only MapOp supports Callback at the moment
  void AddCallbacks(std::vector<std::shared_ptr<DSCallback>> callbacks) { callback_manager_.AddCallbacks(callbacks); }

 protected:
  /// \brief Removes a parent operator from this operator
  /// \notes External callers do not have access to this function
  /// \param[in] parent The parent node to remove
  void RemoveParent(const DatasetOp *parent);

  /// \brief Adds a parent operator to this operator
  /// \notes External callers do not have access to this function
  /// \param[in] parent The parent node to add
  void AddParent(DatasetOp *parent);

  /// Compute the current op's column map using its child's column map.
  /// Get called during the tree post-prepare phase in PrepareNodePostAction.
  /// This base implementation just inherits the map from child 0, and can only be used if the number of children is 1.
  /// Operations changing the column map it inherits from the child must overwrite this function.
  /// \return - Status
  virtual Status ComputeColMap();

  /// Increase op_current_repeats_ by 1 when one repeat finished.
  /// If this repeat happen to be the last repeat in the current epoch, also increase op_current_epochs_ by 1.
  void UpdateRepeatAndEpochCounter();

  std::vector<std::shared_ptr<DatasetOp>> child_;                // Child nodes
  std::vector<DatasetOp *> parent_;                              // Parent nodes. No ownership
  std::shared_ptr<SamplerRT> sampler_;                           // Some leaf ops might have a sampler
  int32_t oc_queue_size_;                                        // Capacity for each out_connector_
  int32_t operator_id_;                                          // Generated id for the node
  ExecutionTree *tree_;                                          // Back pointer to our tree.
  OpState state_;                                                // The state of the operator, Running, Idle, Terminated
  int32_t op_total_repeats_;                                     // Required number of repeats for the operator
  int32_t op_num_repeats_per_epoch_;                             // Total number of repeats per epoch for the operator
  int32_t op_current_repeats_;                                   // Current number of repeats the operator has handled
  int32_t op_current_epochs_;                                    // Current number of epochs the operator has handled
  std::unique_ptr<DbConnector> out_connector_;                   // Output Connector
  std::unordered_map<std::string, int32_t> column_name_id_map_;  // Mapping between col index and col name
  std::mutex column_name_map_mutex_;                             // For protecting shared access to the column map
  CallbackManager callback_manager_;                             // Manages callbacks associated with a DatasetOp
  int64_t dataset_size_;                                         // Size of the dataset
  int64_t num_classes_;                                          // Number of classes

 private:
  /// Sets the operator id.
  /// \notes No public interface.  Only the class itself, or it's friend the execution tree can set
  /// this
  /// \param op_id - the Id value to set into the operator
  void set_id(int32_t op_id) { operator_id_ = op_id; }

  /// Sets the tree into the op so that the operator has a back pointer to the tree.
  /// \param tree - the tree to assign to the op.
  void set_tree(ExecutionTree *tree) { tree_ = tree; }
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_DATASET_OP_H_
