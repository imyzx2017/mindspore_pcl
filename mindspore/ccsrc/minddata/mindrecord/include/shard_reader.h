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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_READER_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_READER_H_

#include <dirent.h>
#include <signal.h>
#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/prctl.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stack>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "minddata/mindrecord/include/shard_category.h"
#include "minddata/mindrecord/include/shard_column.h"
#include "minddata/mindrecord/include/shard_distributed_sample.h"
#include "minddata/mindrecord/include/shard_error.h"
#include "minddata/mindrecord/include/shard_index_generator.h"
#include "minddata/mindrecord/include/shard_operator.h"
#include "minddata/mindrecord/include/shard_pk_sample.h"
#include "minddata/mindrecord/include/shard_reader.h"
#include "minddata/mindrecord/include/shard_sample.h"
#include "minddata/mindrecord/include/shard_shuffle.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace mindrecord {
using ROW_GROUPS =
  std::tuple<MSRStatus, std::vector<std::vector<std::vector<uint64_t>>>, std::vector<std::vector<json>>>;
using ROW_GROUP_BRIEF =
  std::tuple<MSRStatus, std::string, int, uint64_t, std::vector<std::vector<uint64_t>>, std::vector<json>>;
using TASK_RETURN_CONTENT =
  std::pair<MSRStatus, std::pair<TaskType, std::vector<std::tuple<std::vector<uint8_t>, json>>>>;
const int kNumBatchInMap = 1000;  // iterator buffer size in row-reader mode

class ShardReader {
 public:
  ShardReader();

  virtual ~ShardReader();

  /// \brief open files and initialize reader, c++ API
  /// \param[in] file_paths the path of ONE file, any file in dataset is fine or file list
  /// \param[in] load_dataset load dataset from single file or not
  /// \param[in] n_consumer number of threads when reading
  /// \param[in] selected_columns column list to be populated
  /// \param[in] operators operators applied to data, operator type is shuffle, sample or category
  /// \return MSRStatus the status of MSRStatus
  MSRStatus Open(const std::vector<std::string> &file_paths, bool load_dataset, int n_consumer = 4,
                 const std::vector<std::string> &selected_columns = {},
                 const std::vector<std::shared_ptr<ShardOperator>> &operators = {}, const int num_padded = 0);

  /// \brief open files and initialize reader, python API
  /// \param[in] file_paths the path of ONE file, any file in dataset is fine or file list
  /// \param[in] load_dataset load dataset from single file or not
  /// \param[in] n_consumer number of threads when reading
  /// \param[in] selected_columns column list to be populated
  /// \param[in] operators operators applied to data, operator type is shuffle, sample or category
  /// \return MSRStatus the status of MSRStatus
  MSRStatus OpenPy(const std::vector<std::string> &file_paths, bool load_dataset, const int &n_consumer = 4,
                   const std::vector<std::string> &selected_columns = {},
                   const std::vector<std::shared_ptr<ShardOperator>> &operators = {});

  /// \brief close reader
  /// \return null
  void Close();

  /// \brief read the file, get schema meta,statistics and index, single-thread mode
  /// \return MSRStatus the status of MSRStatus
  MSRStatus Open();

  /// \brief read the file, get schema meta,statistics and index, multiple-thread mode
  /// \return MSRStatus the status of MSRStatus
  MSRStatus Open(int n_consumer);

  /// \brief launch threads to get batches
  /// \param[in] is_simple_reader trigger threads if false; do nothing if true
  /// \return MSRStatus the status of MSRStatus
  MSRStatus Launch(bool is_simple_reader = false);

  /// \brief aim to get the meta data
  /// \return the metadata
  std::shared_ptr<ShardHeader> GetShardHeader() const;

  /// \brief aim to get columns context
  /// \return the columns
  std::shared_ptr<ShardColumn> GetShardColumn() const;

  /// \brief get the number of shards
  /// \return # of shards
  int GetShardCount() const;

  /// \brief get the number of rows in database
  /// \param[in] file_paths the path of ONE file, any file in dataset is fine or file list
  /// \param[in] load_dataset load dataset from single file or not
  /// \param[in] op smart pointer refer to ShardCategory or ShardSample object
  /// \param[out] count # of rows
  /// \return MSRStatus the status of MSRStatus
  MSRStatus CountTotalRows(const std::vector<std::string> &file_paths, bool load_dataset,
                           const std::shared_ptr<ShardOperator> &op, int64_t *count, const int num_padded);

  /// \brief shuffle task with incremental seed
  /// \return void
  void ShuffleTask();

  /// \brief get the number of rows in database
  /// \return # of rows
  int GetNumRows() const;

  /// \brief Read the summary of row groups
  /// \return the tuple of 4 elements
  ///         1. Sharding ID
  ///         2. Row group ID
  ///         3. The row ID started in row group
  ///         4. # of rows in row group
  std::vector<std::tuple<int, int, int, uint64_t>> ReadRowGroupSummary();

  /// \brief Read 1 row group data, excluding images
  /// \param[in] groupID row group ID
  /// \param[in] shard_id sharding ID
  /// \param[in] columns multi-columns retrieved
  /// \return the tuple of 5 elements
  ///         1. file name where row group is located
  ///         2. Actual row group size
  ///         3. Offset address of row group in file
  ///         4. The list of image offset in page [startOffset, endOffset)
  ///         5. The list of columns data
  ROW_GROUP_BRIEF ReadRowGroupBrief(int group_id, int shard_id,
                                    const std::vector<std::string> &columns = std::vector<std::string>());

  /// \brief Read 1 row group data, excluding images, following an index field criteria
  /// \param[in] groupID row group ID
  /// \param[in] shard_id sharding ID
  /// \param[in] column-value pair of criteria to fulfill
  /// \param[in] columns multi-columns retrieved
  /// \return the tuple of 5 elements
  ///         1. file name where row group is located
  ///         2. Actual row group size
  ///         3. Offset address of row group in file
  ///         4. The list of image offset in page [startOffset, endOffset)
  ///         5. The list of columns data
  ROW_GROUP_BRIEF ReadRowGroupCriteria(int group_id, int shard_id, const std::pair<std::string, std::string> &criteria,
                                       const std::vector<std::string> &columns = std::vector<std::string>());

  /// \brief return a batch, given that one is ready
  /// \return a batch of images and image data
  std::vector<std::tuple<std::vector<uint8_t>, json>> GetNext();

  /// \brief return a row by id
  /// \return a batch of images and image data
  std::pair<TaskType, std::vector<std::tuple<std::vector<uint8_t>, json>>> GetNextById(const int64_t &task_id,
                                                                                       const int32_t &consumer_id);

  /// \brief return a batch, given that one is ready, python API
  /// \return a batch of images and image data
  std::vector<std::tuple<std::vector<std::vector<uint8_t>>, pybind11::object>> GetNextPy();

  /// \brief  get blob filed list
  /// \return blob field list
  std::pair<ShardType, std::vector<std::string>> GetBlobFields();

  /// \brief reset reader
  /// \return null
  void Reset();

  /// \brief set flag of all-in-index
  /// \return null
  void SetAllInIndex(bool all_in_index) { all_in_index_ = all_in_index; }

  /// \brief get all classes
  MSRStatus GetAllClasses(const std::string &category_field, std::set<std::string> &categories);

  /// \brief get the size of blob data
  MSRStatus GetTotalBlobSize(int64_t *total_blob_size);

 protected:
  /// \brief sqlite call back function
  static int SelectCallback(void *p_data, int num_fields, char **p_fields, char **p_col_names);

 private:
  /// \brief wrap up labels to json format
  MSRStatus ConvertLabelToJson(const std::vector<std::vector<std::string>> &labels, std::shared_ptr<std::fstream> fs,
                               std::vector<std::vector<std::vector<uint64_t>>> &offsets, int shard_id,
                               const std::vector<std::string> &columns, std::vector<std::vector<json>> &column_values);

  /// \brief read all rows for specified columns
  ROW_GROUPS ReadAllRowGroup(std::vector<std::string> &columns);

  /// \brief read all rows in one shard
  MSRStatus ReadAllRowsInShard(int shard_id, const std::string &sql, const std::vector<std::string> &columns,
                               std::vector<std::vector<std::vector<uint64_t>>> &offsets,
                               std::vector<std::vector<json>> &column_values);

  /// \brief initialize reader
  MSRStatus Init(const std::vector<std::string> &file_paths, bool load_dataset);

  /// \brief validate column list
  MSRStatus CheckColumnList(const std::vector<std::string> &selected_columns);

  /// \brief populate one row by task list in row-reader mode
  MSRStatus ConsumerByRow(int consumer_id);

  /// \brief get offset address of images within page
  std::vector<std::vector<uint64_t>> GetImageOffset(int group_id, int shard_id,
                                                    const std::pair<std::string, std::string> &criteria = {"", ""});

  /// \brief execute sqlite query with prepare statement
  MSRStatus QueryWithCriteria(sqlite3 *db, string &sql, string criteria, std::vector<std::vector<std::string>> &labels);

  /// \brief get column values
  std::pair<MSRStatus, std::vector<json>> GetLabels(int group_id, int shard_id, const std::vector<std::string> &columns,
                                                    const std::pair<std::string, std::string> &criteria = {"", ""});

  /// \brief get column values from raw data page
  std::pair<MSRStatus, std::vector<json>> GetLabelsFromPage(int group_id, int shard_id,
                                                            const std::vector<std::string> &columns,
                                                            const std::pair<std::string, std::string> &criteria = {"",
                                                                                                                   ""});

  /// \brief create category-applied task list
  MSRStatus CreateTasksByCategory(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
                                  const std::shared_ptr<ShardOperator> &op);

  /// \brief create task list in row-reader mode
  MSRStatus CreateTasksByRow(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
                             const std::vector<std::shared_ptr<ShardOperator>> &operators);

  /// \brief crate task list
  MSRStatus CreateTasks(const std::vector<std::tuple<int, int, int, uint64_t>> &row_group_summary,
                        const std::vector<std::shared_ptr<ShardOperator>> &operators);

  /// \brief check if all specified columns are in index table
  void CheckIfColumnInIndex(const std::vector<std::string> &columns);

  /// \brief open multiple file handle
  void FileStreamsOperator();

  /// \brief read one row by one task
  TASK_RETURN_CONTENT ConsumerOneTask(int task_id, uint32_t consumer_id);

  /// \brief get labels from binary file
  std::pair<MSRStatus, std::vector<json>> GetLabelsFromBinaryFile(
    int shard_id, const std::vector<std::string> &columns, const std::vector<std::vector<std::string>> &label_offsets);

  /// \brief get classes in one shard
  void GetClassesInShard(sqlite3 *db, int shard_id, const std::string sql, std::set<std::string> &categories);

  /// \brief get number of classes
  int64_t GetNumClasses(const std::string &category_field);

  /// \brief get meta of header
  std::pair<MSRStatus, std::vector<std::string>> GetMeta(const std::string &file_path, json &meta_data);

  /// \brief extract uncompressed data based on column list
  std::pair<MSRStatus, std::vector<std::vector<uint8_t>>> UnCompressBlob(const std::vector<uint8_t> &raw_blob_data);

 protected:
  uint64_t header_size_;                       // header size
  uint64_t page_size_;                         // page size
  int shard_count_;                            // number of shards
  std::shared_ptr<ShardHeader> shard_header_;  // shard header
  std::shared_ptr<ShardColumn> shard_column_;  // shard column

  std::vector<sqlite3 *> database_paths_;                                        // sqlite handle list
  std::vector<string> file_paths_;                                               // file paths
  std::vector<std::shared_ptr<std::fstream>> file_streams_;                      // single-file handle list
  std::vector<std::vector<std::shared_ptr<std::fstream>>> file_streams_random_;  // multiple-file handle list

 private:
  int n_consumer_;                                         // number of workers (threads)
  std::vector<std::string> selected_columns_;              // columns which will be read
  std::map<string, uint64_t> column_schema_id_;            // column-schema map
  std::vector<std::shared_ptr<ShardOperator>> operators_;  // data operators, including shuffle, sample and category
  ShardTask tasks_;                                        // shard task
  std::mutex shard_locker_;                                // locker of shard

  // flags
  bool all_in_index_ = true;  // if all columns are stored in index-table
  bool interrupt_ = false;    // reader interrupted

  int num_padded_;  // number of padding samples

  // Delivery/Iterator mode begin
  const std::string kThreadName = "THRD_ITER_";  // prefix of thread name
  std::vector<std::thread> thread_set_;          // thread list
  int num_rows_;                                 // number of rows
  int64_t total_blob_size_;                      // total size of blob data
  std::mutex mtx_delivery_;                      // locker for delivery
  std::condition_variable cv_delivery_;          // conditional variable for delivery
  std::condition_variable cv_iterator_;          // conditional variable for iterator
  std::atomic<int> task_id_;                     // task ID which is working
  std::atomic<int> deliver_id_;                  // delivery ID which is picked up by iterator
  // map of delivery
  std::unordered_map<int, std::shared_ptr<std::vector<std::tuple<std::vector<uint8_t>, json>>>> delivery_map_;
  // Delivery/Iterator mode end
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_READER_H_
