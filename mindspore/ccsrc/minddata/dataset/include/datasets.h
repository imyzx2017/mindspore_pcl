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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASETS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASETS_H_

#include <unistd.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/ir/cache/dataset_cache.h"

#include "minddata/dataset/core/constants.h"
#include "minddata/dataset/engine/consumers/tree_consumer.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/include/iterator.h"
#include "minddata/dataset/include/samplers.h"
#include "minddata/dataset/include/tensor.h"
#include "minddata/dataset/include/type_id.h"
#include "minddata/dataset/kernels/c_func_op.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/path.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/text/sentence_piece_vocab.h"
#include "minddata/dataset/text/vocab.h"
#endif

namespace mindspore {
namespace dataset {

class Tensor;
class TensorShape;
class TreeAdapter;
class TreeGetters;
#ifndef ENABLE_ANDROID
class Vocab;
#endif

class DatasetNode;

class Iterator;

class TensorOperation;
class SchemaObj;
class SamplerObj;
class CsvBase;

// Dataset classes (in alphabetical order)
class BatchDataset;
class MapDataset;
class ProjectDataset;
class ShuffleDataset;
#ifndef ENABLE_ANDROID
class BucketBatchByLengthDataset;
class FilterDataset;
class CSVDataset;
class TransferDataset;
class ConcatDataset;
class RenameDataset;
#endif

class RepeatDataset;

#ifndef ENABLE_ANDROID
class SkipDataset;
class TakeDataset;
class ZipDataset;
#endif

/// \class Dataset datasets.h
/// \brief A base class to represent a dataset in the data pipeline.
class Dataset : public std::enable_shared_from_this<Dataset> {
 public:
  // need friend class so they can access the children_ field
  friend class Iterator;
  friend class TransferNode;

  /// \brief Constructor
  Dataset();

  /// \brief Destructor
  ~Dataset() = default;

  /// \brief Gets the dataset size
  /// \return dataset size. If failed, return -1
  int64_t GetDatasetSize();

  /// \brief Gets the output type
  /// \return a vector of DataType. If failed, return an empty vector
  std::vector<DataType> GetOutputTypes();

  /// \brief Gets the output shape
  /// \return a vector of TensorShape. If failed, return an empty vector
  std::vector<TensorShape> GetOutputShapes();

  /// \brief Gets the batch size
  /// \return int64_t
  int64_t GetBatchSize();

  /// \brief Gets the repeat count
  /// \return int64_t
  int64_t GetRepeatCount();

  /// \brief Gets the number of classes
  /// \return number of classes. If failed, return -1
  int64_t GetNumClasses();

  /// \brief Gets the column names
  /// \return Names of the columns. If failed, return an empty vector
  std::vector<std::string> GetColumnNames();

  /// \brief Gets the class indexing
  /// \return a map of ClassIndexing. If failed, return an empty map
  std::vector<std::pair<std::string, std::vector<int32_t>>> GetClassIndexing();

  /// \brief Setter function for runtime number of workers
  /// \param[in] num_workers The number of threads in this operator
  /// \return Shared pointer to the original object
  std::shared_ptr<Dataset> SetNumWorkers(int32_t num_workers);

  /// \brief Function to create an Iterator over the Dataset pipeline
  /// \param[in] columns List of columns to be used to specify the order of columns
  /// \return Shared pointer to the Iterator
  std::shared_ptr<Iterator> CreateIterator(std::vector<std::string> columns = {});

#ifndef ENABLE_ANDROID
  /// \brief Function to transfer data through a device.
  /// \notes If device is Ascend, features of data will be transferred one by one. The limitation
  ///     of data transmission per time is 256M.
  /// \param[in] send_epoch_end Whether to send end of sequence to device or not (default=true).
  /// \return Returns true if no error encountered else false.
  bool DeviceQueue(bool send_epoch_end = true);

  /// \brief Function to create a Saver to save the dynamic data processed by the dataset pipeline
  /// \note Usage restrictions:
  ///     1. Supported dataset formats: 'mindrecord' only
  ///     2. To save the samples in order, set dataset's shuffle to false and num_files to 1.
  ///     3. Before calling the function, do not use batch operator, repeat operator or data augmentation operators
  ///        with random attribute in map operator.
  ///     4. Mindrecord does not support bool, uint64, multi-dimensional uint8(drop dimension) nor
  ///        multi-dimensional string.
  /// \param[in] file_name Path to dataset file
  /// \param[in] num_files Number of dataset files (default=1)
  /// \param[in] file_type Dataset format (default="mindrecord")
  /// \return Returns true if no error encountered else false
  bool Save(std::string dataset_path, int32_t num_files = 1, std::string dataset_type = "mindrecord");
#endif

  /// \brief Function to create a BatchDataset
  /// \notes Combines batch_size number of consecutive rows into batches
  /// \param[in] batch_size The number of rows each batch is created with
  /// \param[in] drop_remainder Determines whether or not to drop the last possibly incomplete
  ///     batch. If true, and if there are less than batch_size rows
  ///     available to make the last batch, then those rows will
  ///     be dropped and not propagated to the next node
  /// \return Shared pointer to the current BatchDataset
  std::shared_ptr<BatchDataset> Batch(int32_t batch_size, bool drop_remainder = false);

#ifndef ENABLE_ANDROID
  /// \brief Function to create a BucketBatchByLengthDataset
  /// \notes Bucket elements according to their lengths. Each bucket will be padded and batched when
  ///    they are full.
  /// \param[in] column_names Columns passed to element_length_function
  /// \param[in] bucket_boundaries A list consisting of the upper boundaries of the buckets.
  ///    Must be strictly increasing. If there are n boundaries, n+1 buckets are created: One bucket for
  ///    [0, bucket_boundaries[0]), one bucket for [bucket_boundaries[i], bucket_boundaries[i+1]) for each
  ///    0<i<n, and one bucket for [bucket_boundaries[n-1], inf).
  /// \param[in] bucket_batch_sizes A list consisting of the batch sizes for each bucket.
  ///    Must contain elements equal to the size of bucket_boundaries + 1.
  /// \param[in] element_length_function A function pointer that takes in TensorRow and outputs a TensorRow.
  ///    The output must contain a single tensor containing a single int32_t. If no value is provided,
  ///    then size of column_names must be 1, and the size of the first dimension of that column will be taken
  ///    as the length (default=nullptr)
  /// \param[in] pad_info Represents how to batch each column. The key corresponds to the column name, the value must
  ///    be a tuple of 2 elements.  The first element corresponds to the shape to pad to, and the second element
  ///    corresponds to the value to pad with. If a column is not specified, then that column will be padded to the
  ///    longest in the current batch, and 0 will be used as the padding value. Any unspecified dimensions will be
  ///    padded to the longest in the current batch, unless if pad_to_bucket_boundary is true. If no padding is
  ///    wanted, set pad_info to None (default=empty dictionary).
  /// \param[in] pad_to_bucket_boundary If true, will pad each unspecified dimension in pad_info to the
  ///    bucket_boundary minus 1. If there are any elements that fall into the last bucket,
  ///    an error will occur (default=false).
  /// \param[in] drop_remainder If true, will drop the last batch for each bucket if it is not a full batch
  ///    (default=false).
  /// \return Shared pointer to the current BucketBatchByLengthDataset
  std::shared_ptr<BucketBatchByLengthDataset> BucketBatchByLength(
    const std::vector<std::string> &column_names, const std::vector<int32_t> &bucket_boundaries,
    const std::vector<int32_t> &bucket_batch_sizes,
    std::function<TensorRow(TensorRow)> element_length_function = nullptr,
    const std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> &pad_info = {},
    bool pad_to_bucket_boundary = false, bool drop_remainder = false) {
    return std::make_shared<BucketBatchByLengthDataset>(shared_from_this(), column_names, bucket_boundaries,
                                                        bucket_batch_sizes, element_length_function, pad_info,
                                                        pad_to_bucket_boundary, drop_remainder);
  }

  /// \brief Function to create a SentencePieceVocab from source dataset
  /// \notes Build a SentencePieceVocab from a dataset.
  /// \param[in] col_names Column names to get words from. It can be a vector of column names
  /// \param[in] vocab_size Vocabulary size. The type is uint32
  /// \param[in] character_coverage Percentage of characters covered by the model, must be between
  ///     0.98 and 1.0 Good defaults are: 0.9995 for languages with rich character sets like
  ///     Japanese or Chinese character sets, and 1.0 for other languages with small character sets.
  /// \param[in] model_type Model type. Choose from unigram (default), bpe, char, or word.
  ///     The input sentence must be pretokenized when using word type.
  /// \param[in] params A vector contains more option parameters of sentencepiece library
  std::shared_ptr<SentencePieceVocab> BuildSentencePieceVocab(
    const std::vector<std::string> &col_names, uint32_t vocab_size, float character_coverage,
    SentencePieceModel model_type, const std::unordered_map<std::string, std::string> &params);

  /// \brief Function to create a Vocab from source dataset
  /// \notes Build a vocab from a dataset. This would collect all the unique words in a dataset and return a vocab
  ///    which contains top_k most frequent words (if top_k is specified)
  /// \param[in] columns Column names to get words from. It can be a vector of column names
  /// \param[in] freq_range A tuple of integers (min_frequency, max_frequency). Words within the frequency
  ///    range would be kept. 0 <= min_frequency <= max_frequency <= total_words. min_frequency/max_frequency
  ///    can be set to default, which corresponds to 0/total_words separately
  /// \param[in] top_k Number of words to be built into vocab. top_k most frequent words are
  ///    taken. The top_k is taken after freq_range. If not enough top_k, all words will be taken
  /// \param[in] special_tokens A list of strings, each one is a special token
  /// \param[in] special_first Whether special_tokens will be prepended/appended to vocab, If special_tokens
  ///    is specified and special_first is set to default, special_tokens will be prepended
  /// \return Shared pointer to the current Vocab
  std::shared_ptr<Vocab> BuildVocab(const std::vector<std::string> &columns = {},
                                    const std::pair<int64_t, int64_t> &freq_range = {0, kDeMaxFreq},
                                    int64_t top_k = kDeMaxTopk, const std::vector<std::string> &special_tokens = {},
                                    bool special_first = true);

  /// \brief Function to create a ConcatDataset
  /// \notes Concat the datasets in the input
  /// \param[in] datasets List of shared pointers to the dataset that should be concatenated together
  /// \return Shared pointer to the current ConcatDataset
  std::shared_ptr<ConcatDataset> Concat(const std::vector<std::shared_ptr<Dataset>> &datasets) {
    std::vector<std::shared_ptr<Dataset>> all_datasets = datasets;
    all_datasets.push_back(shared_from_this());
    return std::make_shared<ConcatDataset>(all_datasets);
  }

  /// \brief Function to filter dataset by predicate
  /// \notes If input_columns is not provided or empty, all columns will be used
  /// \param[in] predicate Function callable which returns a boolean value. If false then filter the element
  /// \param[in] input_columns List of names of the input columns to filter
  /// \return Shared pointer to the current FilterNode
  std::shared_ptr<FilterDataset> Filter(std::function<TensorRow(TensorRow)> predicate,
                                        std::vector<std::string> input_columns = {}) {
    return std::make_shared<FilterDataset>(shared_from_this(), predicate, input_columns);
  }
#endif

  /// \brief Function to create a MapDataset
  /// \notes Applies each operation in operations to this dataset
  /// \param[in] operations Vector of operations to be applied on the dataset. Operations are
  ///     applied in the order they appear in this list
  /// \param[in] input_columns Vector of the names of the columns that will be passed to the first
  ///     operation as input. The size of this list must match the number of
  ///     input columns expected by the first operator. The default input_columns
  ///     is the first column
  /// \param[in] output_columns Vector of names assigned to the columns outputted by the last operation
  ///     This parameter is mandatory if len(input_columns) != len(output_columns)
  ///     The size of this list must match the number of output columns of the
  ///     last operation. The default output_columns will have the same
  ///     name as the input columns, i.e., the columns will be replaced
  /// \param[in] project_columns A list of column names to project
  /// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
  ///     The cache feature is under development and is not recommended.
  /// \return Shared pointer to the current MapDataset
  std::shared_ptr<MapDataset> Map(std::vector<std::shared_ptr<TensorOperation>> operations,
                                  std::vector<std::string> input_columns = {},
                                  std::vector<std::string> output_columns = {},
                                  const std::vector<std::string> &project_columns = {},
                                  const std::shared_ptr<DatasetCache> &cache = nullptr,
                                  std::vector<std::shared_ptr<DSCallback>> callbacks = {}) {
    return std::make_shared<MapDataset>(shared_from_this(), operations, input_columns, output_columns, project_columns,
                                        cache, callbacks);
  }

  /// \brief Function to create a Project Dataset
  /// \notes Applies project to the dataset
  /// \param[in] columns The name of columns to project
  /// \return Shared pointer to the current Dataset
  std::shared_ptr<ProjectDataset> Project(const std::vector<std::string> &columns) {
    return std::make_shared<ProjectDataset>(shared_from_this(), columns);
  }

#ifndef ENABLE_ANDROID
  /// \brief Function to create a Rename Dataset
  /// \notes Renames the columns in the input dataset
  /// \param[in] input_columns List of the input columns to rename
  /// \param[in] output_columns List of the output columns
  /// \return Shared pointer to the current Dataset
  std::shared_ptr<RenameDataset> Rename(const std::vector<std::string> &input_columns,
                                        const std::vector<std::string> &output_columns) {
    return std::make_shared<RenameDataset>(shared_from_this(), input_columns, output_columns);
  }
#endif
  /// \brief Function to create a RepeatDataset
  /// \notes Repeats this dataset count times. Repeat indefinitely if count is -1
  /// \param[in] count Number of times the dataset should be repeated
  /// \return Shared pointer to the current Dataset
  /// \note Repeat will return shared pointer to `Dataset` instead of `RepeatDataset`
  ///     due to a limitation in the current implementation
  std::shared_ptr<RepeatDataset> Repeat(int32_t count = -1) {
    return std::make_shared<RepeatDataset>(shared_from_this(), count);
  }
#ifndef ENABLE_ANDROID
  /// \brief Function to create a Shuffle Dataset
  /// \notes Randomly shuffles the rows of this dataset
  /// \param[in] buffer_size The size of the buffer (must be larger than 1) for shuffling
  /// \return Shared pointer to the current ShuffleDataset
  std::shared_ptr<ShuffleDataset> Shuffle(int32_t buffer_size) {
    return std::make_shared<ShuffleDataset>(shared_from_this(), buffer_size);
  }

  /// \brief Function to create a SkipDataset
  /// \notes Skips count elements in this dataset.
  /// \param[in] count Number of elements the dataset to be skipped.
  /// \return Shared pointer to the current SkipDataset
  std::shared_ptr<SkipDataset> Skip(int32_t count) { return std::make_shared<SkipDataset>(shared_from_this(), count); }

  /// \brief Function to create a TakeDataset
  /// \notes Takes count elements in this dataset.
  /// \param[in] count Number of elements the dataset to be taken.
  /// \return Shared pointer to the current Dataset
  std::shared_ptr<TakeDataset> Take(int32_t count = -1) {
    return std::make_shared<TakeDataset>(shared_from_this(), count);
  }

  /// \brief Function to create a Zip Dataset
  /// \notes Applies zip to the dataset
  /// \param[in] datasets A list of shared pointers to the datasets that we want to zip
  /// \return Shared pointer to the current Dataset
  std::shared_ptr<ZipDataset> Zip(const std::vector<std::shared_ptr<Dataset>> &datasets) {
    std::vector<std::shared_ptr<Dataset>> all_datasets = datasets;
    all_datasets.push_back(shared_from_this());
    return std::make_shared<ZipDataset>(all_datasets);
  }
#endif

  std::shared_ptr<DatasetNode> IRNode() { return ir_node_; }

 protected:
  std::shared_ptr<TreeGetters> tree_getters_;
  std::shared_ptr<DatasetNode> ir_node_;
};

class SchemaObj {
 public:
  /// \brief Constructor
  explicit SchemaObj(const std::string &schema_file = "");

  /// \brief Destructor
  ~SchemaObj() = default;

  /// \brief SchemaObj init function
  /// \return bool true if schema init success
  bool init();

  /// \brief Add new column to the schema
  /// \param[in] name name of the column.
  /// \param[in] de_type data type of the column(TypeId).
  /// \param[in] shape shape of the column.
  /// \return bool true if schema init success
  bool add_column(std::string name, TypeId de_type, std::vector<int32_t> shape);

  /// \brief Add new column to the schema
  /// \param[in] name name of the column.
  /// \param[in] de_type data type of the column(std::string).
  /// \param[in] shape shape of the column.
  /// \return bool true if schema init success
  bool add_column(std::string name, std::string de_type, std::vector<int32_t> shape);

  /// \brief Get a JSON string of the schema
  /// \return JSON string of the schema
  std::string to_json();

  /// \brief Get a JSON string of the schema
  std::string to_string() { return to_json(); }

  /// \brief set a new value to dataset_type
  inline void set_dataset_type(std::string dataset_type) { dataset_type_ = dataset_type; }

  /// \brief set a new value to num_rows
  inline void set_num_rows(int32_t num_rows) { num_rows_ = num_rows; }

  /// \brief get the current num_rows
  inline int32_t get_num_rows() { return num_rows_; }

 private:
  /// \brief Parse the columns and add it to columns
  /// \param[in] columns dataset attribution information, decoded from schema file.
  ///    support both nlohmann::json::value_t::array and nlohmann::json::value_t::onject.
  /// \return JSON string of the schema
  bool parse_column(nlohmann::json columns);

  /// \brief Get schema file from json file
  /// \param[in] json_obj object of json parsed.
  /// \return bool true if json dump success
  bool from_json(nlohmann::json json_obj);

  int32_t num_rows_;
  std::string dataset_type_;
  std::string schema_file_;
  nlohmann::json columns_;
};

class BatchDataset : public Dataset {
 public:
  BatchDataset(std::shared_ptr<Dataset> input, int32_t batch_size, bool drop_remainder = false);
};

#ifndef ENABLE_ANDROID
class BucketBatchByLengthDataset : public Dataset {
 public:
  BucketBatchByLengthDataset(
    std::shared_ptr<Dataset> input, const std::vector<std::string> &column_names,
    const std::vector<int32_t> &bucket_boundaries, const std::vector<int32_t> &bucket_batch_sizes,
    std::function<TensorRow(TensorRow)> element_length_function = nullptr,
    const std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> &pad_info = {},
    bool pad_to_bucket_boundary = false, bool drop_remainder = false);
};

class ConcatDataset : public Dataset {
 public:
  explicit ConcatDataset(const std::vector<std::shared_ptr<Dataset>> &input);
};

class FilterDataset : public Dataset {
 public:
  FilterDataset(std::shared_ptr<Dataset> input, std::function<TensorRow(TensorRow)> predicate,
                std::vector<std::string> input_columns);
};
#endif

class MapDataset : public Dataset {
 public:
  MapDataset(std::shared_ptr<Dataset> input, std::vector<std::shared_ptr<TensorOperation>> operations,
             std::vector<std::string> input_columns, std::vector<std::string> output_columns,
             const std::vector<std::string> &project_columns, const std::shared_ptr<DatasetCache> &cache,
             std::vector<std::shared_ptr<DSCallback>> callbacks);
};

class ProjectDataset : public Dataset {
 public:
  ProjectDataset(std::shared_ptr<Dataset> input, const std::vector<std::string> &columns);
};

#ifndef ENABLE_ANDROID
class RenameDataset : public Dataset {
 public:
  RenameDataset(std::shared_ptr<Dataset> input, const std::vector<std::string> &input_columns,
                const std::vector<std::string> &output_columns);
};
#endif

class RepeatDataset : public Dataset {
 public:
  RepeatDataset(std::shared_ptr<Dataset> input, int32_t count);
};

class ShuffleDataset : public Dataset {
 public:
  ShuffleDataset(std::shared_ptr<Dataset> input, int32_t buffer_size);
};

#ifndef ENABLE_ANDROID
class SkipDataset : public Dataset {
 public:
  SkipDataset(std::shared_ptr<Dataset> input, int32_t count);
};

class TakeDataset : public Dataset {
 public:
  TakeDataset(std::shared_ptr<Dataset> input, int32_t count);
};

class ZipDataset : public Dataset {
 public:
  explicit ZipDataset(const std::vector<std::shared_ptr<Dataset>> &inputs);
};
#endif
/// \brief Function to create a SchemaObj
/// \param[in] schema_file Path of schema file
/// \return Shared pointer to the current schema
std::shared_ptr<SchemaObj> Schema(const std::string &schema_file = "");

class AlbumDataset : public Dataset {
 public:
  AlbumDataset(const std::string &dataset_dir, const std::string &data_schema,
               const std::vector<std::string> &column_names = {}, bool decode = false,
               const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
               const std::shared_ptr<DatasetCache> &cache = nullptr);
};

/// \brief Function to create an AlbumDataset
/// \notes The generated dataset is specified through setting a schema
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] data_schema Path to dataset schema file
/// \param[in] column_names Column names used to specify columns to load, if empty, will read all columns.
///     (default = {})
/// \param[in] decode the option to decode the images in dataset (default = false)
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is not given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
///     The cache feature is under development and is not recommended.
/// \return Shared pointer to the current Dataset
std::shared_ptr<AlbumDataset> Album(const std::string &dataset_dir, const std::string &data_schema,
                                    const std::vector<std::string> &column_names = {}, bool decode = false,
                                    const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                                    const std::shared_ptr<DatasetCache> &cache = nullptr);

#ifndef ENABLE_ANDROID
class CelebADataset : public Dataset {
 public:
  explicit CelebADataset(const std::string &dataset_dir, const std::string &usage = "all",
                         const std::shared_ptr<SamplerObj> &sampler = RandomSampler(), bool decode = false,
                         const std::set<std::string> &extensions = {},
                         const std::shared_ptr<DatasetCache> &cache = nullptr);
};

/// \brief Function to create a CelebADataset
/// \notes The generated dataset has two columns ['image', 'attr'].
///      The type of the image tensor is uint8. The attr tensor is uint32 and one hot type.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage One of "all", "train", "valid" or "test" (default = "all").
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is not given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] extensions Set of file extensions to be included in the dataset (default={}).
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
///     The cache feature is under development and is not recommended.
/// \return Shared pointer to the current Dataset
std::shared_ptr<CelebADataset> CelebA(const std::string &dataset_dir, const std::string &usage = "all",
                                      const std::shared_ptr<SamplerObj> &sampler = RandomSampler(), bool decode = false,
                                      const std::set<std::string> &extensions = {},
                                      const std::shared_ptr<DatasetCache> &cache = nullptr);

class Cifar10Dataset : public Dataset {
 public:
  explicit Cifar10Dataset(const std::string &dataset_dir, const std::string &usage = "all",
                          const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                          const std::shared_ptr<DatasetCache> &cache = nullptr);
};

/// \brief Function to create a Cifar10 Dataset
/// \notes The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] usage of CIFAR10, can be "train", "test" or "all" (default = "all").
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is not given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
///     The cache feature is under development and is not recommended.
/// \return Shared pointer to the current Dataset
std::shared_ptr<Cifar10Dataset> Cifar10(const std::string &dataset_dir, const std::string &usage = "all",
                                        const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                                        const std::shared_ptr<DatasetCache> &cache = nullptr);

class Cifar100Dataset : public Dataset {
 public:
  explicit Cifar100Dataset(const std::string &dataset_dir, const std::string &usage = "all",
                           const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                           const std::shared_ptr<DatasetCache> &cache = nullptr);
};

/// \brief Function to create a Cifar100 Dataset
/// \notes The generated dataset has three columns ["image", "coarse_label", "fine_label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] usage of CIFAR100, can be "train", "test" or "all" (default = "all").
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is not given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
///     The cache feature is under development and is not recommended.
/// \return Shared pointer to the current Dataset
std::shared_ptr<Cifar100Dataset> Cifar100(const std::string &dataset_dir, const std::string &usage = "all",
                                          const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                                          const std::shared_ptr<DatasetCache> &cache = nullptr);

class CLUEDataset : public Dataset {
 public:
  explicit CLUEDataset(const std::vector<std::string> &dataset_files, const std::string &task = "AFQMC",
                       const std::string &usage = "train", int64_t num_samples = 0,
                       ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1, int32_t shard_id = 0,
                       const std::shared_ptr<DatasetCache> &cache = nullptr);
};

/// \brief Function to create a CLUEDataset
/// \notes The generated dataset has a variable number of columns depending on the task and usage
/// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
///     will be sorted in a lexicographical order.
/// \param[in] task The kind of task, one of "AFQMC", "TNEWS", "IFLYTEK", "CMNLI", "WSC" and "CSL" (default="AFQMC").
/// \param[in] usage Be used to "train", "test" or "eval" data (default="train").
/// \param[in] num_samples The number of samples to be included in the dataset.
///     (Default = 0 means all samples.)
/// \param[in] shuffle The mode for shuffling data every epoch. (Default=ShuffleMode.kGlobal)
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1)
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified. (Default = 0)
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
///     The cache feature is under development and is not recommended.
/// \return Shared pointer to the current CLUEDataset
std::shared_ptr<CLUEDataset> CLUE(const std::vector<std::string> &dataset_files, const std::string &task = "AFQMC",
                                  const std::string &usage = "train", int64_t num_samples = 0,
                                  ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1,
                                  int32_t shard_id = 0, const std::shared_ptr<DatasetCache> &cache = nullptr);

class CocoDataset : public Dataset {
 public:
  CocoDataset(const std::string &dataset_dir, const std::string &annotation_file, const std::string &task = "Detection",
              const bool &decode = false, const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
              const std::shared_ptr<DatasetCache> &cache = nullptr);
};

/// \brief Function to create a CocoDataset
/// \notes The generated dataset has multi-columns :
///     - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
///                                  ['iscrowd', dtype=uint32]].
///     - task='Stuff', column: [['image', dtype=uint8], ['segmentation',dtype=float32], ['iscrowd', dtype=uint32]].
///     - task='Keypoint', column: [['image', dtype=uint8], ['keypoints', dtype=float32],
///                                 ['num_keypoints', dtype=uint32]].
///     - task='Panoptic', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
///                                 ['iscrowd', dtype=uint32], ['area', dtype=uitn32]].
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] annotation_file Path to the annotation json
/// \param[in] task Set the task type of reading coco data, now support 'Detection'/'Stuff'/'Panoptic'/'Keypoint'
/// \param[in] decode Decode the images after reading
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is not given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
///     The cache feature is under development and is not recommended.
/// \return Shared pointer to the current Dataset
std::shared_ptr<CocoDataset> Coco(const std::string &dataset_dir, const std::string &annotation_file,
                                  const std::string &task = "Detection", const bool &decode = false,
                                  const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                                  const std::shared_ptr<DatasetCache> &cache = nullptr);

class CSVDataset : public Dataset {
 public:
  explicit CSVDataset(const std::vector<std::string> &dataset_files, char field_delim = ',',
                      const std::vector<std::shared_ptr<CsvBase>> &column_defaults = {},
                      const std::vector<std::string> &column_names = {}, int64_t num_samples = 0,
                      ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1, int32_t shard_id = 0,
                      const std::shared_ptr<DatasetCache> &cache = nullptr);
};

/// \brief Function to create a CSVDataset
/// \notes The generated dataset has a variable number of columns
/// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
///    will be sorted in a lexicographical order.
/// \param[in] field_delim A char that indicates the delimiter to separate fields (default=',').
/// \param[in] column_defaults List of default values for the CSV field (default={}). Each item in the list is
///    either a valid type (float, int, or string). If this is not provided, treats all columns as string type.
/// \param[in] column_names List of column names of the dataset (default={}). If this is not provided, infers the
///    column_names from the first row of CSV file.
/// \param[in] num_samples The number of samples to be included in the dataset.
///    (Default = 0 means all samples.)
/// \param[in] shuffle The mode for shuffling data every epoch. (Default=ShuffleMode::kGlobal)
///    Can be any of:
///    ShuffleMode::kFalse - No shuffling is performed.
///    ShuffleMode::kFiles - Shuffle files only.
///    ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1)
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///    specified only when num_shards is also specified. (Default = 0)
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
///     The cache feature is under development and is not recommended.
/// \return Shared pointer to the current Dataset
std::shared_ptr<CSVDataset> CSV(const std::vector<std::string> &dataset_files, char field_delim = ',',
                                const std::vector<std::shared_ptr<CsvBase>> &column_defaults = {},
                                const std::vector<std::string> &column_names = {}, int64_t num_samples = 0,
                                ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1,
                                int32_t shard_id = 0, const std::shared_ptr<DatasetCache> &cache = nullptr);

class ImageFolderDataset : public Dataset {
 public:
  explicit ImageFolderDataset(const std::string &dataset_dir, bool decode = false,
                              const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                              const std::set<std::string> &extensions = {},
                              const std::map<std::string, int32_t> &class_indexing = {},
                              const std::shared_ptr<DatasetCache> &cache = nullptr);
};

/// \brief Function to create an ImageFolderDataset
/// \notes A source dataset that reads images from a tree of directories
///     All images within one folder have the same label
///     The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] decode A flag to decode in ImageFolder
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is not given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] extensions File extensions to be read
/// \param[in] class_indexing a class name to label map
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
///     The cache feature is under development and is not recommended.
/// \return Shared pointer to the current ImageFolderDataset
std::shared_ptr<ImageFolderDataset> ImageFolder(const std::string &dataset_dir, bool decode = false,
                                                const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                                                const std::set<std::string> &extensions = {},
                                                const std::map<std::string, int32_t> &class_indexing = {},
                                                const std::shared_ptr<DatasetCache> &cache = nullptr);

class ManifestDataset : public Dataset {
 public:
  explicit ManifestDataset(const std::string &dataset_file, const std::string &usage = "train",
                           const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                           const std::map<std::string, int32_t> &class_indexing = {}, bool decode = false,
                           const std::shared_ptr<DatasetCache> &cache = nullptr);
};

/// \brief Function to create a ManifestDataset
/// \notes The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_file The dataset file to be read
/// \param[in] usage Need "train", "eval" or "inference" data (default="train")
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is not given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] class_indexing A str-to-int mapping from label name to index (default={}, the folder
///     names will be sorted alphabetically and each class will be given a unique index starting from 0).
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
///     The cache feature is under development and is not recommended.
/// \return Shared pointer to the current ManifestDataset
std::shared_ptr<ManifestDataset> Manifest(const std::string &dataset_file, const std::string &usage = "train",
                                          const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                                          const std::map<std::string, int32_t> &class_indexing = {},
                                          bool decode = false, const std::shared_ptr<DatasetCache> &cache = nullptr);

class MindDataDataset : public Dataset {
 public:
  explicit MindDataDataset(const std::string &dataset_file, const std::vector<std::string> &columns_list = {},
                           const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                           nlohmann::json padded_sample = nullptr, int64_t num_padded = 0);
  explicit MindDataDataset(const std::vector<std::string> &dataset_files,
                           const std::vector<std::string> &columns_list = {},
                           const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                           nlohmann::json padded_sample = nullptr, int64_t num_padded = 0);
};

/// \brief Function to create a MindDataDataset
/// \param[in] dataset_file File name of one component of a mindrecord source. Other files with identical source
///     in the same path will be found and loaded automatically.
/// \param[in] columns_list List of columns to be read (default={})
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is not given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()),
///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
/// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
/// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
/// \return Shared pointer to the current MindDataDataset
std::shared_ptr<MindDataDataset> MindData(const std::string &dataset_file,
                                          const std::vector<std::string> &columns_list = {},
                                          const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                                          nlohmann::json padded_sample = nullptr, int64_t num_padded = 0);

/// \brief Function to create a MindDataDataset
/// \param[in] dataset_files List of dataset files to be read directly.
/// \param[in] columns_list List of columns to be read (default={})
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is not given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()),
///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
/// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
/// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
/// \return Shared pointer to the current MindDataDataset
std::shared_ptr<MindDataDataset> MindData(const std::vector<std::string> &dataset_files,
                                          const std::vector<std::string> &columns_list = {},
                                          const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                                          nlohmann::json padded_sample = nullptr, int64_t num_padded = 0);

class MnistDataset : public Dataset {
 public:
  explicit MnistDataset(const std::string &dataset_dir, const std::string &usage = "all",
                        const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                        const std::shared_ptr<DatasetCache> &cache = nullptr);
};

/// \brief Function to create a MnistDataset
/// \notes The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] usage of MNIST, can be "train", "test" or "all" (default = "all").
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is not given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
///     The cache feature is under development and is not recommended.
/// \return Shared pointer to the current MnistDataset
std::shared_ptr<MnistDataset> Mnist(const std::string &dataset_dir, const std::string &usage = "all",
                                    const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                                    const std::shared_ptr<DatasetCache> &cache = nullptr);

/// \brief Function to create a ConcatDataset
/// \notes Reload "+" operator to concat two datasets
/// \param[in] datasets1 Shared pointer to the first dataset to be concatenated
/// \param[in] datasets2 Shared pointer to the second dataset to be concatenated
/// \return Shared pointer to the current ConcatDataset
std::shared_ptr<ConcatDataset> operator+(const std::shared_ptr<Dataset> &datasets1,
                                         const std::shared_ptr<Dataset> &datasets2);

class RandomDataDataset : public Dataset {
 public:
  RandomDataDataset(const int32_t &total_rows, std::shared_ptr<SchemaObj> schema,
                    const std::vector<std::string> &columns_list, std::shared_ptr<DatasetCache> cache);

  RandomDataDataset(const int32_t &total_rows, std::string schema_path, const std::vector<std::string> &columns_list,
                    std::shared_ptr<DatasetCache> cache);
};

/// \brief Function to create a RandomDataset
/// \param[in] total_rows Number of rows for the dataset to generate (default=0, number of rows is random)
/// \param[in] schema SchemaObj to set column type, data type and data shape
/// \param[in] columns_list List of columns to be read (default={}, read all columns)
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is not given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
///     The cache feature is under development and is not recommended.
/// \return Shared pointer to the current Dataset
template <typename T = std::shared_ptr<SchemaObj>>
std::shared_ptr<RandomDataDataset> RandomData(const int32_t &total_rows = 0, const T &schema = nullptr,
                                              const std::vector<std::string> &columns_list = {},
                                              const std::shared_ptr<DatasetCache> &cache = nullptr) {
  std::shared_ptr<RandomDataDataset> ds;
  if constexpr (std::is_same<T, std::nullptr_t>::value || std::is_same<T, std::shared_ptr<SchemaObj>>::value) {
    std::shared_ptr<SchemaObj> schema_obj = schema;
    ds = std::make_shared<RandomDataDataset>(total_rows, std::move(schema_obj), std::move(columns_list), cache);
  } else {
    ds = std::make_shared<RandomDataDataset>(total_rows, std::move(schema), std::move(columns_list), cache);
  }
  return ds;
}

class TextFileDataset : public Dataset {
 public:
  explicit TextFileDataset(const std::vector<std::string> &dataset_files, int64_t num_samples = 0,
                           ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1, int32_t shard_id = 0,
                           const std::shared_ptr<DatasetCache> &cache = nullptr);
};

/// \brief Function to create a TextFileDataset
/// \notes The generated dataset has one column ['text']
/// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
///     will be sorted in a lexicographical order.
/// \param[in] num_samples The number of samples to be included in the dataset.
///     (Default = 0 means all samples.)
/// \param[in] shuffle The mode for shuffling data every epoch. (Default=ShuffleMode.kGlobal)
///     Can be any of:
///     ShuffleMode.kFalse - No shuffling is performed.
///     ShuffleMode.kFiles - Shuffle files only.
///     ShuffleMode.kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1)
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified. (Default = 0)
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
///     The cache feature is under development and is not recommended.
/// \return Shared pointer to the current TextFileDataset
std::shared_ptr<TextFileDataset> TextFile(const std::vector<std::string> &dataset_files, int64_t num_samples = 0,
                                          ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1,
                                          int32_t shard_id = 0, const std::shared_ptr<DatasetCache> &cache = nullptr);

class TFRecordDataset : public Dataset {
 public:
  TFRecordDataset(const std::vector<std::string> &dataset_files, std::string schema,
                  const std::vector<std::string> &columns_list, int64_t num_samples, ShuffleMode shuffle,
                  int32_t num_shards, int32_t shard_id, bool shard_equal_rows, std::shared_ptr<DatasetCache> cache);

  /// \brief Constructor
  /// \note Parameter 'schema' is shared pointer to Schema object
  TFRecordDataset(const std::vector<std::string> &dataset_files, std::shared_ptr<SchemaObj> schema,
                  const std::vector<std::string> &columns_list, int64_t num_samples, ShuffleMode shuffle,
                  int32_t num_shards, int32_t shard_id, bool shard_equal_rows, std::shared_ptr<DatasetCache> cache);
};

/// \brief Function to create a TFRecordDataset
/// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
///     will be sorted in a lexicographical order.
/// \param[in] schema SchemaObj or string to schema path. (Default = nullptr, which means that the
///     meta data from the TFData file is considered the schema.)
/// \param[in] columns_list List of columns to be read. (Default = {}, read all columns)
/// \param[in] num_samples The number of samples to be included in the dataset.
///     (Default = 0 means all samples.)
///     If num_samples is 0 and numRows(parsed from schema) does not exist, read the full dataset;
///     If num_samples is 0 and numRows(parsed from schema) is greater than 0, read numRows rows;
///     If both num_samples and numRows(parsed from schema) are greater than 0, read num_samples rows.
/// \param[in] shuffle The mode for shuffling data every epoch. (Default = ShuffleMode::kGlobal)
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1)
/// \param[in] shard_id The shard ID within num_shards. This argument should be specified only
///     when num_shards is also specified. (Default = 0)
/// \param[in] shard_equal_rows Get equal rows for all shards. (Default = False, number of rows of
///     each shard may be not equal)
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
///     The cache feature is under development and is not recommended.
/// \return Shared pointer to the current TFRecordDataset
template <typename T = std::shared_ptr<SchemaObj>>
std::shared_ptr<TFRecordDataset> TFRecord(const std::vector<std::string> &dataset_files, const T &schema = nullptr,
                                          const std::vector<std::string> &columns_list = {}, int64_t num_samples = 0,
                                          ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1,
                                          int32_t shard_id = 0, bool shard_equal_rows = false,
                                          const std::shared_ptr<DatasetCache> &cache = nullptr) {
  std::shared_ptr<TFRecordDataset> ds = nullptr;
  if constexpr (std::is_same<T, std::nullptr_t>::value || std::is_same<T, std::shared_ptr<SchemaObj>>::value) {
    std::shared_ptr<SchemaObj> schema_obj = schema;
    ds = std::make_shared<TFRecordDataset>(dataset_files, schema_obj, columns_list, num_samples, shuffle, num_shards,
                                           shard_id, shard_equal_rows, cache);
  } else {
    std::string schema_path = schema;
    if (!schema_path.empty()) {
      Path schema_file(schema_path);
      if (!schema_file.Exists()) {
        MS_LOG(ERROR) << "TFRecordDataset: schema path [" << schema_path << "] is invalid or does not exist.";
        return nullptr;
      }
    }
    ds = std::make_shared<TFRecordDataset>(dataset_files, schema_path, columns_list, num_samples, shuffle, num_shards,
                                           shard_id, shard_equal_rows, cache);
  }
  return ds;
}

class VOCDataset : public Dataset {
 public:
  explicit VOCDataset(const std::string &dataset_dir, const std::string &task = "Segmentation",
                      const std::string &usage = "train", const std::map<std::string, int32_t> &class_indexing = {},
                      bool decode = false, const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                      const std::shared_ptr<DatasetCache> &cache = nullptr);
};

/// \brief Function to create a VOCDataset
/// \notes The generated dataset has multi-columns :
///     - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['label', dtype=uint32],
///                                  ['difficult', dtype=uint32], ['truncate', dtype=uint32]].
///     - task='Segmentation', column: [['image', dtype=uint8], ['target',dtype=uint8]].
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] task Set the task type of reading voc data, now only support "Segmentation" or "Detection"
/// \param[in] usage The type of data list text file to be read (default = "train").
/// \param[in] class_indexing A str-to-int mapping from label name to index, only valid in "Detection" task
/// \param[in] decode Decode the images after reading
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is not given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
///     The cache feature is under development and is not recommended.
/// \return Shared pointer to the current Dataset
std::shared_ptr<VOCDataset> VOC(const std::string &dataset_dir, const std::string &task = "Segmentation",
                                const std::string &usage = "train",
                                const std::map<std::string, int32_t> &class_indexing = {}, bool decode = false,
                                const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                                const std::shared_ptr<DatasetCache> &cache = nullptr);

/// \brief Function the create a cache to be attached to a dataset
/// \param id A user assigned session id for the current pipeline
/// \param mem_sz Size of the memory set aside for the row caching. 0 for unlimited
/// \param spill Spill to disk if out of memory
/// \param hostname optional host name
/// \param port optional port
/// \param num_connections optional number of connections
/// \param prefetch_sz optional prefetch size
/// \return Shared pointer to DatasetCache. If error, nullptr is returned.
std::shared_ptr<DatasetCache> CreateDatasetCache(session_id_type id, uint64_t mem_sz, bool spill,
                                                 std::optional<std::string> hostname = std::nullopt,
                                                 std::optional<int32_t> port = std::nullopt,
                                                 std::optional<int32_t> num_connections = std::nullopt,
                                                 std::optional<int32_t> prefetch_sz = std::nullopt);

/// \brief Function to create a ZipDataset
/// \notes Applies zip to the dataset
/// \param[in] datasets List of shared pointers to the datasets that we want to zip
/// \return Shared pointer to the current Dataset
std::shared_ptr<ZipDataset> Zip(const std::vector<std::shared_ptr<Dataset>> &datasets);
#endif
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASETS_H_
