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
#include "minddata/dataset/api/python/de_pipeline.h"

#include <algorithm>
#include <map>
#include <set>

#include "minddata/dataset/callback/py_ds_callback.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/cache/cache_client.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/bucket_batch_by_length_op.h"
#include "minddata/dataset/engine/datasetops/cache_op.h"
#include "minddata/dataset/engine/datasetops/device_queue_op.h"
#include "minddata/dataset/engine/datasetops/epoch_ctrl_op.h"
#include "minddata/dataset/engine/datasetops/filter_op.h"
#include "minddata/dataset/engine/datasetops/source/celeba_op.h"
#include "minddata/dataset/engine/datasetops/source/cifar_op.h"
#include "minddata/dataset/engine/datasetops/source/clue_op.h"
#include "minddata/dataset/engine/datasetops/source/coco_op.h"
#include "minddata/dataset/engine/datasetops/source/csv_op.h"
#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"
#include "minddata/dataset/engine/datasetops/source/manifest_op.h"
#include "minddata/dataset/engine/datasetops/source/mnist_op.h"
#include "minddata/dataset/engine/datasetops/source/random_data_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/datasetops/source/text_file_op.h"
#include "minddata/dataset/engine/datasetops/source/voc_op.h"
#include "minddata/dataset/kernels/py_func_op.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"
#include "minddata/mindrecord/include/shard_category.h"
#include "minddata/mindrecord/include/shard_distributed_sample.h"
#include "minddata/mindrecord/include/shard_header.h"
#include "minddata/mindrecord/include/shard_index_generator.h"
#include "minddata/mindrecord/include/shard_sample.h"
#include "minddata/mindrecord/include/shard_shuffle.h"
#include "minddata/mindrecord/include/shard_writer.h"
#include "pybind11/stl.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
using json = nlohmann::json;
using pFunction = Status (DEPipeline::*)(const py::dict &, std::shared_ptr<DatasetOp> *, std::shared_ptr<DatasetOp> *);

static std::unordered_map<uint32_t, pFunction> g_parse_op_func_ = {
  {kShuffle, &DEPipeline::ParseShuffleOp},
  {kMindrecord, &DEPipeline::ParseMindRecordOp},
  {kMap, &DEPipeline::ParseMapOp},
  {kFilter, &DEPipeline::ParseFilterOp},
  {kBatch, &DEPipeline::ParseBatchOp},
  {kBucketBatch, &DEPipeline::ParseBucketBatchByLengthOp},
  {kBarrier, &DEPipeline::ParseBarrierOp},
  {kRepeat, &DEPipeline::ParseRepeatOp},
  {kSkip, &DEPipeline::ParseSkipOp},
  {kZip, &DEPipeline::ParseZipOp},
  {kConcat, &DEPipeline::ParseConcatOp},
  {kRename, &DEPipeline::ParseRenameOp},
  {kDeviceQueue, &DEPipeline::ParseDeviceQueueOp},
  {kGenerator, &DEPipeline::ParseGeneratorOp},
  {kTfReader, &DEPipeline::ParseTFReaderOp},
  {kProject, &DEPipeline::ParseProjectOp},
  {kTake, &DEPipeline::ParseTakeOp},
  {kImageFolder, &DEPipeline::ParseImageFolderOp},
  {kMnist, &DEPipeline::ParseMnistOp},
  {kManifest, &DEPipeline::ParseManifestOp},
  {kVoc, &DEPipeline::ParseVOCOp},
  {kCoco, &DEPipeline::ParseCocoOp},
  {kCifar10, &DEPipeline::ParseCifar10Op},
  {kCifar100, &DEPipeline::ParseCifar100Op},
  {kCelebA, &DEPipeline::ParseCelebAOp},
  {kRandomData, &DEPipeline::ParseRandomDataOp},
  {kTextFile, &DEPipeline::ParseTextFileOp},
  {kBuildVocab, &DEPipeline::ParseBuildVocabOp},
  {kClue, &DEPipeline::ParseClueOp},
  {kEpochCtrl, &DEPipeline::ParseEpochCtrlOp},
  {kCsv, &DEPipeline::ParseCsvOp},
  {kSentencePieceVocab, &DEPipeline::ParseBuildSentencePieceVocabOp}};

DEPipeline::DEPipeline() : iterator_(nullptr) {
  try {
    // One time init
    (void)GlobalInit();

    // Instantiate the execution tree
    tree_ = std::make_shared<ExecutionTree>();
    repeat_num_ = 1;
    batch_size_ = 1;
    num_rows_ = 0;
    num_classes_ = 0;
    temp_batch_size_ = 1;
    temp_drop_remainder_ = false;
  } catch (const std::exception &err) {
    MS_LOG(ERROR) << "Dataset pipeline exception caught on init: " << err.what() << ".";
    return;
  }
}

DEPipeline::~DEPipeline() {
  {
    // Release GIL before joining all threads
    py::gil_scoped_release gil_release;
    // Release tree
    tree_.reset();
  }
}

// Function to add a Node to the Execution Tree.
Status DEPipeline::AddNodeToTree(const OpName &op_name, const py::dict &args, py::dict *output) {
  // For each operator, Parse through the list of arguments, then call the respective builder/constructor.
  // Note that each call to the parse function may result in building more than one dataset operator.
  // For example, one call to ParseNNNOp may result in multiple internal C nodes:
  // nodeA
  //   |
  // nodeB
  //   |
  // nodeC
  // However, the python side dataset is more abstract, and it does not know about the potential subtree that
  // is being built here. Since the python api is hooking tree nodes together (parent/child hookups), the
  // python side needs to know about nodeA and NodeC to be able to appropriately hook up parents and child
  // to this subtee.
  // Thus, it is required that both the top-most parent and bottom-most child are returned from the parse
  // function.
  DsOpPtr top = nullptr;
  DsOpPtr bottom = nullptr;
  auto iter = g_parse_op_func_.find(op_name);
  if (iter != g_parse_op_func_.end()) {
    pFunction func = iter->second;
    RETURN_IF_NOT_OK((this->*func)(args, &top, &bottom));

    if (top == nullptr) {
      RETURN_STATUS_UNEXPECTED("An operator was parsed but it did not produce a C node.");
    }

    // It is not required that the parse function always produces the bottom pointer. If it's still null,
    // then set top and bottom to be the same operator
    if (bottom == nullptr) bottom = top;

    // Pack these pointers into a py dict so that we can return both back to python.
    (*output)["top"] = top;
    (*output)["bottom"] = bottom;
  } else {
    RETURN_STATUS_UNEXPECTED("No such Op");
  }
  // Associate current dataset op node with the tree.
  RETURN_IF_NOT_OK(tree_->AssociateNode(top));
  return Status::OK();
}
// Function to add a child and parent relationship.
Status DEPipeline::AddChildToParentNode(const DsOpPtr &child_op, const DsOpPtr &parent_op) {
  // Link this relationship.
  // Note parent node takes ownership of the child
  return (parent_op->AddChild(child_op));
}

// Function to assign the node as root.
Status DEPipeline::AssignRootNode(const DsOpPtr &dataset_op) { return (tree_->AssignRoot(dataset_op)); }

// Function to prepare the tree
Status DEPipeline::PrepareTree(const int32_t num_epochs) { return tree_->Prepare(num_epochs); }

// Function to launch the tree execution.
Status DEPipeline::LaunchTreeExec() {
  RETURN_IF_NOT_OK(tree_->Launch());
  iterator_ = std::make_unique<DatasetIterator>(tree_);
  if (iterator_ == nullptr) RETURN_STATUS_UNEXPECTED("Cannot create an Iterator.");
  return Status::OK();
}

void DEPipeline::PrintTree() {
  for (auto itr = tree_->begin(); itr != tree_->end(); ++itr) {
    std::stringstream ss;
    ss << *itr;
    MS_LOG(DEBUG) << "Operator ID is " << itr->id() << ". Details: " << ss.str().c_str() << ".";
  }
}

Status DEPipeline::GetColumnNames(py::list *output) {
  if (!tree_->isPrepared()) {
    RETURN_STATUS_UNEXPECTED("GetColumnNames: Make sure to call prepare before calling GetColumnNames.");
  }
  std::unordered_map<std::string, int32_t> column_name_id_map = tree_->root()->column_name_id_map();
  if (column_name_id_map.empty())
    RETURN_STATUS_UNEXPECTED("GetColumnNames: Column names was empty. Make sure Prepare is called.");
  std::vector<std::pair<std::string, int32_t>> column_name_id_vector(column_name_id_map.begin(),
                                                                     column_name_id_map.end());
  std::sort(column_name_id_vector.begin(), column_name_id_vector.end(),
            [](const std::pair<std::string, int32_t> &a, const std::pair<std::string, int32_t> &b) {
              return a.second < b.second;
            });
  for (auto item : column_name_id_vector) {
    (*output).append(item.first);
  }
  return Status::OK();
}

Status DEPipeline::GetNextAsMap(py::dict *output) {
  std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> vec;
  Status s;
  {
    py::gil_scoped_release gil_release;
    s = iterator_->GetNextAsOrderedPair(&vec);
  }
  RETURN_IF_NOT_OK(s);

  // Generate Python dict, python dict maintains its insertion order
  for (const auto &pair : vec) {
    (*output)[common::SafeCStr(pair.first)] = pair.second;
  }
  return Status::OK();
}

Status DEPipeline::GetNextAsList(py::list *output) {
  TensorRow row;
  Status s;
  {
    py::gil_scoped_release gil_release;
    s = iterator_->FetchNextTensorRow(&row);
  }
  RETURN_IF_NOT_OK(s);
  // Generate Python list as return
  for (auto el : row) {
    output->append(el);
  }
  return Status::OK();
}

Status DEPipeline::GetOutputShapes(py::list *output) {
  std::vector<TensorShape> shapes;
  Status s;
  {
    py::gil_scoped_release gil_release;
    s = iterator_->GetOutputShapes(&shapes);
  }
  RETURN_IF_NOT_OK(s);
  for (auto el : shapes) {
    py::list shape;
    for (auto dim : el.AsVector()) {
      shape.append(dim);
    }
    output->append(shape);
  }
  return Status::OK();
}

Status DEPipeline::GetOutputTypes(py::list *output) {
  std::vector<DataType> types;
  Status s;
  {
    py::gil_scoped_release gil_release;
    s = iterator_->GetOutputTypes(&types);
  }
  RETURN_IF_NOT_OK(s);
  for (auto el : types) {
    output->append(el.AsNumpyType());
  }
  return Status::OK();
}

int DEPipeline::GetDatasetSize() const { return num_rows_ / batch_size_; }

int DEPipeline::GetBatchSize() const { return batch_size_; }

int DEPipeline::GetRepeatCount() const { return repeat_num_; }

float ToFloat(const py::handle &handle) { return py::reinterpret_borrow<py::float_>(handle); }

Status DEPipeline::StopSend() {
  // tree_.root() must be DeviceQueueOp
  DeviceQueueOp *op = dynamic_cast<DeviceQueueOp *>(tree_->root().get());
  if (op == nullptr) {
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "StopSend only supported by DeviceQueueOp");
  }
  op->StopSend();
  return Status::OK();
}

Status DEPipeline::ContinueSend() {
  // tree_.root() must be DeviceQueueOp
  DeviceQueueOp *op = dynamic_cast<DeviceQueueOp *>(tree_->root().get());
  if (op == nullptr) {
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "ContinueSend only supported by DeviceQueueOp");
  }
  op->ContinueSend();
  return Status::OK();
}

int ToInt(const py::handle &handle) { return py::reinterpret_borrow<py::int_>(handle); }

bool ToBool(const py::handle &handle) { return py::reinterpret_borrow<py::bool_>(handle); }

std::string ToString(const py::handle &handle) { return py::reinterpret_borrow<py::str>(handle); }

std::vector<std::string> ToStringVector(const py::handle handle) {
  py::list list = py::reinterpret_borrow<py::list>(handle);
  std::vector<std::string> vector;
  for (auto l : list) {
    if (!l.is_none())
      vector.push_back(py::str(l));
    else
      vector.emplace_back("");
  }
  return vector;
}

std::set<std::string> ToStringSet(const py::handle handle) {
  py::list list = py::reinterpret_borrow<py::list>(handle);
  std::set<std::string> set;
  for (auto l : list) {
    if (!l.is_none()) {
      (void)set.insert(py::str(l));
    }
  }
  return set;
}

std::map<std::string, int32_t> ToStringMap(const py::handle handle) {
  py::dict dict = py::reinterpret_borrow<py::dict>(handle);
  std::map<std::string, int32_t> map;
  for (auto p : dict) {
    (void)map.insert(std::make_pair(ToString(p.first), ToInt(p.second)));
  }
  return map;
}

std::vector<int> ToIntVector(const py::handle handle) {
  py::list list = py::reinterpret_borrow<py::list>(handle);
  std::vector<int> vector;
  for (auto l : list) {
    if (!l.is_none()) {
      vector.push_back(ToInt(l));
    }
  }
  return vector;
}

std::vector<DataType> ToTypeVector(const py::handle handle) {
  py::list list = py::reinterpret_borrow<py::list>(handle);
  std::vector<DataType> vector;
  for (auto l : list) {
    if (l.is_none()) {
      vector.emplace_back(DataType());
    } else {
      vector.push_back(l.cast<DataType>());
    }
  }
  return vector;
}

Status DEPipeline::SetBatchParameters(const py::dict &args) {
  if (args["batch_size"].is_none()) {
    std::string err_msg = "Error: batchSize is invalid or not set.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  temp_batch_size_ = ToInt(args["batch_size"]);
  CHECK_FAIL_RETURN_UNEXPECTED(temp_batch_size_ > 0, "Error: batchSize is invalid.");
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "drop_remainder") {
        temp_drop_remainder_ = ToBool(value);
      }
    }
  }

  return Status::OK();
}

Status DEPipeline::ParseShuffleOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                  std::shared_ptr<DatasetOp> *bottom) {
  std::shared_ptr<ShuffleOp::Builder> builder = std::make_shared<ShuffleOp::Builder>();
  if (!args["buffer_size"].is_none()) {
    (void)builder->SetShuffleSize(ToInt(args["buffer_size"]));
  } else {
    std::string err_msg = "Error: Shuffle buffer size is missing";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // Optional arguments
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "reshuffle_each_epoch") {
        (void)builder->SetReshuffleEachEpoch(ToBool(args["reshuffle_each_epoch"]));
      }
    }
  }

  std::shared_ptr<ShuffleOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *top = op;
  return Status::OK();
}

Status DEPipeline::SaveDataset(const std::vector<std::string> &file_names, const std::string &file_type) {
  Status s;
  auto mr_header = std::make_shared<mindrecord::ShardHeader>();
  auto mr_writer = std::make_unique<mindrecord::ShardWriter>();
  std::vector<std::string> blob_fields;
  uint64_t mr_schema_id = 0;
  if (mindrecord::SUCCESS != mindrecord::ShardWriter::initialize(&mr_writer, file_names)) {
    RETURN_STATUS_UNEXPECTED("Error: failed to initialize ShardWriter.");
  }

  TensorRow row;
  std::unordered_map<std::string, int32_t> column_name_id_map;
  for (auto el : iterator_->GetColumnNameMap()) {
    std::string column_name = el.first;
    std::transform(column_name.begin(), column_name.end(), column_name.begin(),
                   [](unsigned char c) { return ispunct(c) ? '_' : c; });
    column_name_id_map[column_name] = el.second;
  }
  bool first_loop = true;  // build schema in first loop
  do {
    json row_raw_data;
    std::map<std::string, std::unique_ptr<std::vector<uint8_t>>> row_bin_data;
    {
      py::gil_scoped_release gil_release;
      s = iterator_->FetchNextTensorRow(&row);
    }
    RETURN_IF_NOT_OK(s);
    if (row.empty()) break;
    if (first_loop) {
      json mr_json;
      std::vector<std::string> index_fields;
      s = FetchMetaFromTensorRow(column_name_id_map, row, &mr_json, &index_fields);
      RETURN_IF_NOT_OK(s);
      MS_LOG(DEBUG) << "Schema of saved mindrecord: " << mr_json.dump();
      if (mindrecord::SUCCESS !=
          mindrecord::ShardHeader::initialize(&mr_header, mr_json, index_fields, blob_fields, mr_schema_id)) {
        RETURN_STATUS_UNEXPECTED("Error: failed to initialize ShardHeader.");
      }
      mr_writer->SetShardHeader(mr_header);
      first_loop = false;
    }
    // construct data
    if (!row.empty()) {  // write data
      s = FetchDataFromTensorRow(row, column_name_id_map, &row_raw_data, &row_bin_data);
      RETURN_IF_NOT_OK(s);
      std::shared_ptr<std::vector<uint8_t>> output_bin_data;
      mr_writer->MergeBlobData(blob_fields, row_bin_data, &output_bin_data);
      std::map<std::uint64_t, std::vector<json>> raw_data;
      raw_data.insert(std::pair<uint64_t, std::vector<json>>(mr_schema_id, std::vector<json>{row_raw_data}));
      std::vector<std::vector<uint8_t>> bin_data;
      if (nullptr != output_bin_data) {
        bin_data.emplace_back(*output_bin_data);
      }
      mr_writer->WriteRawData(raw_data, bin_data);
    }
  } while (!row.empty());
  mr_writer->Commit();
  if (mindrecord::SUCCESS != mindrecord::ShardIndexGenerator::finalize(file_names)) {
    RETURN_STATUS_UNEXPECTED("Error: failed to finalize ShardIndexGenerator.");
  }
  return Status::OK();
}

Status DEPipeline::FetchDataFromTensorRow(const TensorRow &row,
                                          const std::unordered_map<std::string, int32_t> &column_name_id_map,
                                          json *row_raw_data,
                                          std::map<std::string, std::unique_ptr<std::vector<uint8_t>>> *row_bin_data) {
  if (row_raw_data == nullptr) {
    RETURN_STATUS_UNEXPECTED("error: row raw data is NULL.");
  }
  if (row_bin_data == nullptr) {
    RETURN_STATUS_UNEXPECTED("error: row bin data is NULL.");
  }
  if (column_name_id_map.empty()) {
    RETURN_STATUS_UNEXPECTED("Error: column not found");
  }
  Status s;
  for (auto &col : column_name_id_map) {
    auto idx = col.second;
    auto column_name = col.first;
    auto &tensor = row[idx];
    auto column_type = tensor->type();

    std::unique_ptr<std::vector<uint8_t>> data_ptr;
    if (column_type == DataType::DE_INT8) {
      std::unique_ptr<int32_t> data;
      std::unique_ptr<int8_t> dummy;
      s = TransfromTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy, true);
      RETURN_IF_NOT_OK(s);
      if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
    } else if (column_type == DataType::DE_INT16) {
      std::unique_ptr<int32_t> data;
      std::unique_ptr<int16_t> dummy;
      s = TransfromTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy, true);
      RETURN_IF_NOT_OK(s);
      if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
    } else if (column_type == DataType::DE_UINT16) {
      std::unique_ptr<int32_t> data;
      std::unique_ptr<uint16_t> dummy;
      s = TransfromTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy, true);
      RETURN_IF_NOT_OK(s);
      if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
    } else if (column_type == DataType::DE_UINT8) {
      std::unique_ptr<uint8_t> data, dummy;
      s = TransfromTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy);
      RETURN_IF_NOT_OK(s);
      if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
    } else if (column_type == DataType::DE_INT32) {
      std::unique_ptr<int32_t> data, dummy;
      s = TransfromTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy);
      RETURN_IF_NOT_OK(s);
      if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
    } else if (column_type == DataType::DE_UINT32) {
      std::unique_ptr<int64_t> data;
      std::unique_ptr<uint32_t> dummy;
      s = TransfromTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy, true);
      RETURN_IF_NOT_OK(s);
      if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
    } else if (column_type == DataType::DE_INT64) {
      std::unique_ptr<int64_t> data, dummy;
      s = TransfromTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy);
      RETURN_IF_NOT_OK(s);
      if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
    } else if (column_type == DataType::DE_FLOAT32) {
      std::unique_ptr<float> data, dummy;
      s = TransfromTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy);
      RETURN_IF_NOT_OK(s);
      if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
    } else if (column_type == DataType::DE_FLOAT64) {
      std::unique_ptr<double> data, dummy;
      s = TransfromTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy);
      RETURN_IF_NOT_OK(s);
      if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
    } else if (column_type == DataType::DE_STRING) {
      std::string_view sv;
      RETURN_IF_NOT_OK(tensor->GetItemAt(&sv, {0}));  // assume scalar string tensor
      std::string ss(sv);
      (*row_raw_data)[column_name] = std::move(ss);
      continue;
    } else {
      RETURN_STATUS_UNEXPECTED("Got unexpected type when casting data.");
    }
    RETURN_IF_NOT_OK(s);
    if (data_ptr != nullptr) {
      (*row_bin_data)[column_name] = std::move(data_ptr);
    }
  }
  return Status::OK();
}

template <typename T, typename S>
Status DEPipeline::TransfromTensor(const unsigned char *src, const TensorShape &shape, const int64_t num_of_elements,
                                   std::unique_ptr<T> *data, std::unique_ptr<std::vector<uint8_t>> *data_ptr,
                                   std::unique_ptr<S> *s, bool need_convert) {
  if (nullptr == src) {
    RETURN_STATUS_UNEXPECTED("Error: buffer of Tensor is NULL.");
  }
  *data_ptr = std::make_unique<std::vector<uint8_t>>(num_of_elements * sizeof(T));
  if (need_convert) {
    auto tmp_ptr = std::make_unique<std::vector<uint8_t>>(num_of_elements * sizeof(S));
    std::copy(src, src + sizeof(S) * num_of_elements, tmp_ptr->begin());
    auto s_ptr = reinterpret_cast<S *>(&(*(tmp_ptr->begin())));
    auto el = std::make_unique<T>();
    for (uint32_t i = 0; i < num_of_elements; ++i) {
      *el = *(s_ptr + i);
      auto t_ptr = reinterpret_cast<uint8_t *>(el.get());
      for (uint32_t j = 0; j < sizeof(T); ++j) {
        *((*data_ptr)->begin() + i * sizeof(T) + j) = *(t_ptr + j);
      }
    }
  } else {
    std::copy(src, src + sizeof(T) * num_of_elements, (*data_ptr)->begin());
  }
  if (shape.empty()) {
    *data = std::make_unique<T>();
    auto t_ptr = reinterpret_cast<uint8_t *>((*data).get());
    for (uint32_t i = 0; i < sizeof(T); ++i) {
      *(t_ptr + i) = *((*data_ptr)->begin() + i);
    }
  }
  return Status::OK();
}

Status DEPipeline::FetchMetaFromTensorRow(const std::unordered_map<std::string, int32_t> &column_name_id_map,
                                          const TensorRow &row, json *schema, std::vector<std::string> *index_fields) {
  if (schema == nullptr) {
    RETURN_STATUS_UNEXPECTED("error: schema is NULL.");
  }
  if (index_fields == nullptr) {
    RETURN_STATUS_UNEXPECTED("error: index fields is NULL.");
  }
  if (column_name_id_map.empty()) {
    RETURN_STATUS_UNEXPECTED("Error: column not found.");
  }
  json dataset_schema;
  for (auto &col : column_name_id_map) {
    auto idx = col.second;
    auto column_name = col.first;
    auto &tensor = row[idx];
    auto column_type = tensor->type();
    auto column_shape = tensor->shape();

    std::string mr_type;
    auto shapes = column_shape.AsVector();
    std::vector<int> mr_shape(shapes.begin(), shapes.end());
    std::string el = column_type.ToString();
    dataset_schema[column_name] = el;
    if (mindrecord::kTypesMap.find(el) == mindrecord::kTypesMap.end()) {
      std::string err_msg("Error: can not support data type: " + el);
      RETURN_STATUS_UNEXPECTED(err_msg);
    } else {
      mr_type = mindrecord::kTypesMap.at(el);
    }
    if (mr_shape.empty()) {
      if (mr_type == "bytes") {  // map to int32 when bytes without shape.
        mr_type = "int32";
      }
      (*schema)[column_name] = {{"type", mr_type}};
    } else {
      if (mr_type == "string") {  // mindrecord can not support string with shape.
        std::string err_msg("Error: mindrecord can not support multi-dimensional string tensor.");
        RETURN_STATUS_UNEXPECTED(err_msg);
      }
      if (mr_type == "bytes") {  // ignore shape of bytes in minrecord
        (*schema)[column_name] = {{"type", mr_type}};
      } else {
        (*schema)[column_name] = {{"type", mr_type}, {"shape", mr_shape}};
      }
    }
    if (mr_type == "bytes" || !mr_shape.empty()) continue;
    index_fields->emplace_back(column_name);  // candidate of index fields
  }
  MS_LOG(DEBUG) << "Schema of dataset: " << dataset_schema.dump();
  return Status::OK();
}
Status DEPipeline::BuildMindrecordSamplerChain(const py::handle &handle,
                                               std::vector<std::shared_ptr<mindrecord::ShardOperator>> *operators,
                                               int num_padded) {
  auto sampler = py::reinterpret_borrow<py::object>(handle);
  auto create = sampler.attr("create_for_minddataset");
  auto op = create().cast<std::shared_ptr<mindrecord::ShardOperator>>();
  std::stack<std::shared_ptr<mindrecord::ShardOperator>> stack_ops;
  while (op != nullptr) {
    auto sampler_op = std::dynamic_pointer_cast<mindrecord::ShardDistributedSample>(op);
    if (sampler_op && num_padded > 0) {
      sampler_op->SetNumPaddedSamples(num_padded);
      stack_ops.push(sampler_op);
    } else {
      stack_ops.push(op);
    }
    op = op->GetChildOp();
  }
  while (!stack_ops.empty()) {
    operators->push_back(stack_ops.top());
    stack_ops.pop();
  }
  return Status::OK();
}

Status DEPipeline::ParseMindRecordOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                     std::shared_ptr<DatasetOp> *bottom) {
  if (args["dataset_file"].is_none()) {
    std::string err_msg = "Error: at least one of dataset_files is missing";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  std::shared_ptr<MindRecordOp::Builder> builder = std::make_shared<MindRecordOp::Builder>();
  bool load_dataset = ToBool(args["load_dataset"]);
  if (load_dataset == true) {
    (void)builder->SetDatasetFile({ToString(args["dataset_file"])});
  } else {
    (void)builder->SetDatasetFile(ToStringVector(args["dataset_file"]));
  }
  (void)builder->SetLoadDataset(load_dataset);
  std::vector<std::string> in_col_names;
  if (!args["columns_list"].is_none()) {
    in_col_names = ToStringVector(args["columns_list"]);
    if (in_col_names.empty() || in_col_names[0].empty()) {
      std::string err_msg = "Error: columns_list is invalid or not set.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    (void)builder->SetColumnsToLoad(in_col_names);
  }

  if (!args["padded_sample"].is_none()) {
    (void)builder->SetPaddedSample(args["padded_sample"]);
    (void)builder->SetNumToPadSamples(ToInt(args["num_padded"]));
  }
  std::vector<std::shared_ptr<mindrecord::ShardOperator>> operators;
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        (void)builder->SetNumMindRecordWorkers(ToInt(value));
      } else if (key == "sampler") {
        int num_padded = 0;
        if (!args["num_padded"].is_none()) {
          num_padded = ToInt(args["num_padded"]);
        }
        RETURN_IF_NOT_OK(BuildMindrecordSamplerChain(value, &operators, num_padded));
      }
    }
  }

  if (!operators.empty()) {
    (void)builder->SetOperators(operators);
  }
  std::shared_ptr<MindRecordOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  num_rows_ = op->num_rows();
  *top = op;
  return Status::OK();
}

Status DEPipeline::ParseMapOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                              std::shared_ptr<DatasetOp> *bottom) {
  MapOp::Builder map_builder;
  std::vector<std::shared_ptr<TensorOp>> tensor_op_list;
  std::vector<std::string> project_columns;
  std::shared_ptr<CacheClient> cache_client = nullptr;
  int num_workers = 0;

  if (args["operations"].is_none()) RETURN_STATUS_UNEXPECTED("Error: 'operations' is not set. \n");

  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "input_columns") {
        std::vector<std::string> in_col_names = ToStringVector(args["input_columns"]);
        (void)map_builder.SetInColNames(in_col_names);
      } else if (key == "output_columns") {
        (void)map_builder.SetOutColNames(ToStringVector(value));
      } else if (key == "column_order") {
        project_columns = ToStringVector(value);
      } else if (key == "num_parallel_workers") {
        num_workers = ToInt(value);
        (void)map_builder.SetNumWorkers(num_workers);
      } else if (key == "prefetch_size") {
        (void)map_builder.SetOpConnectorSize(ToInt(value));
      } else if (key == "operations") {
        py::handle tensor_ops = args["operations"];
        // operation can be a list of TensorOps or a single TensorOp.
        if (py::isinstance<py::list>(tensor_ops)) {
          for (auto op : tensor_ops) {
            std::shared_ptr<TensorOp> tensor_op;
            if (py::isinstance<TensorOp>(op)) {
              tensor_op = op.cast<std::shared_ptr<TensorOp>>();
            } else if (py::isinstance<py::function>(op)) {
              tensor_op = std::make_shared<PyFuncOp>(op.cast<py::function>());
            } else {
              RETURN_STATUS_UNEXPECTED("Error: tensor_op is not recognised (not TensorOp and not pyfunc).");
            }
            tensor_op_list.push_back(tensor_op);
          }
        }
        CHECK_FAIL_RETURN_UNEXPECTED(!tensor_op_list.empty(), "Error: tensor_op is invalid or not set.");
        (void)map_builder.SetTensorFuncs(std::move(tensor_op_list));
      } else if (key == "cache") {
        cache_client = value.cast<std::shared_ptr<CacheClient>>();
      } else if (key == "callbacks") {
        std::vector<std::shared_ptr<DSCallback>> callbacks;
        std::transform(value.begin(), value.end(), std::back_inserter(callbacks),
                       [](py::handle cb) { return cb.cast<std::shared_ptr<PyDSCallback>>(); });
        (void)map_builder.AddCallbacks(callbacks);
      } else {
        RETURN_STATUS_UNEXPECTED("Error in parsing MapOp: Unhandled key: " + key);
      }
    }
  }

  std::shared_ptr<MapOp> map_op;
  RETURN_IF_NOT_OK(map_builder.Build(&map_op));
  RETURN_IF_NOT_OK(tree_->AssociateNode(map_op));
  *top = map_op;

  // Add a project op over top of the map if the user wanted to reposition the columns
  if (!project_columns.empty()) {
    ProjectOp::Builder proj_builder(project_columns);
    std::shared_ptr<ProjectOp> proj_op;
    RETURN_IF_NOT_OK(proj_builder.Build(&proj_op));
    RETURN_IF_NOT_OK(tree_->AssociateNode(proj_op));
    RETURN_IF_NOT_OK(proj_op->AddChild(map_op));
    *top = proj_op;
    *bottom = map_op;
  }

  // Additionally, add a cache if required.  This will go over top of the project op if one
  // was created, otherwise it goes over top of the map op
  if (cache_client) {
    std::shared_ptr<DatasetOp> cache_op = nullptr;
    RETURN_IF_NOT_OK(AddCacheOp(cache_client, num_workers, *top, &cache_op));
    *top = cache_op;
    *bottom = map_op;
  }

  return Status::OK();
}

Status DEPipeline::ParseFilterOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                 std::shared_ptr<DatasetOp> *bottom) {
  std::shared_ptr<FilterOp::Builder> builder = std::make_shared<FilterOp::Builder>();

  if (args["predicate"].is_none()) {
    RETURN_STATUS_UNEXPECTED("Error: 'predicate' is not set. \n");
  }

  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        (void)builder->SetNumWorkers(ToInt(value));
      } else if (key == "predicate") {
        py::handle op = args["predicate"];
        if (!py::isinstance<py::function>(op)) {
          RETURN_STATUS_UNEXPECTED("Error: predicate is not recognised (not pyfunc).");
        }
        std::shared_ptr<TensorOp> py_func;
        py_func = std::make_shared<PyFuncOp>(value.cast<py::function>(), DataType::DE_BOOL);
        (void)builder->SetPredicateFunc(py_func);
      } else if (key == "input_columns") {
        std::vector<std::string> in_col_names = ToStringVector(args["input_columns"]);
        (void)builder->SetInColNames(in_col_names);
      } else {
        RETURN_STATUS_UNEXPECTED("Error: Unhandled key: " + key);
      }
    }
  }

  std::shared_ptr<FilterOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *top = op;
  return Status::OK();
}

Status DEPipeline::ParseRepeatOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                 std::shared_ptr<DatasetOp> *bottom) {
  if (args["count"].is_none()) {
    std::string err_msg = "Error: count is invalid or not set.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  repeat_num_ = ToInt(args["count"]);
  std::shared_ptr<RepeatOp> op;
  RETURN_IF_NOT_OK(RepeatOp::Builder(ToInt(args["count"])).Build(&op));
  *top = op;
  return Status::OK();
}

Status DEPipeline::ParseSkipOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                               std::shared_ptr<DatasetOp> *bottom) {
  if (args["count"].is_none()) {
    std::string err_msg = "Error: count is invalid or not set.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::shared_ptr<SkipOp> op;
  RETURN_IF_NOT_OK(SkipOp::Builder(ToInt(args["count"])).Build(&op));
  *top = op;
  return Status::OK();
}

Status DEPipeline::ParseEpochCtrlOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                    std::shared_ptr<DatasetOp> *bottom) {
  if (args["count"].is_none()) {
    std::string err_msg = "Error: count is invalid or not set.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::shared_ptr<EpochCtrlOp> op;
  RETURN_IF_NOT_OK(EpochCtrlOp::Builder(ToInt(args["count"])).Build(&op));
  *top = op;
  return Status::OK();
}

Status DEPipeline::ParseGeneratorOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                    std::shared_ptr<DatasetOp> *bottom) {
  std::shared_ptr<GeneratorOp::Builder> builder = std::make_shared<GeneratorOp::Builder>();
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "source") {
        py::object obj = py::cast(&value);
        if (!py::isinstance<py::function>(obj)) {
          std::string err_msg = "Error: generator is invalid or not set.";
          RETURN_STATUS_UNEXPECTED(err_msg);
        }
        (void)builder->SetGeneratorFunction(obj.cast<py::function>());
      } else if (key == "column_names") {
        (void)builder->SetColumnNames(ToStringVector(value));
      } else if (key == "column_types") {
        (void)builder->SetColumnTypes(ToTypeVector(value));
      }
    }
  }
  std::shared_ptr<GeneratorOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *top = op;
  return Status::OK();
}

Status DEPipeline::ParseBatchOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                std::shared_ptr<DatasetOp> *bottom) {
  std::shared_ptr<BatchOp::Builder> builder;
  std::vector<std::string> project_columns;
  if (py::isinstance<py::int_>(args["batch_size"])) {
    batch_size_ = ToInt(args["batch_size"]);
    CHECK_FAIL_RETURN_UNEXPECTED(batch_size_ > 0, "Error: batch_size is invalid.");
    builder = std::make_shared<BatchOp::Builder>(batch_size_);
  } else if (py::isinstance<py::function>(args["batch_size"])) {
    builder = std::make_shared<BatchOp::Builder>(1);
    (void)builder->SetBatchSizeFunc(args["batch_size"].cast<py::function>());
  } else {
    RETURN_STATUS_UNEXPECTED("Error: batch_size is neither an Integer nor a python function.");
  }
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "drop_remainder") {
        (void)builder->SetDrop(ToBool(value));
      } else if (key == "num_parallel_workers") {
        (void)builder->SetNumWorkers(ToInt(value));
      } else if (key == "per_batch_map") {
        (void)builder->SetBatchMapFunc(value.cast<py::function>());
      } else if (key == "input_columns") {
        (void)builder->SetInColNames(ToStringVector(value));
      } else if (key == "output_columns") {
        (void)builder->SetOutColNames(ToStringVector(value));
      } else if (key == "column_order") {
        project_columns = ToStringVector(value);
      } else if (key == "pad_info") {
        PadInfo pad_info;
        RETURN_IF_NOT_OK(ParsePadInfo(value, &pad_info));
        (void)builder->SetPaddingMap(pad_info, true);
      }
    }
  }

  std::shared_ptr<BatchOp> batch_op;
  RETURN_IF_NOT_OK(builder->Build(&batch_op));
  *top = batch_op;

  // Add a project op over top of the batch if the user wanted to reposition the columns after per_batch_map
  if (!project_columns.empty()) {
    ProjectOp::Builder proj_builder(project_columns);
    std::shared_ptr<ProjectOp> proj_op;
    RETURN_IF_NOT_OK(proj_builder.Build(&proj_op));
    RETURN_IF_NOT_OK(tree_->AssociateNode(batch_op));
    RETURN_IF_NOT_OK(tree_->AssociateNode(proj_op));
    RETURN_IF_NOT_OK(proj_op->AddChild(batch_op));
    *top = proj_op;
    *bottom = batch_op;
  }
  return Status::OK();
}

Status DEPipeline::ParseBucketBatchByLengthOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                              std::shared_ptr<DatasetOp> *bottom) {
  std::vector<std::string> mandatory_arguments = {"length_dependent_columns", "bucket_boundaries",
                                                  "bucket_batch_sizes"};
  for (auto name : mandatory_arguments) {
    if (args[name.c_str()].is_none()) {
      std::string err_msg = "Error: " + name + " is not set.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }

  std::shared_ptr<BucketBatchByLengthOp::Builder> builder = std::make_shared<BucketBatchByLengthOp::Builder>(
    ToStringVector(args[mandatory_arguments[0].c_str()]), ToIntVector(args[mandatory_arguments[1].c_str()]),
    ToIntVector(args[mandatory_arguments[2].c_str()]));

  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "length_dependent_columns") {
        (void)builder->SetLengthDependentColumns(ToStringVector(value));
      }
      if (key == "bucket_boundaries") {
        (void)builder->SetBucketBoundaries(ToIntVector(value));
      }
      if (key == "bucket_batch_sizes") {
        (void)builder->SetBucketBatchSizes(ToIntVector(value));
      }
      if (key == "element_length_function") {
        std::shared_ptr<TensorOp> py_func;
        py_func = std::make_shared<PyFuncOp>(value.cast<py::function>(), DataType::DE_INT32);
        (void)builder->SetElementLengthFunction(py_func);
      }
      if (key == "pad_info") {
        PadInfo pad_info;
        RETURN_IF_NOT_OK(ParsePadInfo(value, &pad_info));
        (void)builder->SetPadInfo(pad_info);
      }
      if (key == "pad_to_bucket_boundary") {
        (void)builder->SetPadToBucketBoundary(ToBool(value));
      }
      if (key == "drop_remainder") {
        (void)builder->SetDropRemainder(ToBool(value));
      }
    }
  }

  std::shared_ptr<BucketBatchByLengthOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *top = op;
  return Status::OK();
}

Status DEPipeline::ParseBarrierOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                  std::shared_ptr<DatasetOp> *bottom) {
  std::shared_ptr<BarrierOp::Builder> builder = std::make_shared<BarrierOp::Builder>();
  // Right now barrier should only take num_rows_per_buffer = 1
  // The reason for this is because having it otherwise can lead to blocking issues
  // See barrier_op.h for more details
  (void)builder->SetRowsPerBuffer(1);
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "condition_name") {
        (void)builder->SetConditionName(ToString(value));
      } else if (key == "condition_func") {
        (void)builder->SetConditionFunc(value.cast<py::function>());
      }
    }
  }

  std::shared_ptr<BarrierOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *top = op;
  return Status::OK();
}

Status DEPipeline::ParseDeviceQueueOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                      std::shared_ptr<DatasetOp> *bottom) {
  int32_t prefetch_size = 0;
  if (args.contains("prefetch_size")) {
    if (args["prefetch_size"].is_none()) {
      prefetch_size = 16;
    } else {
      prefetch_size = ToInt(args["prefetch_size"]);
    }
  }
  std::shared_ptr<DeviceQueueOp::Builder> builder = std::make_shared<DeviceQueueOp::Builder>(prefetch_size);
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "queue_name") {
        (void)builder->SetChannelName(ToString(value));
      } else if (key == "device_type") {
        (void)builder->SetDeviceType(ToString(value));
      } else if (key == "device_id") {
        (void)builder->SetDeviceId(ToInt(value));
      } else if (key == "send_epoch_end") {
        (void)builder->SetSendEpochEnd(ToBool(value));
      } else if (key == "total_batch") {
        (void)builder->SetTotalBatch(ToInt(value));
      }
    }
  }
  std::shared_ptr<DeviceQueueOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *top = op;
  return Status::OK();
}

Status DEPipeline::ParseRenameOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                 std::shared_ptr<DatasetOp> *bottom) {
  std::vector<std::string> in_col_names;
  std::vector<std::string> out_col_names;
  std::shared_ptr<RenameOp::Builder> builder = std::make_shared<RenameOp::Builder>();
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "input_columns") {
        in_col_names = ToStringVector(value);
      } else if (key == "output_columns") {
        out_col_names = ToStringVector(value);
      }
    }
  }
  if (in_col_names.empty() || in_col_names[0].empty()) {
    std::string err_msg = "Error: input_column_names is invalid or not set.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  if (out_col_names.empty() || out_col_names[0].empty()) {
    std::string err_msg = "Error: output_column_names is invalid or not set.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  (void)builder->SetInColNames(in_col_names);
  (void)builder->SetOutColNames(out_col_names);
  std::shared_ptr<RenameOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *top = op;
  return Status::OK();
}

Status DEPipeline::ParseTakeOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                               std::shared_ptr<DatasetOp> *bottom) {
  if (args["count"].is_none()) {
    std::string err_msg = "Error: count is invalid or not set.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::shared_ptr<TakeOp> op;
  RETURN_IF_NOT_OK(TakeOp::Builder(ToInt(args["count"])).Build(&op));
  *top = op;
  return Status::OK();
}

Status DEPipeline::ParseZipOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                              std::shared_ptr<DatasetOp> *bottom) {
  std::shared_ptr<ZipOp::Builder> builder = std::make_shared<ZipOp::Builder>();
  std::shared_ptr<ZipOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *top = op;
  return Status::OK();
}

Status DEPipeline::ParseConcatOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                 std::shared_ptr<DatasetOp> *bottom) {
  std::shared_ptr<ConcatOp::Builder> builder = std::make_shared<ConcatOp::Builder>();
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        std::shared_ptr<SamplerRT> sampler = create().cast<std::shared_ptr<SamplerRT>>();
        (void)builder->SetSampler(std::move(sampler));
      }
      if (key == "children_flag_and_nums") {
        auto childFlag = py::reinterpret_borrow<py::list>(value).cast<std::vector<std::pair<int, int>>>();
        (void)builder->SetChildrenFlagAndNums(childFlag);
      }
      if (key == "children_start_end_index") {
        auto childIndex = py::reinterpret_borrow<py::list>(value).cast<std::vector<std::pair<int, int>>>();
        (void)builder->SetChildrenStartEndIndex(childIndex);
      }
    }
  }
  std::shared_ptr<ConcatOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *top = op;
  return Status::OK();
}

Status DEPipeline::ParseTFReaderOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                   std::shared_ptr<DatasetOp> *bottom) {
  // Required arguments
  std::vector<std::string> files_list;
  std::shared_ptr<CacheClient> cache_client = nullptr;
  std::shared_ptr<SamplerRT> sampler = nullptr;
  int num_workers = 0;
  std::shared_ptr<TFReaderOp::Builder> builder = std::make_shared<TFReaderOp::Builder>();
  if (!args["dataset_files"].is_none()) {
    files_list = ToStringVector(args["dataset_files"]);
    (void)builder->SetDatasetFilesList(files_list);
  } else {
    std::string err_msg = "Error: at least one of dataset_files or schema_file is missing";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::vector<std::string> columns_to_load;
  bool schema_exists = false;
  bool shuffle_required = false;
  int64_t num_devices = 0;
  int64_t total_rows = 0;
  // Optional arguments
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        num_workers = ToInt(value);
        (void)builder->SetNumWorkers(num_workers);
      } else if (key == "columns_list") {
        columns_to_load = ToStringVector(value);
        (void)builder->SetColumnsToLoad(columns_to_load);
      } else if (key == "shuffle_files") {
        (void)builder->SetShuffleFiles(ToBool(value));
      } else if (key == "shuffle_global") {
        shuffle_required = ToBool(value);
      } else if (key == "schema_file_path" || key == "schema_json_string") {
        schema_exists = true;
      } else if (key == "num_samples") {
        total_rows = ToInt(value);
        (void)builder->setTotalRows(total_rows);
      } else if (key == "num_shards") {
        num_devices = ToInt(value);
        (void)builder->SetNumDevices(num_devices);
      } else if (key == "shard_id") {
        (void)builder->SetDeviceId(ToInt(value));
      } else if (key == "shard_equal_rows") {
        (void)builder->SetShardEqualRows(ToBool(value));
      } else if (key == "cache") {
        cache_client = value.cast<std::shared_ptr<CacheClient>>();
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        sampler = create().cast<std::shared_ptr<SamplerRT>>();
      }
    }
  }
  if (schema_exists) {
    std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
    if (args.contains("schema_file_path")) {
      RETURN_IF_NOT_OK(schema->LoadSchemaFile(ToString(args["schema_file_path"]), columns_to_load));
    } else {
      RETURN_IF_NOT_OK(schema->LoadSchemaString(ToString(args["schema_json_string"]), columns_to_load));
    }
    (void)builder->SetDataSchema(std::move(schema));
  }

  // If the user gave a sampler, but they did not ask for a cache, then by itself this is not allowed
  // because TFReaderOp is a non-mappable dataset that does not support sampling.
  // However, if a cache operator is injected at some other place higher in the tree, that cache can
  // inherit this sampler from the leaf, providing sampling support from the caching layer.
  // That is why we save the sampler here in a leaf node that does not use sampling.
  if (sampler) {
    (void)builder->SetSampler(std::move(sampler));
  } else if (cache_client) {
    const int64_t num_samples = 0;
    const int64_t start_index = 0;
    sampler = std::make_shared<SequentialSamplerRT>(num_samples, start_index);
    (void)builder->SetSampler(std::move(sampler));
  }

  std::shared_ptr<TFReaderOp> tf_op;
  RETURN_IF_NOT_OK(builder->Build(&tf_op));
  RETURN_IF_NOT_OK(tree_->AssociateNode(tf_op));
  *top = tf_op;

  if (!cache_client && shuffle_required) {
    const boolean estimate = true;
    const int64_t workers = 8;
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t shuffle_size = 0;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset via estimate and then compute the shuffle size
    RETURN_IF_NOT_OK(TFReaderOp::CountTotalRows(&num_rows, files_list, workers, estimate));
    RETURN_IF_NOT_OK(ComputeShuffleSize(files_list.size(), num_devices, num_rows, total_rows, &shuffle_size));

    // Add the shuffle op over top of this op and return the subtree (top/bottom) to caller
    RETURN_IF_NOT_OK(AddShuffleOp(shuffle_size, tf_op, &shuffle_op));
    *top = shuffle_op;
    *bottom = tf_op;
  }

  // Add a cache op over this op if required and update the output subtree (top/bottom)
  if (cache_client) {
    // Note, it is not allowed to have both shuffle and cache
    std::shared_ptr<DatasetOp> cache_op = nullptr;
    RETURN_IF_NOT_OK(AddCacheOp(cache_client, num_workers, tf_op, &cache_op));
    *top = cache_op;
    *bottom = tf_op;
  }

  return Status::OK();
}

Status DEPipeline::ParseProjectOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                  std::shared_ptr<DatasetOp> *bottom) {
  if (args["columns"].is_none()) {
    std::string err_msg = "Error: columns is missing";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::vector<std::string> columns_to_project = ToStringVector(args["columns"]);
  std::shared_ptr<ProjectOp::Builder> builder = std::make_shared<ProjectOp::Builder>(columns_to_project);
  std::shared_ptr<ProjectOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *top = op;
  return Status::OK();
}

Status DEPipeline::ParseImageFolderOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                      std::shared_ptr<DatasetOp> *bottom) {
  // Required arguments
  if (args["dataset_dir"].is_none()) {
    std::string err_msg = "Error: No dataset path specified";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  int num_workers = 0;
  std::shared_ptr<CacheClient> cache_client = nullptr;
  std::shared_ptr<ImageFolderOp::Builder> builder = std::make_shared<ImageFolderOp::Builder>();
  (void)builder->SetImageFolderDir(ToString(args["dataset_dir"]));

  // Optional arguments
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        num_workers = ToInt(value);
        (void)builder->SetNumWorkers(num_workers);
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        std::shared_ptr<SamplerRT> sampler = create().cast<std::shared_ptr<SamplerRT>>();
        (void)builder->SetSampler(std::move(sampler));
      } else if (key == "extensions") {
        (void)builder->SetExtensions(ToStringSet(value));
      } else if (key == "class_indexing") {
        (void)builder->SetClassIndex(ToStringMap(value));
      } else if (key == "decode") {
        (void)builder->SetDecode(ToBool(value));
      } else if (key == "cache") {
        cache_client = value.cast<std::shared_ptr<CacheClient>>();
      }
    }
  }
  std::shared_ptr<ImageFolderOp> if_op;
  RETURN_IF_NOT_OK(builder->Build(&if_op));
  RETURN_IF_NOT_OK(tree_->AssociateNode(if_op));
  *top = if_op;

  // Additionally, add a cache if required.
  // Note that this cache op is only acting as a place holder for the caching position
  // within the tree.  Later, a pre-pass will execute a tree transform to set up the actual
  // caching logic in the tree.
  if (cache_client) {
    std::shared_ptr<DatasetOp> cache_op = nullptr;
    RETURN_IF_NOT_OK(AddCacheOp(cache_client, num_workers, if_op, &cache_op));
    *top = cache_op;
    *bottom = if_op;
  }

  return Status::OK();
}

Status DEPipeline::ParseManifestOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                   std::shared_ptr<DatasetOp> *bottom) {
  // Required arguments
  if (args["dataset_file"].is_none()) {
    std::string err_msg = "Error: No dataset files specified for manifest";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  int num_workers = 0;
  std::shared_ptr<CacheClient> cache_client = nullptr;
  std::shared_ptr<ManifestOp::Builder> builder = std::make_shared<ManifestOp::Builder>();
  (void)builder->SetManifestFile(ToString(args["dataset_file"]));

  // Optional arguments
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        num_workers = ToInt(value);
        (void)builder->SetNumWorkers(num_workers);
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        std::shared_ptr<SamplerRT> sampler = create().cast<std::shared_ptr<SamplerRT>>();
        (void)builder->SetSampler(std::move(sampler));
      } else if (key == "class_indexing") {
        (void)builder->SetClassIndex(ToStringMap(value));
      } else if (key == "decode") {
        (void)builder->SetDecode(ToBool(value));
      } else if (key == "usage") {
        (void)builder->SetUsage(ToString(value));
      } else if (key == "cache") {
        cache_client = value.cast<std::shared_ptr<CacheClient>>();
      }
    }
  }
  std::shared_ptr<ManifestOp> manifest_op;
  RETURN_IF_NOT_OK(builder->Build(&manifest_op));
  RETURN_IF_NOT_OK(tree_->AssociateNode(manifest_op));
  *top = manifest_op;

  // Additionally, add a cache if required.
  // Note that this cache op is only acting as a place holder for the caching position
  // within the tree.  Later, a pre-pass will execute a tree transform to set up the actual
  // caching logic in the tree.
  if (cache_client) {
    std::shared_ptr<DatasetOp> cache_op = nullptr;
    RETURN_IF_NOT_OK(AddCacheOp(cache_client, num_workers, manifest_op, &cache_op));
    *top = cache_op;
    *bottom = manifest_op;
  }

  return Status::OK();
}

Status DEPipeline::ParseVOCOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                              std::shared_ptr<DatasetOp> *bottom) {
  CHECK_FAIL_RETURN_UNEXPECTED(!args["dataset_dir"].is_none(), "Error: No dataset path specified.");
  CHECK_FAIL_RETURN_UNEXPECTED(!args["task"].is_none(), "Error: No task specified.");
  CHECK_FAIL_RETURN_UNEXPECTED(!args["usage"].is_none(), "Error: No usage specified.");

  int num_workers = 0;
  std::shared_ptr<CacheClient> cache_client = nullptr;
  std::shared_ptr<VOCOp::Builder> builder = std::make_shared<VOCOp::Builder>();
  (void)builder->SetDir(ToString(args["dataset_dir"]));
  (void)builder->SetTask(ToString(args["task"]));
  (void)builder->SetUsage(ToString(args["usage"]));
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        num_workers = ToInt(value);
        (void)builder->SetNumWorkers(num_workers);
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        std::shared_ptr<SamplerRT> sampler = create().cast<std::shared_ptr<SamplerRT>>();
        (void)builder->SetSampler(std::move(sampler));
      } else if (key == "decode") {
        (void)builder->SetDecode(ToBool(value));
      } else if (key == "class_indexing") {
        (void)builder->SetClassIndex(ToStringMap(value));
      } else if (key == "cache") {
        cache_client = value.cast<std::shared_ptr<CacheClient>>();
      }
    }
  }
  std::shared_ptr<VOCOp> voc_op;
  RETURN_IF_NOT_OK(builder->Build(&voc_op));
  RETURN_IF_NOT_OK(tree_->AssociateNode(voc_op));
  *top = voc_op;

  // Additionally, add a cache if required.
  // Note that this cache op is only acting as a place holder for the caching position
  // within the tree.  Later, a pre-pass will execute a tree transform to set up the actual
  // caching logic in the tree.
  if (cache_client) {
    std::shared_ptr<DatasetOp> cache_op = nullptr;
    RETURN_IF_NOT_OK(AddCacheOp(cache_client, num_workers, voc_op, &cache_op));
    *top = cache_op;
    *bottom = voc_op;
  }

  return Status::OK();
}

Status DEPipeline::ParseCocoOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                               std::shared_ptr<DatasetOp> *bottom) {
  if (args["dataset_dir"].is_none()) {
    std::string err_msg = "Error: No dataset path specified";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  if (args["annotation_file"].is_none()) {
    std::string err_msg = "Error: No annotation_file specified";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  if (args["task"].is_none()) {
    std::string err_msg = "Error: No task specified";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  int num_workers = 0;
  std::shared_ptr<CacheClient> cache_client = nullptr;
  std::shared_ptr<CocoOp::Builder> builder = std::make_shared<CocoOp::Builder>();
  (void)builder->SetDir(ToString(args["dataset_dir"]));
  (void)builder->SetFile(ToString(args["annotation_file"]));
  (void)builder->SetTask(ToString(args["task"]));
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        num_workers = ToInt(value);
        (void)builder->SetNumWorkers(num_workers);
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        std::shared_ptr<SamplerRT> sampler = create().cast<std::shared_ptr<SamplerRT>>();
        (void)builder->SetSampler(std::move(sampler));
      } else if (key == "decode") {
        (void)builder->SetDecode(ToBool(value));
      } else if (key == "cache") {
        cache_client = value.cast<std::shared_ptr<CacheClient>>();
      }
    }
  }
  std::shared_ptr<CocoOp> coco_op;
  RETURN_IF_NOT_OK(builder->Build(&coco_op));
  RETURN_IF_NOT_OK(tree_->AssociateNode(coco_op));
  *top = coco_op;

  // Additionally, add a cache if required.
  // Note that this cache op is only acting as a place holder for the caching position
  // within the tree.  Later, a pre-pass will execute a tree transform to set up the actual
  // caching logic in the tree.
  if (cache_client) {
    std::shared_ptr<DatasetOp> cache_op = nullptr;
    RETURN_IF_NOT_OK(AddCacheOp(cache_client, num_workers, coco_op, &cache_op));
    *top = cache_op;
    *bottom = coco_op;
  }

  return Status::OK();
}

Status DEPipeline::ParseCifar10Op(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                  std::shared_ptr<DatasetOp> *bottom) {
  // Required arguments
  if (args["dataset_dir"].is_none()) {
    std::string err_msg = "Error: No dataset path specified";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  int num_workers = 0;
  std::shared_ptr<CacheClient> cache_client = nullptr;
  std::shared_ptr<CifarOp::Builder> builder = std::make_shared<CifarOp::Builder>();
  (void)builder->SetCifarDir(ToString(args["dataset_dir"]));

  // Optional arguments
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        num_workers = ToInt(value);
        (void)builder->SetNumWorkers(num_workers);
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        std::shared_ptr<SamplerRT> sampler = create().cast<std::shared_ptr<SamplerRT>>();
        (void)builder->SetSampler(std::move(sampler));
      } else if (key == "usage") {
        (void)builder->SetUsage(ToString(value));
      } else if (key == "cache") {
        cache_client = value.cast<std::shared_ptr<CacheClient>>();
      }
    }
  }

  (void)builder->SetCifarType(true);

  std::shared_ptr<CifarOp> cifar_op;
  RETURN_IF_NOT_OK(builder->Build(&cifar_op));
  RETURN_IF_NOT_OK(tree_->AssociateNode(cifar_op));
  *top = cifar_op;

  // Additionally, add a cache if required.
  // Note that this cache op is only acting as a place holder for the caching position
  // within the tree.  Later, a pre-pass will execute a tree transform to set up the actual
  // caching logic in the tree.
  if (cache_client) {
    std::shared_ptr<DatasetOp> cache_op = nullptr;
    RETURN_IF_NOT_OK(AddCacheOp(cache_client, num_workers, cifar_op, &cache_op));
    *top = cache_op;
    *bottom = cifar_op;
  }

  return Status::OK();
}

Status DEPipeline::ParseCifar100Op(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                   std::shared_ptr<DatasetOp> *bottom) {
  // Required arguments
  if (args["dataset_dir"].is_none()) {
    std::string err_msg = "Error: No dataset path specified";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  int num_workers = 0;
  std::shared_ptr<CacheClient> cache_client = nullptr;
  std::shared_ptr<CifarOp::Builder> builder = std::make_shared<CifarOp::Builder>();
  (void)builder->SetCifarDir(ToString(args["dataset_dir"]));

  // Optional arguments
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        num_workers = ToInt(value);
        (void)builder->SetNumWorkers(num_workers);
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        std::shared_ptr<SamplerRT> sampler = create().cast<std::shared_ptr<SamplerRT>>();
        (void)builder->SetSampler(std::move(sampler));
      } else if (key == "usage") {
        (void)builder->SetUsage(ToString(value));
      } else if (key == "cache") {
        cache_client = value.cast<std::shared_ptr<CacheClient>>();
      }
    }
  }

  (void)builder->SetCifarType(false);

  std::shared_ptr<CifarOp> cifar_op;
  RETURN_IF_NOT_OK(builder->Build(&cifar_op));
  RETURN_IF_NOT_OK(tree_->AssociateNode(cifar_op));
  *top = cifar_op;

  // Additionally, add a cache if required.
  // Note that this cache op is only acting as a place holder for the caching position
  // within the tree.  Later, a pre-pass will execute a tree transform to set up the actual
  // caching logic in the tree.
  if (cache_client) {
    std::shared_ptr<DatasetOp> cache_op = nullptr;
    RETURN_IF_NOT_OK(AddCacheOp(cache_client, num_workers, cifar_op, &cache_op));
    *top = cache_op;
    *bottom = cifar_op;
  }
  return Status::OK();
}

Status DEPipeline::ParseRandomDataOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                     std::shared_ptr<DatasetOp> *bottom) {
  // Required arguments
  RandomDataOp::Builder builder;
  std::shared_ptr<CacheClient> cache_client = nullptr;
  std::shared_ptr<SamplerRT> sampler = nullptr;
  int num_workers = 0;

  if (args["total_rows"].is_none()) {
    std::string err_msg = "Error: total_rows is a required argument";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::vector<std::string> columns_to_load;
  bool schema_exists = false;
  // Optional arguments
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        num_workers = ToInt(value);
        (void)builder.SetNumWorkers(num_workers);
      } else if (key == "schema_file_path" || key == "schema_json_string") {
        schema_exists = true;
      } else if (key == "columns_list") {
        columns_to_load = ToStringVector(value);
      } else if (key == "total_rows") {
        // This is not sampling here. The random data op needs to know how much data to generate.
        (void)builder.SetTotalRows(ToInt(value));
      } else if (key == "cache") {
        cache_client = value.cast<std::shared_ptr<CacheClient>>();
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        sampler = create().cast<std::shared_ptr<SamplerRT>>();
      }
    }
  }
  if (schema_exists) {
    std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
    if (args.contains("schema_file_path")) {
      RETURN_IF_NOT_OK(schema->LoadSchemaFile(ToString(args["schema_file_path"]), columns_to_load));
    } else {
      RETURN_IF_NOT_OK(schema->LoadSchemaString(ToString(args["schema_json_string"]), columns_to_load));
    }
    (void)builder.SetDataSchema(std::move(schema));
  }

  // If the user gave a sampler, but they did not ask for a cache, then by itself this is not allowed
  // because RandomDataOp is a non-mappable dataset that does not support sampling.
  // However, if a cache operator is injected at some other place higher in the tree, that cache can
  // inherit this sampler from the leaf, providing sampling support from the caching layer.
  // That is why we save the sampler here in a leaf node that does not use sampling.
  if (sampler) {
    (void)builder.SetSampler(std::move(sampler));
  } else if (cache_client) {
    const int64_t num_samples = 0;
    const int64_t start_index = 0;
    sampler = std::make_shared<SequentialSamplerRT>(num_samples, start_index);
    (void)builder.SetSampler(std::move(sampler));
  }

  std::shared_ptr<RandomDataOp> random_op = nullptr;
  RETURN_IF_NOT_OK(builder.Build(&random_op));
  RETURN_IF_NOT_OK(tree_->AssociateNode(random_op));
  *top = random_op;

  // Add a cache op over this op if required and update the output subtree (top/bottom)
  if (cache_client) {
    std::shared_ptr<DatasetOp> cache_op = nullptr;
    RETURN_IF_NOT_OK(AddCacheOp(cache_client, num_workers, random_op, &cache_op));
    *top = cache_op;
    *bottom = random_op;
  }

  return Status::OK();
}

int32_t DEPipeline::GetNumClasses() const { return num_classes_; }

Status DEPipeline::ParseMnistOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                std::shared_ptr<DatasetOp> *bottom) {
  // Required arguments
  if (args["dataset_dir"].is_none()) {
    std::string err_msg = "Error: No dataset path specified";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  int num_workers = 0;
  std::shared_ptr<CacheClient> cache_client = nullptr;
  std::shared_ptr<MnistOp::Builder> builder = std::make_shared<MnistOp::Builder>();
  (void)builder->SetDir(ToString(args["dataset_dir"]));

  // Optional arguments
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        num_workers = ToInt(value);
        (void)builder->SetNumWorkers(num_workers);
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        std::shared_ptr<SamplerRT> sampler = create().cast<std::shared_ptr<SamplerRT>>();
        (void)builder->SetSampler(std::move(sampler));
      } else if (key == "usage") {
        (void)builder->SetUsage(ToString(value));
      } else if (key == "cache") {
        cache_client = value.cast<std::shared_ptr<CacheClient>>();
      }
    }
  }
  std::shared_ptr<MnistOp> mnist_op;
  RETURN_IF_NOT_OK(builder->Build(&mnist_op));
  RETURN_IF_NOT_OK(tree_->AssociateNode(mnist_op));
  *top = mnist_op;

  // Additionally, add a cache if required.
  // Note that this cache op is only acting as a place holder for the caching position
  // within the tree.  Later, a pre-pass will execute a tree transform to set up the actual
  // caching logic in the tree.
  if (cache_client) {
    std::shared_ptr<DatasetOp> cache_op = nullptr;
    RETURN_IF_NOT_OK(AddCacheOp(cache_client, num_workers, mnist_op, &cache_op));
    *top = cache_op;
    *bottom = mnist_op;
  }

  return Status::OK();
}

Status DEPipeline::ParseCelebAOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                 std::shared_ptr<DatasetOp> *bottom) {
  // Required arguments
  if (args["dataset_dir"].is_none()) {
    std::string err_msg = "Error: No dataset path specified";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, err_msg);
  }

  int num_workers = 0;
  std::shared_ptr<CacheClient> cache_client = nullptr;
  std::shared_ptr<CelebAOp::Builder> builder = std::make_shared<CelebAOp::Builder>();
  if (builder == nullptr) {
    std::string err_msg = "Create celebaop builder failed";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, err_msg);
  }
  (void)builder->SetCelebADir(ToString(args["dataset_dir"]));
  for (const auto &arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        num_workers = ToInt(value);
        (void)builder->SetNumWorkers(num_workers);
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        std::shared_ptr<SamplerRT> sampler = create().cast<std::shared_ptr<SamplerRT>>();
        (void)builder->SetSampler(std::move(sampler));
      } else if (key == "decode") {
        (void)builder->SetDecode(ToBool(value));
      } else if (key == "extensions") {
        (void)builder->SetExtensions(ToStringSet(value));
      } else if (key == "usage") {
        (void)builder->SetUsage(ToString(value));
      } else if (key == "cache") {
        cache_client = value.cast<std::shared_ptr<CacheClient>>();
      }
    }
  }

  std::shared_ptr<CelebAOp> celeba_op;
  RETURN_IF_NOT_OK(builder->Build(&celeba_op));
  RETURN_IF_NOT_OK(tree_->AssociateNode(celeba_op));
  *top = celeba_op;

  // Additionally, add a cache if required.
  // Note that this cache op is only acting as a place holder for the caching position
  // within the tree.  Later, a pre-pass will execute a tree transform to set up the actual
  // caching logic in the tree.
  if (cache_client) {
    std::shared_ptr<DatasetOp> cache_op = nullptr;
    RETURN_IF_NOT_OK(AddCacheOp(cache_client, num_workers, celeba_op, &cache_op));
    *top = cache_op;
    *bottom = celeba_op;
  }

  return Status::OK();
}

Status DEPipeline::ParseTextFileOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                   std::shared_ptr<DatasetOp> *bottom) {
  // Required arguments
  std::vector<std::string> files_list;
  std::shared_ptr<CacheClient> cache_client = nullptr;
  std::shared_ptr<SamplerRT> sampler = nullptr;
  int num_workers = 0;
  std::shared_ptr<TextFileOp::Builder> builder = std::make_shared<TextFileOp::Builder>();
  if (!args["dataset_files"].is_none()) {
    files_list = ToStringVector(args["dataset_files"]);
    (void)builder->SetTextFilesList(files_list);
  } else {
    RETURN_STATUS_UNEXPECTED("Error: dataset_files is missing");
  }
  // Optional arguments
  bool shuffle_required = false;
  int64_t num_devices = 0;
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        num_workers = ToInt(value);
        (void)builder->SetNumWorkers(num_workers);
      } else if (key == "shuffle_files") {
        (void)builder->SetShuffleFiles(ToBool(value));
      } else if (key == "shuffle_global") {
        shuffle_required = ToBool(value);
      } else if (key == "num_samples") {
        (void)builder->SetTotalRows(ToInt(value));
      } else if (key == "num_shards") {
        num_devices = ToInt(value);
        (void)builder->SetNumDevices(num_devices);
      } else if (key == "shard_id") {
        (void)builder->SetDeviceId(ToInt(value));
      } else if (key == "cache") {
        cache_client = value.cast<std::shared_ptr<CacheClient>>();
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        sampler = create().cast<std::shared_ptr<SamplerRT>>();
      }
    }
  }

  // If the user gave a sampler, but they did not ask for a cache, then by itself this is not allowed
  // because TextFileOp is a non-mappable dataset that does not support sampling.
  // However, if a cache operator is injected at some other place higher in the tree, that cache can
  // inherit this sampler from the leaf, providing sampling support from the caching layer.
  // That is why we save the sampler here in a leaf node that does not use sampling.
  if (sampler) {
    (void)builder->SetSampler(std::move(sampler));
  } else if (cache_client) {
    int64_t num_samples = 0;
    int64_t start_index = 0;
    sampler = std::make_shared<SequentialSamplerRT>(num_samples, start_index);
    (void)builder->SetSampler(std::move(sampler));
  }

  std::shared_ptr<TextFileOp> txt_op;
  RETURN_IF_NOT_OK(builder->Build(&txt_op));
  RETURN_IF_NOT_OK(tree_->AssociateNode(txt_op));
  *top = txt_op;

  if (!cache_client && shuffle_required) {
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t shuffle_size = 0;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset and then compute the shuffle size
    RETURN_IF_NOT_OK(TextFileOp::CountAllFileRows(files_list, &num_rows));
    RETURN_IF_NOT_OK(ComputeShuffleSize(files_list.size(), num_devices, num_rows, 0, &shuffle_size));

    // Add the shuffle op over top of this op and return the subtree (top/bottom) to caller
    RETURN_IF_NOT_OK(AddShuffleOp(shuffle_size, txt_op, &shuffle_op));
    *top = shuffle_op;
    *bottom = txt_op;
  }

  // Add a cache op over this op if required and update the output subtree (top/bottom)
  if (cache_client) {
    // Note, it is not allowed to have both shuffle and cache
    std::shared_ptr<DatasetOp> cache_op = nullptr;
    RETURN_IF_NOT_OK(AddCacheOp(cache_client, num_workers, txt_op, &cache_op));
    *top = cache_op;
    *bottom = txt_op;
  }

  return Status::OK();
}

Status DEPipeline::ParsePadInfo(py::handle value, PadInfo *pad_info) {
  for (auto p : py::reinterpret_borrow<py::dict>(value)) {
    if (!p.second.is_none()) {
      auto tp = py::reinterpret_borrow<py::tuple>(p.second);
      CHECK_FAIL_RETURN_UNEXPECTED(tp.size() == 2, "tuple in pad_info must be (list,int) or (list,float)");
      TensorShape shape = tp[0].is_none() ? TensorShape::CreateUnknownRankShape() : TensorShape(tp[0]);
      std::shared_ptr<Tensor> pad_val = nullptr;
      if (py::isinstance<py::str>(tp[1])) {
        std::string pad_val_string = tp[1].is_none() ? "" : ToString(tp[1]);
        CHECK_FAIL_RETURN_UNEXPECTED(
          Tensor::CreateFromVector(std::vector<std::string>{pad_val_string}, TensorShape::CreateScalar(), &pad_val),
          "Cannot create pad_value Tensor");
      } else {
        float pad_val_float = tp[1].is_none() ? 0 : ToFloat(tp[1]);
        CHECK_FAIL_RETURN_UNEXPECTED(
          Tensor::CreateEmpty(TensorShape::CreateScalar(), DataType(DataType::DE_FLOAT32), &pad_val),
          "Cannot create pad_value Tensor");
        pad_val->SetItemAt<float>({}, pad_val_float);
      }
      (void)pad_info->insert({ToString(p.first), {shape, pad_val}});
    } else {  // tuple is None
      (void)pad_info->insert({ToString(p.first), {TensorShape({}), nullptr}});
    }
  }
  return Status::OK();
}

Status DEPipeline::ParseBuildVocabOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                     std::shared_ptr<DatasetOp> *bottom) {
  std::shared_ptr<BuildVocabOp::Builder> builder = std::make_shared<BuildVocabOp::Builder>();
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "freq_range") {
        py::tuple tp = py::reinterpret_borrow<py::tuple>(value);
        if (!tp[0].is_none()) (void)builder->SetMinFreq(py::reinterpret_borrow<py::int_>(tp[0]));
        if (!tp[1].is_none()) (void)builder->SetMaxFreq(py::reinterpret_borrow<py::int_>(tp[1]));
      } else if (key == "top_k") {
        builder->SetTopK(py::reinterpret_borrow<py::int_>(value));
      } else if (key == "columns") {
        (void)builder->SetColumnNames(ToStringVector(value));
      } else if (key == "vocab") {
        (void)builder->SetVocab(value.cast<std::shared_ptr<Vocab>>());
      } else if (key == "num_parallel_workers") {
        (void)builder->SetNumWorkers(ToInt(value));
      } else if (key == "special_first") {
        (void)builder->SetSpecialFirst(ToBool(value));
      } else if (key == "special_tokens") {
        (void)builder->SetSpecialTokens(ToStringVector(value));
      }
    }
  }
  std::shared_ptr<BuildVocabOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *top = op;
  return Status::OK();
}

Status DEPipeline::ParseBuildSentencePieceVocabOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                                  std::shared_ptr<DatasetOp> *bottom) {
  std::shared_ptr<BuildSentencePieceVocabOp::Builder> builder = std::make_shared<BuildSentencePieceVocabOp::Builder>();
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "vocab_size") {
        builder->SetVocabSize(ToInt(value));
      } else if (key == "columns") {
        (void)builder->SetColumnNames(ToStringVector(value));
      } else if (key == "character_coverage") {
        (void)builder->SetCharacterCoverage(ToFloat(value));
      } else if (key == "params") {
        std::unordered_map<std::string, std::string> params;
        for (auto param : py::reinterpret_borrow<py::dict>(value)) {
          std::string param_key = py::reinterpret_borrow<py::str>(param.first);
          if (param_key == "input" || param_key == "vocab_size" || param_key == "model_prefix" ||
              param_key == "character_coverage" || param_key == "model_type") {
            continue;
          }
          params[param_key] = py::reinterpret_borrow<py::str>(param.second);
        }
        (void)builder->SetParams(params);
      } else if (key == "vocab") {
        (void)builder->SetVocab(value.cast<std::shared_ptr<SentencePieceVocab>>());
      } else if (key == "model_type") {
        (void)builder->SetModelType(value.cast<SentencePieceModel>());
      }
    }
  }
  std::shared_ptr<BuildSentencePieceVocabOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *top = op;
  return Status::OK();
}

Status DEPipeline::ParseClueOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                               std::shared_ptr<DatasetOp> *bottom) {
  std::vector<std::string> files_list;
  std::shared_ptr<CacheClient> cache_client = nullptr;
  std::shared_ptr<SamplerRT> sampler = nullptr;
  int num_workers = 0;

  std::shared_ptr<ClueOp::Builder> builder = std::make_shared<ClueOp::Builder>();
  if (!args["dataset_files"].is_none()) {
    files_list = ToStringVector(args["dataset_files"]);
    (void)builder->SetClueFilesList(files_list);
  } else {
    RETURN_STATUS_UNEXPECTED("Error: dataset_files is missing");
  }
  // Optional arguments
  bool shuffle_required = false;
  int64_t num_devices = 0;
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        num_workers = ToInt(value);
        (void)builder->SetNumWorkers(num_workers);
      } else if (key == "shuffle_files") {
        (void)builder->SetShuffleFiles(ToBool(value));
      } else if (key == "shuffle_global") {
        shuffle_required = ToBool(value);
      } else if (key == "num_samples") {
        (void)builder->SetNumSamples(ToInt(value));
      } else if (key == "num_shards") {
        num_devices = ToInt(value);
        (void)builder->SetNumDevices(num_devices);
      } else if (key == "shard_id") {
        (void)builder->SetDeviceId(ToInt(value));
      } else if (key == "cols_to_keyword") {
        std::map<std::string, std::string> map_dict;
        for (auto p : py::reinterpret_borrow<py::dict>(value)) {
          if (!p.second.is_none()) {
            map_dict.insert({ToString(p.first), ToString(p.second)});
          } else {
            map_dict.insert({ToString(p.first), ToString(p.first)});
          }
        }
        (void)builder->SetColsKeyMap(map_dict);
      } else if (key == "cache") {
        cache_client = value.cast<std::shared_ptr<CacheClient>>();
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        sampler = create().cast<std::shared_ptr<SamplerRT>>();
      }
    }
  }

  // If the user gave a sampler, but they did not ask for a cache, then by itself this is not allowed
  // because ClueOp is a non-mappable dataset that does not support sampling.
  // However, if a cache operator is injected at some other place higher in the tree, that cache can
  // inherit this sampler from the leaf, providing sampling support from the caching layer.
  // That is why we save the sampler here in a leaf node that does not use sampling.
  if (sampler) {
    (void)builder->SetSampler(std::move(sampler));
  } else if (cache_client) {
    int64_t num_samples = 0;
    int64_t start_index = 0;
    sampler = std::make_shared<SequentialSamplerRT>(num_samples, start_index);
    (void)builder->SetSampler(std::move(sampler));
  }

  std::shared_ptr<ClueOp> clue_op;
  RETURN_IF_NOT_OK(builder->Build(&clue_op));
  RETURN_IF_NOT_OK(tree_->AssociateNode(clue_op));
  *top = clue_op;

  if (!cache_client && shuffle_required) {
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t shuffle_size = 0;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset and then compute the shuffle size
    RETURN_IF_NOT_OK(ClueOp::CountAllFileRows(files_list, &num_rows));
    RETURN_IF_NOT_OK(ComputeShuffleSize(files_list.size(), num_devices, num_rows, 0, &shuffle_size));

    // Add the shuffle op over top of this op and return the subtree (top/bottom) to caller
    RETURN_IF_NOT_OK(AddShuffleOp(shuffle_size, clue_op, &shuffle_op));
    *top = shuffle_op;
    *bottom = clue_op;
  }

  // Add a cache op over this op if required and update the output subtree (top/bottom)
  if (cache_client) {
    // Note, it is not allowed to have both shuffle and cache
    std::shared_ptr<DatasetOp> cache_op = nullptr;
    RETURN_IF_NOT_OK(AddCacheOp(cache_client, num_workers, clue_op, &cache_op));
    *top = cache_op;
    *bottom = clue_op;
  }

  return Status::OK();
}

// Helper function to inject the cache operator over top of the current operation being built.
Status DEPipeline::AddCacheOp(std::shared_ptr<CacheClient> cache_client, int num_workers,
                              std::shared_ptr<DatasetOp> input_op, std::shared_ptr<DatasetOp> *cache_op) {
  std::shared_ptr<CacheOp> new_cache_op = nullptr;
  CacheOp::Builder cache_builder;
  // use the same number of workers as the leaf. We need some optimization here, the user does not
  // give the cache op number of workers directly.
  if (num_workers != 0) {
    (void)cache_builder.SetNumWorkers(num_workers);
  }
  (void)cache_builder.SetClient(cache_client);
  RETURN_IF_NOT_OK(cache_builder.Build(&new_cache_op));
  RETURN_IF_NOT_OK(tree_->AssociateNode(new_cache_op));
  RETURN_IF_NOT_OK(new_cache_op->AddChild(input_op));
  // We have now created:
  //
  // CacheOp
  //   |
  // input_op
  //
  *cache_op = new_cache_op;

  return Status::OK();
}

Status DEPipeline::ParseCsvOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                              std::shared_ptr<DatasetOp> *bottom) {
  std::vector<std::string> files_list;
  std::shared_ptr<CacheClient> cache_client = nullptr;
  std::shared_ptr<SamplerRT> sampler = nullptr;
  int num_workers = 0;
  std::shared_ptr<CsvOp::Builder> builder = std::make_shared<CsvOp::Builder>();
  if (!args["dataset_files"].is_none()) {
    files_list = ToStringVector(args["dataset_files"]);
    (void)builder->SetCsvFilesList(files_list);
  } else {
    RETURN_STATUS_UNEXPECTED("Error: dataset_files is missing");
  }

  // Optional arguments
  bool shuffle_required = false;
  int64_t num_devices = 0;
  std::vector<std::string> col_names;
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        num_workers = ToInt(value);
        (void)builder->SetNumWorkers(num_workers);
      } else if (key == "shuffle_files") {
        (void)builder->SetShuffleFiles(ToBool(value));
      } else if (key == "shuffle_global") {
        shuffle_required = ToBool(value);
      } else if (key == "num_samples") {
        (void)builder->SetNumSamples(ToInt(value));
      } else if (key == "num_shards") {
        num_devices = ToInt(value);
        (void)builder->SetNumDevices(num_devices);
      } else if (key == "shard_id") {
        (void)builder->SetDeviceId(ToInt(value));
      } else if (key == "field_delim") {
        (void)builder->SetFieldDelim(ToString(value)[0]);
      } else if (key == "column_defaults") {
        py::list py_object_list = py::reinterpret_borrow<py::list>(value);
        std::vector<std::shared_ptr<CsvOp::BaseRecord>> column_default_list;
        for (auto l : py_object_list) {
          std::string type_s = (std::string)py::str(l.get_type().attr("__name__"));
          if (type_s == "int") {
            column_default_list.push_back(std::make_shared<CsvOp::Record<int>>(CsvOp::INT, ToInt(l)));
          } else if (type_s == "float") {
            column_default_list.push_back(std::make_shared<CsvOp::Record<float>>(CsvOp::FLOAT, ToFloat(l)));
          } else if (type_s == "str") {
            column_default_list.push_back(std::make_shared<CsvOp::Record<std::string>>(CsvOp::STRING, ToString(l)));
          } else {
            RETURN_STATUS_UNEXPECTED("Record type is not allowed");
          }
        }
        (void)builder->SetColumDefault(column_default_list);
      } else if (key == "column_names") {
        col_names = ToStringVector(value);
        (void)builder->SetColumName(col_names);
      } else if (key == "cache") {
        cache_client = value.cast<std::shared_ptr<CacheClient>>();
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        sampler = create().cast<std::shared_ptr<SamplerRT>>();
      }
    }
  }

  // If the user gave a sampler, but they did not ask for a cache, then by itself this is not allowed
  // because CsvOp is a non-mappable dataset that does not support sampling.
  // However, if a cache operator is injected at some other place higher in the tree, that cache can
  // inherit this sampler from the leaf, providing sampling support from the caching layer.
  // That is why we save the sampler here in a leaf node that does not use sampling.
  if (sampler) {
    (void)builder->SetSampler(std::move(sampler));
  } else if (cache_client) {
    int64_t num_samples = 0;
    int64_t start_index = 0;
    sampler = std::make_shared<SequentialSamplerRT>(num_samples, start_index);
    (void)builder->SetSampler(std::move(sampler));
  }

  std::shared_ptr<CsvOp> csv_op;
  RETURN_IF_NOT_OK(builder->Build(&csv_op));
  RETURN_IF_NOT_OK(tree_->AssociateNode(csv_op));
  *top = csv_op;

  if (!cache_client && shuffle_required) {
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t shuffle_size = 0;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset and then compute the shuffle size
    RETURN_IF_NOT_OK(CsvOp::CountAllFileRows(files_list, col_names.empty(), &num_rows));
    RETURN_IF_NOT_OK(ComputeShuffleSize(files_list.size(), num_devices, num_rows, 0, &shuffle_size));

    // Add the shuffle op over top of this op and return the subtree (top/bottom) to caller
    RETURN_IF_NOT_OK(AddShuffleOp(shuffle_size, csv_op, &shuffle_op));
    *top = shuffle_op;
    *bottom = csv_op;
  }

  // Add a cache op over this op if required and update the output subtree (top/bottom)
  if (cache_client) {
    // Note, it is not allowed to have both shuffle and cache
    std::shared_ptr<DatasetOp> cache_op = nullptr;
    RETURN_IF_NOT_OK(AddCacheOp(cache_client, num_workers, csv_op, &cache_op));
    *top = cache_op;
    *bottom = csv_op;
  }

  return Status::OK();
}

// Helper function to inject a shuffle operator over top of the current operation being built.
Status DEPipeline::AddShuffleOp(int64_t shuffle_size, std::shared_ptr<DatasetOp> input_op,
                                std::shared_ptr<DatasetOp> *shuffle_op) {
  std::shared_ptr<ShuffleOp> new_shuffle_op = nullptr;
  ShuffleOp::Builder shuffle_builder;

  (void)shuffle_builder.SetShuffleSize(shuffle_size);
  RETURN_IF_NOT_OK(shuffle_builder.Build(&new_shuffle_op));
  RETURN_IF_NOT_OK(tree_->AssociateNode(new_shuffle_op));
  RETURN_IF_NOT_OK(new_shuffle_op->AddChild(input_op));
  // We have now created:
  //
  // ShuffleOp
  //    |
  // input_op
  //
  *shuffle_op = new_shuffle_op;

  return Status::OK();
}

// Common code for computing a default shuffle size
Status DEPipeline::ComputeShuffleSize(int64_t num_files, int64_t num_devices, int64_t num_rows, int64_t total_rows,
                                      int64_t *shuffle_size) {
  const int64_t average_files_multiplier = 4;
  const int64_t shuffle_max = 10000;
  int64_t avg_rows_per_file = 0;

  // Adjust the num rows per shard if sharding was given
  if (num_devices > 0) {
    if (num_rows % num_devices == 0) {
      num_rows = num_rows / num_devices;
    } else {
      num_rows = (num_rows / num_devices) + 1;
    }
  }

  // Cap based on total rows directive.  Some ops do not have this and give value of 0.
  if (total_rows > 0) {
    num_rows = std::min(num_rows, total_rows);
  }

  // get the average per file
  avg_rows_per_file = num_rows / num_files;

  *shuffle_size = std::max(avg_rows_per_file * average_files_multiplier, shuffle_max);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
