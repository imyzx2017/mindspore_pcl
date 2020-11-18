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

#include "tools/net_train/net_train.h"
#define __STDC_FORMAT_MACROS
#include <cinttypes>
#undef __STDC_FORMAT_MACROS
#include <algorithm>
#include <utility>
#include "src/common/common.h"
#include "include/ms_tensor.h"
#include "include/context.h"
#include "src/runtime/runtime_api.h"
#include "include/version.h"

namespace mindspore {
namespace lite {
static const char *DELIM_COLON = ":";
static const char *DELIM_COMMA = ",";
static const char *DELIM_SLASH = "/";

void SaveFile(std::string path, void *buf, size_t size) {
  std::ofstream ofs(path);
  assert(true == ofs.good());
  assert(true == ofs.is_open());

  ofs.seekp(0, std::ios::beg);
  ofs.write((const char *)buf, size);
  ofs.close();
}

int NetTrain::GenerateRandomData(size_t size, void *data) {
  MS_ASSERT(data != nullptr);
  char *casted_data = static_cast<char *>(data);
  for (size_t i = 0; i < size; i++) {
    casted_data[i] = static_cast<char>(i);
  }
  return RET_OK;
}

int NetTrain::GenerateInputData() {
  for (auto tensor : ms_inputs_) {
    MS_ASSERT(tensor != nullptr);
    auto input_data = tensor->MutableData();
    if (input_data == nullptr) {
      MS_LOG(ERROR) << "MallocData for inTensor failed";
      return RET_ERROR;
    }
    auto tensor_byte_size = tensor->Size();
    auto status = GenerateRandomData(tensor_byte_size, input_data);
    if (status != 0) {
      std::cerr << "GenerateRandomData for inTensor failed: " << status << std::endl;
      MS_LOG(ERROR) << "GenerateRandomData for inTensor failed:" << status;
      return status;
    }
  }
  return RET_OK;
}

int NetTrain::LoadInput() {
  if (flags_->in_data_file_.empty()) {
    auto status = GenerateInputData();
    if (status != 0) {
      std::cerr << "Generate input data error " << status << std::endl;
      MS_LOG(ERROR) << "Generate input data error " << status;
      return status;
    }
  } else {
    auto status = ReadInputFile();
    if (status != 0) {
      std::cerr << "ReadInputFile error, " << status << std::endl;
      MS_LOG(ERROR) << "ReadInputFile error, " << status;
      return status;
    }
  }
  return RET_OK;
}

int NetTrain::ReadInputFile() {
  if (ms_inputs_.empty()) {
    return RET_OK;
  }

  if (this->flags_->in_data_type_ == kImage) {
    MS_LOG(ERROR) << "Not supported image input";
    return RET_ERROR;
  } else {
    for (size_t i = 0; i < flags_->input_data_list_.size(); i++) {
      auto cur_tensor = ms_inputs_.at(i);
      MS_ASSERT(cur_tensor != nullptr);
      size_t size;
      char *bin_buf = ReadFile(flags_->input_data_list_[i].c_str(), &size);
      if (bin_buf == nullptr) {
        MS_LOG(ERROR) << "ReadFile return nullptr";
        return RET_ERROR;
      }
      auto tensor_data_size = cur_tensor->Size();
      if (size != tensor_data_size) {
        std::cerr << "Input binary file size error, required: " << tensor_data_size << ", in fact: " << size
                  << std::endl;
        MS_LOG(ERROR) << "Input binary file size error, required: " << tensor_data_size << ", in fact: " << size;
        delete bin_buf;
        return RET_ERROR;
      }
      auto input_data = cur_tensor->MutableData();
      memcpy(input_data, bin_buf, tensor_data_size);
      delete[](bin_buf);
    }
  }
  return RET_OK;
}

// calibData is FP32
int NetTrain::ReadCalibData() {
  const char *calib_data_path = flags_->data_file_.c_str();
  // read calib data
  std::ifstream in_file(calib_data_path);
  if (!in_file.good()) {
    std::cerr << "file: " << calib_data_path << " is not exist" << std::endl;
    MS_LOG(ERROR) << "file: " << calib_data_path << " is not exist";
    return RET_ERROR;
  }

  if (!in_file.is_open()) {
    std::cerr << "file: " << calib_data_path << " open failed" << std::endl;
    MS_LOG(ERROR) << "file: " << calib_data_path << " open failed";
    in_file.close();
    return RET_ERROR;
  }

  std::string line;

  MS_LOG(INFO) << "Start reading calibData file";
  std::string tensor_name;
  while (!in_file.eof()) {
    getline(in_file, line);
    std::stringstream string_line1(line);
    size_t dim = 0;
    string_line1 >> tensor_name >> dim;
    std::vector<size_t> dims;
    size_t shape_size = 1;
    for (size_t i = 0; i < dim; i++) {
      size_t tmp_dim;
      string_line1 >> tmp_dim;
      dims.push_back(tmp_dim);
      shape_size *= tmp_dim;
    }

    getline(in_file, line);
    std::stringstream string_line2(line);
    std::vector<float> tensor_data;
    for (size_t i = 0; i < shape_size; i++) {
      float tmp_data;
      string_line2 >> tmp_data;
      tensor_data.push_back(tmp_data);
    }
    auto *check_tensor = new CheckTensor(dims, tensor_data);
    this->data_.insert(std::make_pair(tensor_name, check_tensor));
  }
  in_file.close();
  MS_LOG(INFO) << "Finish reading calibData file";
  return RET_OK;
}

int NetTrain::CompareOutput() {
  std::cout << "================ Comparing Output data ================" << std::endl;
  float total_bias = 0;
  int total_size = 0;
  bool has_error = false;

  for (const auto &calib_tensor : data_) {
    std::string node_or_tensor_name = calib_tensor.first;
    auto tensors = session_->GetOutputsByNodeName(node_or_tensor_name);
    mindspore::tensor::MSTensor *tensor = nullptr;
    if (tensors.empty() || tensors.size() != 1) {
      MS_LOG(INFO) << "Cannot find output node: " << node_or_tensor_name
                   << " or node has more than one output tensor, switch to GetOutputByTensorName";
      tensor = session_->GetOutputByTensorName(node_or_tensor_name);
      if (tensor == nullptr) {
        MS_LOG(ERROR) << "Cannot find output tensor " << node_or_tensor_name << ", get model output failed";
        return RET_ERROR;
      }
    } else {
      tensor = tensors.front();
    }
    MS_ASSERT(tensor->MutableData() != nullptr);
    auto outputs = tensor->MutableData();
    float bias = CompareData<float>(node_or_tensor_name, tensor->shape(), reinterpret_cast<float *>(outputs));

    if (bias >= 0) {
      total_bias += bias;
      total_size++;
    } else {
      has_error = true;
      break;
    }
  }

  if (!has_error) {
    float mean_bias;
    if (total_size != 0) {
      mean_bias = total_bias / total_size * 100;
    } else {
      mean_bias = 0;
    }

    std::cout << "Mean bias of all nodes/tensors: " << mean_bias << "%" << std::endl;
    std::cout << "=======================================================" << std::endl << std::endl;

    if (mean_bias > this->flags_->accuracy_threshold_) {
      MS_LOG(ERROR) << "Mean bias of all nodes/tensors is too big: " << mean_bias << "%";
      std::cerr << "Mean bias of all nodes/tensors is too big: " << mean_bias << "%" << std::endl;
      return RET_ERROR;
    } else {
      return RET_OK;
    }
  } else {
    MS_LOG(ERROR) << "Error in CompareData";
    std::cerr << "Error in CompareData" << std::endl;
    std::cout << "=======================================================" << std::endl << std::endl;
    return RET_ERROR;
  }
}

int NetTrain::MarkPerformance() {
  MS_LOG(INFO) << "Running train loops...";
  std::cout << "Running train loops..." << std::endl;
  uint64_t time_min = 1000000;
  uint64_t time_max = 0;
  uint64_t time_avg = 0;

  for (int i = 0; i < flags_->epochs_; i++) {
    session_->BindThread(true);
    auto start = GetTimeUs();
    auto status =
      flags_->time_profiling_ ? session_->RunGraph(before_call_back_, after_call_back_) : session_->RunGraph();
    if (status != 0) {
      MS_LOG(ERROR) << "Inference error " << status;
      std::cerr << "Inference error " << status;
      return status;
    }

    auto end = GetTimeUs();
    auto time = end - start;
    time_min = std::min(time_min, time);
    time_max = std::max(time_max, time);
    time_avg += time;
    session_->BindThread(false);
  }

  if (flags_->time_profiling_) {
    const std::vector<std::string> per_op_name = {"opName", "avg(ms)", "percent", "calledTimes", "opTotalTime"};
    const std::vector<std::string> per_op_type = {"opType", "avg(ms)", "percent", "calledTimes", "opTotalTime"};
    PrintResult(per_op_name, op_times_by_name_);
    PrintResult(per_op_type, op_times_by_type_);
  }

  if (flags_->epochs_ > 0) {
    time_avg /= flags_->epochs_;
    MS_LOG(INFO) << "Model = " << flags_->model_file_.substr(flags_->model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
                 << ", NumThreads = " << flags_->num_threads_ << ", MinRunTime = " << time_min / 1000.0f
                 << ", MaxRuntime = " << time_max / 1000.0f << ", AvgRunTime = " << time_avg / 1000.0f;
    printf("Model = %s, NumThreads = %d, MinRunTime = %f ms, MaxRuntime = %f ms, AvgRunTime = %f ms\n",
           flags_->model_file_.substr(flags_->model_file_.find_last_of(DELIM_SLASH) + 1).c_str(), flags_->num_threads_,
           time_min / 1000.0f, time_max / 1000.0f, time_avg / 1000.0f);
  }
  return RET_OK;
}

int NetTrain::MarkAccuracy() {
  MS_LOG(INFO) << "MarkAccuracy";
  std::cout << "MarkAccuracy" << std::endl;
  for (auto &msInput : ms_inputs_) {
    switch (msInput->data_type()) {
      case TypeId::kNumberTypeFloat:
        PrintInputData<float>(msInput);
        break;
      case TypeId::kNumberTypeFloat32:
        PrintInputData<float>(msInput);
        break;
      case TypeId::kNumberTypeInt32:
        PrintInputData<int>(msInput);
        break;
      default:
        MS_LOG(ERROR) << "Datatype " << msInput->data_type() << " is not supported.";
        return RET_ERROR;
    }
  }
  session_->Eval();

  auto status = session_->RunGraph();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Inference error " << status;
    std::cerr << "Inference error " << status << std::endl;
    return status;
  }

  status = ReadCalibData();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Read calib data error " << status;
    std::cerr << "Read calib data error " << status << std::endl;
    return status;
  }

  status = CompareOutput();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Compare output error " << status;
    std::cerr << "Compare output error " << status << std::endl;
    return status;
  }
  return RET_OK;
}

int NetTrain::RunExportedNet() {
  auto start_prepare_time = GetTimeUs();
  // Load graph
  std::string model_name = flags_->export_file_.substr(flags_->export_file_.find_last_of(DELIM_SLASH) + 1);

  MS_LOG(INFO) << "start reading exported model file";
  std::cout << "start reading exported model file" << std::endl;
  size_t size = 0;
  char *graph_buf = ReadFile(flags_->export_file_.c_str(), &size);
  if (graph_buf == nullptr) {
    MS_LOG(ERROR) << "Read exported model file failed while running " << model_name.c_str();
    std::cerr << "Read exported model file failed while running " << model_name.c_str() << std::endl;
    return RET_ERROR;
  }
  auto model = lite::TrainModel::Import(graph_buf, size);
  delete[](graph_buf);
  if (model == nullptr) {
    MS_LOG(ERROR) << "Import exported model file failed while running " << model_name.c_str();
    std::cerr << "Import exported model file failed while running " << model_name.c_str() << std::endl;
    return RET_ERROR;
  }
  auto context = std::make_shared<Context>();
  if (context == nullptr) {
    MS_LOG(ERROR) << "New context failed while running " << model_name.c_str();
    std::cerr << "New context failed while running " << model_name.c_str() << std::endl;
    return RET_ERROR;
  }

  if (flags_->cpu_bind_mode_ == 2) {
    context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = MID_CPU;
  } else if (flags_->cpu_bind_mode_ == 1) {
    context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = HIGHER_CPU;
  } else {
    context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = NO_BIND;
  }

  context->thread_num_ = flags_->num_threads_;
  // context->enable_float16_ = flags_->enable_fp16_;
  session_ = session::TrainSession::CreateSession(context.get());
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "CreateSession failed while running ", model_name.c_str();
    std::cout << "CreateSession failed while running ", model_name.c_str();
    return RET_ERROR;
  }
  auto ret = session_->CompileTrainGraph(model);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CompileGraph failed while running ", model_name.c_str();
    std::cout << "CompileGraph failed while running ", model_name.c_str();
    return ret;
  }

  ms_inputs_ = session_->GetInputs();
  auto end_prepare_time = GetTimeUs();
  MS_LOG(INFO) << "Exported model PrepareTime = " << (end_prepare_time - start_prepare_time) / 1000 << " ms";
  std::cout << "Exported model PrepareTime = " << (end_prepare_time - start_prepare_time) / 1000 << " ms" << std::endl;

  // Load input
  MS_LOG(INFO) << "start generate input data";
  auto status = LoadInput();
  if (status != 0) {
    MS_LOG(ERROR) << "Generate input data error";
    return status;
  }

  status = session_->RunGraph();
  if (status != 0) {
    MS_LOG(ERROR) << "Inference error " << status;
    std::cerr << "Inference error " << status << std::endl;
    return status;
  }

  if (!flags_->data_file_.empty()) {
    MS_LOG(INFO) << "Check accuracy for exported model";
    std::cout << "Check accuracy for exported model " << std::endl;
    status = MarkAccuracy();
    for (auto &data : data_) {
      data.second->shape.clear();
      data.second->data.clear();
      delete data.second;
    }
    data_.clear();
    if (status != 0) {
      MS_LOG(ERROR) << "Run MarkAccuracy on exported model error: " << status;
      std::cout << "Run MarkAccuracy on exported model error: " << status << std::endl;
      return status;
    }
  }
  return RET_OK;
}

int NetTrain::RunNetTrain() {
  auto start_prepare_time = GetTimeUs();
  // Load graph
  std::string model_name = flags_->model_file_.substr(flags_->model_file_.find_last_of(DELIM_SLASH) + 1);

  MS_LOG(INFO) << "start reading model file";
  std::cout << "start reading model file" << std::endl;
  size_t size = 0;
  char *graph_buf = ReadFile(flags_->model_file_.c_str(), &size);
  if (graph_buf == nullptr) {
    MS_LOG(ERROR) << "Read model file failed while running " << model_name.c_str();
    std::cerr << "Read model file failed while running " << model_name.c_str() << std::endl;
    return RET_ERROR;
  }
  auto model = lite::TrainModel::Import(graph_buf, size);
  delete[](graph_buf);
  if (model == nullptr) {
    MS_LOG(ERROR) << "Import model file failed while running " << model_name.c_str();
    std::cerr << "Import model file failed while running " << model_name.c_str() << std::endl;
    return RET_ERROR;
  }
  auto context = std::make_shared<Context>();
  if (context == nullptr) {
    MS_LOG(ERROR) << "New context failed while running " << model_name.c_str();
    std::cerr << "New context failed while running " << model_name.c_str() << std::endl;
    return RET_ERROR;
  }

  if (flags_->cpu_bind_mode_ == 2) {
    context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = MID_CPU;
  } else if (flags_->cpu_bind_mode_ == 1) {
    context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = HIGHER_CPU;
  } else {
    context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = NO_BIND;
  }
  context->thread_num_ = flags_->num_threads_;
  // context->enable_float16_ = flags_->enable_fp16_;
  session_ = session::TrainSession::CreateSession(context.get());
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "CreateSession failed while running ", model_name.c_str();
    std::cout << "CreateSession failed while running ", model_name.c_str();
    return RET_ERROR;
  }
  auto ret = session_->CompileTrainGraph(model);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CompileGraph failed while running ", model_name.c_str();
    std::cout << "CompileGraph failed while running ", model_name.c_str();
    return ret;
  }

  session_->Train();

  ms_inputs_ = session_->GetInputs();
  auto end_prepare_time = GetTimeUs();
  MS_LOG(INFO) << "PrepareTime = " << (end_prepare_time - start_prepare_time) / 1000 << " ms";
  std::cout << "PrepareTime = " << (end_prepare_time - start_prepare_time) / 1000 << " ms" << std::endl;

  // Load input
  MS_LOG(INFO) << "start generate input data";
  auto status = LoadInput();
  if (status != 0) {
    MS_LOG(ERROR) << "Generate input data error";
    return status;
  }
  if (flags_->epochs_ > 0) {
    status = MarkPerformance();
    if (status != 0) {
      MS_LOG(ERROR) << "Run MarkPerformance error: " << status;
      std::cout << "Run MarkPerformance error: " << status << std::endl;
      return status;
    }
  }
  if (!flags_->data_file_.empty()) {
    status = MarkAccuracy();
    for (auto &data : data_) {
      data.second->shape.clear();
      data.second->data.clear();
      delete data.second;
    }
    data_.clear();
    if (status != 0) {
      MS_LOG(ERROR) << "Run MarkAccuracy error: " << status;
      std::cout << "Run MarkAccuracy error: " << status << std::endl;
      return status;
    }
  }
  if (!flags_->export_file_.empty()) {
    size_t tsize = 0;
    auto buf = session_->ExportToBuf(nullptr, &tsize);
    if (buf == nullptr) {
      MS_LOG(ERROR) << "Run ExportToBuf error";
      std::cout << "Run ExportToBuf error";
      return RET_ERROR;
    }
    SaveFile(flags_->export_file_, buf, size);

    status = RunExportedNet();
    if (status != 0) {
      MS_LOG(ERROR) << "Run Exported model error: " << status;
      std::cout << "Run Exported model error: " << status << std::endl;
      return status;
    }
  }
  return RET_OK;
}

void NetTrainFlags::InitInputDataList() {
  char *saveptr1;
  char *input_list = new char[this->in_data_file_.length() + 1];
  snprintf(input_list, this->in_data_file_.length() + 1, "%s", this->in_data_file_.c_str());
  char *cur_input;
  const char *split_c = ",";
  cur_input = strtok_r(input_list, split_c, &saveptr1);
  while (cur_input != nullptr) {
    input_data_list_.emplace_back(cur_input);
    cur_input = strtok_r(nullptr, split_c, &saveptr1);
  }
  delete[] input_list;
}

void NetTrainFlags::InitResizeDimsList() {
  std::string content;
  content = this->resize_dims_in_;
  std::vector<int64_t> shape;
  auto shape_strs = StringSplit(content, std::string(DELIM_COLON));
  for (const auto &shape_str : shape_strs) {
    shape.clear();
    auto dim_strs = StringSplit(shape_str, std::string(DELIM_COMMA));
    std::cout << "Resize Dims: ";
    for (const auto &dim_str : dim_strs) {
      std::cout << dim_str << " ";
      shape.emplace_back(static_cast<int64_t>(std::stoi(dim_str)));
    }
    std::cout << std::endl;
    this->resize_dims_.emplace_back(shape);
  }
}

int NetTrain::InitCallbackParameter() {
  // before callback
  before_call_back_ = [&](const std::vector<mindspore::tensor::MSTensor *> &before_inputs,
                          const std::vector<mindspore::tensor::MSTensor *> &before_outputs,
                          const mindspore::CallBackParam &callParam) {
    if (before_inputs.empty()) {
      MS_LOG(INFO) << "The num of beforeInputs is empty";
    }
    if (before_outputs.empty()) {
      MS_LOG(INFO) << "The num of beforeOutputs is empty";
    }
    if (op_times_by_type_.find(callParam.node_type) == op_times_by_type_.end()) {
      op_times_by_type_.insert(std::make_pair(callParam.node_type, std::make_pair(0, 0.0f)));
    }
    if (op_times_by_name_.find(callParam.node_name) == op_times_by_name_.end()) {
      op_times_by_name_.insert(std::make_pair(callParam.node_name, std::make_pair(0, 0.0f)));
    }

    op_call_times_total_++;
    op_begin_ = GetTimeUs();
    return true;
  };

  // after callback
  after_call_back_ = [&](const std::vector<mindspore::tensor::MSTensor *> &after_inputs,
                         const std::vector<mindspore::tensor::MSTensor *> &after_outputs,
                         const mindspore::CallBackParam &call_param) {
    uint64_t opEnd = GetTimeUs();

    if (after_inputs.empty()) {
      MS_LOG(INFO) << "The num of after inputs is empty";
    }
    if (after_outputs.empty()) {
      MS_LOG(INFO) << "The num of after outputs is empty";
    }

    float cost = static_cast<float>(opEnd - op_begin_) / 1000.0f;
    op_cost_total_ += cost;
    op_times_by_type_[call_param.node_type].first++;
    op_times_by_type_[call_param.node_type].second += cost;
    op_times_by_name_[call_param.node_name].first++;
    op_times_by_name_[call_param.node_name].second += cost;
    return true;
  };

  return RET_OK;
}

int NetTrain::Init() {
  if (this->flags_ == nullptr) {
    return 1;
  }
  MS_LOG(INFO) << "ModelPath = " << this->flags_->model_file_;
  MS_LOG(INFO) << "InDataPath = " << this->flags_->in_data_file_;
  MS_LOG(INFO) << "InDataType = " << this->flags_->in_data_type_in_;
  MS_LOG(INFO) << "Epochs = " << this->flags_->epochs_;
  MS_LOG(INFO) << "AccuracyThreshold = " << this->flags_->accuracy_threshold_;
  MS_LOG(INFO) << "WarmUpLoopCount = " << this->flags_->warm_up_loop_count_;
  MS_LOG(INFO) << "NumThreads = " << this->flags_->num_threads_;
  MS_LOG(INFO) << "expectedDataFile = " << this->flags_->data_file_;
  MS_LOG(INFO) << "exportDataFile = " << this->flags_->export_file_;

  if (this->flags_->epochs_ < 0) {
    MS_LOG(ERROR) << "epochs:" << this->flags_->epochs_ << " must be equal/greater than 0";
    std::cerr << "epochs:" << this->flags_->epochs_ << " must be equal/greater than 0" << std::endl;
    return RET_ERROR;
  }

  if (this->flags_->num_threads_ < 1) {
    MS_LOG(ERROR) << "numThreads:" << this->flags_->num_threads_ << " must be greater than 0";
    std::cerr << "numThreads:" << this->flags_->num_threads_ << " must be greater than 0" << std::endl;
    return RET_ERROR;
  }

  this->flags_->in_data_type_ = this->flags_->in_data_type_in_ == "img" ? kImage : kBinary;

  if (flags_->in_data_file_.empty() && !flags_->data_file_.empty()) {
    MS_LOG(ERROR) << "expectedDataFile not supported in case that inDataFile is not provided";
    std::cerr << "expectedDataFile is not supported in case that inDataFile is not provided" << std::endl;
    return RET_ERROR;
  }

  if (flags_->in_data_file_.empty() && !flags_->export_file_.empty()) {
    MS_LOG(ERROR) << "exportDataFile not supported in case that inDataFile is not provided";
    std::cerr << "exportDataFile is not supported in case that inDataFile is not provided" << std::endl;
    return RET_ERROR;
  }

  if (flags_->model_file_.empty()) {
    MS_LOG(ERROR) << "modelPath is required";
    std::cerr << "modelPath is required" << std::endl;
    return 1;
  }
  flags_->InitInputDataList();
  flags_->InitResizeDimsList();
  if (!flags_->resize_dims_.empty() && flags_->resize_dims_.size() != flags_->input_data_list_.size()) {
    MS_LOG(ERROR) << "Size of input resizeDims should be equal to size of input inDataPath";
    std::cerr << "Size of input resizeDims should be equal to size of input inDataPath" << std::endl;
    return RET_ERROR;
  }

  if (flags_->time_profiling_) {
    auto status = InitCallbackParameter();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Init callback Parameter failed.";
      std::cerr << "Init callback Parameter failed." << std::endl;
      return RET_ERROR;
    }
  }

  return RET_OK;
}

int NetTrain::PrintResult(const std::vector<std::string> &title,
                          const std::map<std::string, std::pair<int, float>> &result) {
  std::vector<size_t> columnLenMax(5);
  std::vector<std::vector<std::string>> rows;

  for (auto &iter : result) {
    char stringBuf[5][100] = {};
    std::vector<std::string> columns;
    size_t len;

    len = iter.first.size();
    if (len > columnLenMax.at(0)) {
      columnLenMax.at(0) = len + 4;
    }
    columns.push_back(iter.first);

    len = snprintf(stringBuf[1], sizeof(stringBuf[1]), "%f", iter.second.second / flags_->epochs_);
    if (len > columnLenMax.at(1)) {
      columnLenMax.at(1) = len + 4;
    }
    columns.emplace_back(stringBuf[1]);

    len = snprintf(stringBuf[2], sizeof(stringBuf[2]), "%f", iter.second.second / op_cost_total_);
    if (len > columnLenMax.at(2)) {
      columnLenMax.at(2) = len + 4;
    }
    columns.emplace_back(stringBuf[2]);

    len = snprintf(stringBuf[3], sizeof(stringBuf[3]), "%d", iter.second.first);
    if (len > columnLenMax.at(3)) {
      columnLenMax.at(3) = len + 4;
    }
    columns.emplace_back(stringBuf[3]);

    len = snprintf(stringBuf[4], sizeof(stringBuf[4]), "%f", iter.second.second);
    if (len > columnLenMax.at(4)) {
      columnLenMax.at(4) = len + 4;
    }
    columns.emplace_back(stringBuf[4]);

    rows.push_back(columns);
  }

  printf("-------------------------------------------------------------------------\n");
  for (int i = 0; i < 5; i++) {
    auto printBuf = title[i];
    if (printBuf.size() > columnLenMax.at(i)) {
      columnLenMax.at(i) = printBuf.size();
    }
    printBuf.resize(columnLenMax.at(i), ' ');
    printf("%s\t", printBuf.c_str());
  }
  printf("\n");
  for (size_t i = 0; i < rows.size(); i++) {
    for (int j = 0; j < 5; j++) {
      auto printBuf = rows[i][j];
      printBuf.resize(columnLenMax.at(j), ' ');
      printf("%s\t", printBuf.c_str());
    }
    printf("\n");
  }
  return RET_OK;
}

NetTrain::~NetTrain() {
  for (auto iter : this->data_) {
    delete (iter.second);
  }
  this->data_.clear();
  delete (session_);
}

int RunNetTrain(int argc, const char **argv) {
  NetTrainFlags flags;
  Option<std::string> err = flags.ParseFlags(argc, argv);

  if (err.IsSome()) {
    std::cerr << err.Get() << std::endl;
    std::cerr << flags.Usage() << std::endl;
    return RET_ERROR;
  }

  if (flags.help) {
    std::cerr << flags.Usage() << std::endl;
    return RET_OK;
  }

  NetTrain net_trainer(&flags);
  auto status = net_trainer.Init();
  if (status != 0) {
    MS_LOG(ERROR) << "NetTrain init Error : " << status;
    std::cerr << "NetTrain init Error : " << status << std::endl;
    return RET_ERROR;
  }

  status = net_trainer.RunNetTrain();
  if (status != 0) {
    MS_LOG(ERROR) << "Run NetTrain "
                  << flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
                  << " Failed : " << status;
    std::cerr << "Run NetTrain " << flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
              << " Failed : " << status << std::endl;
    return RET_ERROR;
  }

  MS_LOG(INFO) << "Run NetTrain " << flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
               << " Success.";
  std::cout << "Run NetTrain " << flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
            << " Success." << std::endl;
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
