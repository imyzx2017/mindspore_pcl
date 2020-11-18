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

#include "backend/session/infer_session.h"
#include <memory>
#include <algorithm>
#include <fstream>

#include "include/inference.h"
#include "utils/load_onnx/anf_converter.h"
#include "backend/session/session_basic.h"
#include "backend/session/session_factory.h"
#include "backend/session/executor_manager.h"
#include "base/base_ref_utils.h"
#include "backend/kernel_compiler/oplib/oplib.h"
#include "utils/context/context_extends.h"
#include "runtime/device/kernel_runtime_manager.h"

#include "pybind11/pybind11.h"

#ifdef ENABLE_D
#include "utils/ms_context.h"
#endif

using std::string;
using std::vector;

namespace py = pybind11;
namespace mindspore {
namespace inference {
std::shared_ptr<InferSession> InferSession::CreateSession(const std::string &device, uint32_t device_id) {
  try {
    auto session = std::make_shared<MSInferSession>();
    Status ret = session->InitEnv(device, device_id);
    if (ret != SUCCESS) {
      return nullptr;
    }
    return session;
  } catch (std::bad_alloc &e) {
    MS_LOG(ERROR) << "Inference CreatSession failed, failed to alloc memory";
    return nullptr;
  }
}

MSInferSession::MSInferSession() = default;
MSInferSession::~MSInferSession() = default;

std::shared_ptr<std::vector<char>> MSInferSession::ReadFile(const std::string &file) {
  if (file.empty()) {
    MS_LOG(ERROR) << "file is nullptr";
    return nullptr;
  }
  std::string realPath = file;
  std::ifstream ifs(realPath);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file: " << realPath << " is not exist";
    return nullptr;
  }

  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << realPath << "open failed";
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  std::shared_ptr<std::vector<char>> buf(new (std::nothrow) std::vector<char>(size));
  if (buf == nullptr) {
    MS_LOG(ERROR) << "malloc buf failed, file: " << realPath;
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf->data(), size);
  ifs.close();

  return buf;
}

Status MSInferSession::LoadModelFromFile(const std::string &file_name, uint32_t &model_id) {
  auto graphBuf = ReadFile(file_name);
  if (graphBuf == nullptr) {
    MS_LOG(ERROR) << "Read model file failed, file name is " << file_name.c_str();
    return FAILED;
  }
  auto graph = LoadModel(graphBuf->data(), graphBuf->size(), device_type_);
  if (graph == nullptr) {
    MS_LOG(ERROR) << "Load graph model failed, file name is " << file_name.c_str();
    return FAILED;
  }
  Status ret = CompileGraph(graph, model_id);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "Compile graph model failed, file name is " << file_name.c_str();
    return FAILED;
  }
  MS_LOG(INFO) << "Load model from file " << file_name << " success";

#ifdef ENABLE_D
  // set d context
  rtError_t rt_ret = rtCtxGetCurrent(&context_);
  if (rt_ret != RT_ERROR_NONE || context_ == nullptr) {
    MS_LOG(ERROR) << "the ascend device context is null";
    return FAILED;
  }
#endif

  return SUCCESS;
}

Status MSInferSession::UnloadModel(uint32_t model_id) { return SUCCESS; }

Status ServingTensor2MSTensor(size_t index, const InferTensorBase &out_tensor, tensor::TensorPtr &ms_tensor) {
  std::vector<int64_t> shape = out_tensor.shape();
  TypeId data_type;
  const std::map<inference::DataType, TypeId> type2id_map{
    {inference::kMSI_Unknown, TypeId::kNumberTypeBegin},   {inference::kMSI_Bool, TypeId::kNumberTypeBool},
    {inference::kMSI_Int8, TypeId::kNumberTypeInt8},       {inference::kMSI_Uint8, TypeId::kNumberTypeUInt8},
    {inference::kMSI_Int16, TypeId::kNumberTypeInt16},     {inference::kMSI_Uint16, TypeId::kNumberTypeUInt16},
    {inference::kMSI_Int32, TypeId::kNumberTypeInt32},     {inference::kMSI_Uint32, TypeId::kNumberTypeUInt32},
    {inference::kMSI_Int64, TypeId::kNumberTypeInt64},     {inference::kMSI_Uint64, TypeId::kNumberTypeUInt64},
    {inference::kMSI_Float16, TypeId::kNumberTypeFloat16}, {inference::kMSI_Float32, TypeId::kNumberTypeFloat32},
    {inference::kMSI_Float64, TypeId::kNumberTypeFloat64},
  };
  auto it = type2id_map.find(out_tensor.data_type());
  if (it == type2id_map.end()) {
    MSI_LOG_WARNING << "undefined MSI data type " << out_tensor.data_type();
    return FAILED;
  } else {
    data_type = it->second;
  }

  ms_tensor = std::make_shared<tensor::Tensor>(data_type, shape);
  if (out_tensor.data_size() == 0 || ms_tensor->Size() != out_tensor.data_size()) {
    MSI_LOG_ERROR << "input " << std::to_string(index)
                  << " data size not match shape and dtype, calculated required size " << ms_tensor->Size()
                  << ", given " << out_tensor.data_size();
    return INFER_STATUS(INVALID_INPUTS) << "input " << std::to_string(index)
                                        << " data size not match shape and dtype, calculated required size "
                                        << ms_tensor->Size() << ", given " << out_tensor.data_size();
  }
  if (out_tensor.data() == nullptr || ms_tensor->data_c() == nullptr) {
    MSI_LOG_ERROR << "invalid data buffer";
    return FAILED;
  }
  auto ret_code = memcpy_s(ms_tensor->data_c(), ms_tensor->Size(), out_tensor.data(), out_tensor.data_size());
  if (ret_code != 0) {
    MS_LOG(ERROR) << "Failed to copy data from ms_tensor to out_tensor.";
  }
  return SUCCESS;
}

void MSTensor2ServingTensor(tensor::TensorPtr ms_tensor, InferTensorBase &out_tensor) {
  vector<int64_t> shape = ms_tensor->shape();
  out_tensor.set_shape(shape);

  const std::map<TypeId, inference::DataType> id2type_map{
    {TypeId::kNumberTypeBegin, inference::kMSI_Unknown},   {TypeId::kNumberTypeBool, inference::kMSI_Bool},
    {TypeId::kNumberTypeFloat64, inference::kMSI_Float64}, {TypeId::kNumberTypeInt8, inference::kMSI_Int8},
    {TypeId::kNumberTypeUInt8, inference::kMSI_Uint8},     {TypeId::kNumberTypeInt16, inference::kMSI_Int16},
    {TypeId::kNumberTypeUInt16, inference::kMSI_Uint16},   {TypeId::kNumberTypeInt32, inference::kMSI_Int32},
    {TypeId::kNumberTypeUInt32, inference::kMSI_Uint32},   {TypeId::kNumberTypeInt64, inference::kMSI_Int64},
    {TypeId::kNumberTypeUInt64, inference::kMSI_Uint64},   {TypeId::kNumberTypeFloat16, inference::kMSI_Float16},
    {TypeId::kNumberTypeFloat32, inference::kMSI_Float32},
  };
  auto it = id2type_map.find(ms_tensor->data_type());
  if (it == id2type_map.end()) {
    MSI_LOG_WARNING << "undefined MS data type " << ms_tensor->data_type();
    out_tensor.set_data_type(inference::kMSI_Unknown);
  } else {
    out_tensor.set_data_type(it->second);
  }
  out_tensor.set_data(ms_tensor->data_c(), ms_tensor->Size());
}

Status MSInferSession::ExecuteModel(uint32_t model_id, const RequestBase &request, ReplyBase &reply) {
#ifdef ENABLE_D
  if (context_ == nullptr) {
    MS_LOG(ERROR) << "rtCtx is nullptr";
    return FAILED;
  }
  rtError_t rt_ret = rtCtxSetCurrent(context_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "set Ascend rtCtx failed";
    return FAILED;
  }
#endif

  vector<tensor::TensorPtr> inputs;
  for (size_t i = 0; i < request.size(); i++) {
    if (request[i] == nullptr) {
      MS_LOG(ERROR) << "Execute Model " << model_id << " Failed， input tensor is null, index " << i;
      return FAILED;
    }
    tensor::TensorPtr input = nullptr;
    auto ret = ServingTensor2MSTensor(i, *request[i], input);
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "Tensor convert failed";
      return ret;
    }
    inputs.push_back(input);
  }
  auto ret = CheckModelInputs(model_id, inputs);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "Check Model " << model_id << " Inputs Failed";
    return ret;
  }
  vector<tensor::TensorPtr> outputs = RunGraph(model_id, inputs);
  if (outputs.empty()) {
    MS_LOG(ERROR) << "Execute Model " << model_id << " Failed";
    return FAILED;
  }
  reply.clear();
  for (const auto &tensor : outputs) {
    auto out_tensor = reply.add();
    if (out_tensor == nullptr) {
      MS_LOG(ERROR) << "Execute Model " << model_id << " Failed add output tensor failed";
      return FAILED;
    }
    MSTensor2ServingTensor(tensor, *out_tensor);
  }
  return SUCCESS;
}

Status MSInferSession::FinalizeEnv() {
  session::ExecutorManager::Instance().Clear();
  device::KernelRuntimeManager::Instance().ClearRuntimeResource();
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    MS_LOG(ERROR) << "Get Context failed!";
    return FAILED;
  }
  if (!context::CloseTsd(ms_context)) {
    MS_LOG(ERROR) << "Inference CloseTsd failed!";
    return FAILED;
  }
  return SUCCESS;
}

std::shared_ptr<FuncGraph> MSInferSession::LoadModel(const char *model_buf, size_t size, const std::string &device) {
  try {
    auto anf_graph = lite::AnfConverter::RunAnfConverter(model_buf, size);
    return anf_graph;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Inference LoadModel failed";
    return nullptr;
  }
}

void MSInferSession::RegAllOp() {
  static std::mutex init_mutex;
  static bool Initialized = false;

  std::lock_guard<std::mutex> lock(init_mutex);
  if (Initialized) {
    return;
  }
  Initialized = true;
  MsContext::GetInstance()->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  Py_Initialize();
  auto c_expression = PyImport_ImportModule("mindspore._c_expression");
  MS_EXCEPTION_IF_NULL(c_expression);
  PyObject *c_expression_dict = PyModule_GetDict(c_expression);
  MS_EXCEPTION_IF_NULL(c_expression_dict);

  PyObject *op_info_loader_class = PyDict_GetItemString(c_expression_dict, "OpInfoLoaderPy");
  MS_EXCEPTION_IF_NULL(op_info_loader_class);
  PyObject *op_info_loader = PyInstanceMethod_New(op_info_loader_class);
  MS_EXCEPTION_IF_NULL(op_info_loader);
  PyObject *op_info_loader_ins = PyObject_CallObject(op_info_loader, nullptr);
  MS_EXCEPTION_IF_NULL(op_info_loader_ins);
  auto all_ops_info_vector_addr_ul = PyObject_CallMethod(op_info_loader_ins, "get_all_ops_info", nullptr);
  MS_EXCEPTION_IF_NULL(all_ops_info_vector_addr_ul);
  auto all_ops_info_vector_addr = PyLong_AsVoidPtr(all_ops_info_vector_addr_ul);
  auto all_ops_info = static_cast<std::vector<kernel::OpInfo *> *>(all_ops_info_vector_addr);
  for (auto op_info : *all_ops_info) {
    kernel::OpLib::RegOpInfo(std::shared_ptr<kernel::OpInfo>(op_info));
  }
  all_ops_info->clear();
  delete all_ops_info;
  Py_DECREF(op_info_loader);
  Py_DECREF(op_info_loader_class);
  Py_DECREF(c_expression_dict);
  Py_DECREF(c_expression);
  return;
}

Status MSInferSession::CompileGraph(std::shared_ptr<FuncGraph> funcGraphPtr, uint32_t &model_id) {
  MS_ASSERT(session_impl_ != nullptr);
  try {
    auto graph_id = session_impl_->CompileGraph(NOT_NULL(funcGraphPtr));
    py::gil_scoped_release gil_release;
    model_id = graph_id;
    return SUCCESS;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Inference CompileGraph failed";
    return FAILED;
  }
}

std::vector<tensor::TensorPtr> MSInferSession::RunGraph(uint32_t graph_id,
                                                        const std::vector<tensor::TensorPtr> &inputs) {
  try {
    VectorRef outputs;
    session_impl_->RunGraph(graph_id, inputs, &outputs);
    return TransformVectorRefToMultiTensor(outputs);
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Inference Rungraph failed";
    return std::vector<tensor::TensorPtr>();
  }
}

string MSInferSession::AjustTargetName(const std::string &device) {
  if (device == kAscendDevice) {
    return std::string(kAscendDevice) + "Inference";
  } else {
    MS_LOG(ERROR) << "Only support device Ascend right now";
    return "";
  }
}

Status MSInferSession::InitEnv(const std::string &device, uint32_t device_id) {
  RegAllOp();
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    MS_LOG(ERROR) << "Get Context failed!";
    return FAILED;
  }
  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  ms_context->set_param<uint32_t>(MS_CTX_DEVICE_ID, device_id);
  auto ajust_device = AjustTargetName(device);
  if (ajust_device == "") {
    return FAILED;
  }
  ms_context->set_param<std::string>(MS_CTX_DEVICE_TARGET, device);
  if (!context::OpenTsd(ms_context)) {
    MS_LOG(ERROR) << "Session init OpenTsd failed!";
    return FAILED;
  }
  session_impl_ = session::SessionFactory::Get().Create(ajust_device);
  if (session_impl_ == nullptr) {
    MS_LOG(ERROR) << "Session create failed!, please make sure target device:" << device << " is available.";
    return FAILED;
  }
  session_impl_->Init(device_id);
  return SUCCESS;
}

Status MSInferSession::CheckModelInputs(uint32_t graph_id, const std::vector<tensor::TensorPtr> &inputs) const {
  MS_ASSERT(session_impl_ != nullptr);
  std::string error_msg;
  if (!session_impl_->CheckModelInputs(graph_id, inputs, &error_msg)) {
    return INFER_STATUS(INVALID_INPUTS) << error_msg;
  }
  return SUCCESS;
}

Status MSInferSession::GetModelInputsInfo(uint32_t model_id, std::vector<inference::InferTensor> *tensor_list) const {
  vector<tensor::TensorPtr> inputs;
  session_impl_->GetModelInputsInfo(model_id, &inputs);
  if (inputs.size() == 0) {
    MS_LOG(ERROR) << "The model inputs is NULL";
    return FAILED;
  }
  for (const auto &tensor : inputs) {
    InferTensor infer_tensor = InferTensor();
    MSTensor2ServingTensor(tensor, infer_tensor);
    tensor_list->push_back(infer_tensor);
  }
  return SUCCESS;
}
}  // namespace inference
}  // namespace mindspore
