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
#include "tools/optimizer/graph/infershape_pass.h"
#include <vector>
#include <memory>
#include <algorithm>
#include "mindspore/lite/include/errorcode.h"
#include "mindspore/lite/src/ops/primitive_c.h"
#include "tools/anf_importer/import_from_meta_graphT.h"

namespace mindspore::opt {
abstract::AbstractTensorPtr InferShapePass::ConvertLiteTensorToAbstractTensor(lite::Tensor *tensor) {
  MS_ASSERT(nullptr != tensor);
  std::vector<int> shape(tensor->shape());
  auto type_id = static_cast<TypeId>(tensor->data_type());
  auto type_ptr = TypeIdToType(type_id);
  std::vector<int64_t> shape_vector;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                       [](const int32_t &value) { return static_cast<int64_t>(value); });
  auto new_abstract = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  if (new_abstract == nullptr) {
    MS_LOG(ERROR) << "new AbstractTensor failed";
    return nullptr;
  }
  auto new_value = std::make_shared<ParamValueLite>();
  if (new_value == nullptr) {
    MS_LOG(ERROR) << "new ParamValueLite failed";
    return nullptr;
  }
  new_value->set_tensor_shape(tensor->shape());
  new_value->set_tensor_type(tensor->data_type());
  new_value->set_format(tensor->GetFormat());
  new_abstract->set_value(new_value);
  return new_abstract;
}

STATUS InferShapePass::SetParameterAbstract(const ParameterPtr &parameter) {
  MS_ASSERT(parameter != nullptr);
  auto old_abstract = parameter->abstract();
  if (old_abstract == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << parameter->name();
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(old_abstract)) {
    MS_LOG(ERROR) << "Abstract of parameter should be abstract tensor, " << parameter->name();
    return RET_ERROR;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(old_abstract);

  auto typePtr = abstract_tensor->element()->GetTypeTrack();
  if (typePtr == nullptr) {
    MS_LOG(ERROR) << "typePtr is nullptr";
    return RET_ERROR;
  }

  if (!utils::isa<abstract::ShapePtr>(abstract_tensor->BuildShape())) {
    MS_LOG(ERROR) << "Shape of Abstract of parameter should be ShapePtr, " << parameter->name();
    return RET_ERROR;
  }
  auto shape_vector = utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape();
  std::vector<int32_t> shape;
  (void)std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(shape),
                       [](const int64_t &value) { return static_cast<int32_t>(value); });

  auto new_abstract = std::make_shared<abstract::AbstractTensor>(typePtr, shape_vector);
  auto new_value = std::make_shared<ParamValueLite>();
  new_value->set_tensor_shape(shape);  // scalar's shape is {}
  new_value->set_tensor_type(typePtr->type_id());
  new_value->set_format(schema::Format_NHWC);  // default format is NHWC
  if (parameter->has_default()) {
    auto param_value = std::dynamic_pointer_cast<ParamValueLite>(parameter->default_param());
    new_value->set_format(param_value->format());
    new_value->set_tensor_size(param_value->tensor_size());

    char *tensor_data = new (std::nothrow) char[new_value->tensor_size()];
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "new char[] failed";
      return RET_ERROR;
    }
    auto ret = memcpy_s(tensor_data, new_value->tensor_size(), param_value->tensor_addr(), param_value->tensor_size());
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "memcpy error: " << ret;
      return RET_ERROR;
    }
    new_value->set_tensor_addr(tensor_data);
  }
  new_abstract->set_value(new_value);
  parameter->set_abstract(new_abstract);
  return RET_OK;
}

void InferShapePass::FreeTensors(std::vector<lite::Tensor *> *tensors) {
  for (auto tensor : *tensors) {
    delete tensor;
  }
  tensors->clear();
  tensors->shrink_to_fit();
}

STATUS InferShapePass::GetCNodeInputTensors(const CNodePtr &cnode, std::vector<lite::Tensor *> *input_tensors) {
  MS_ASSERT(cnode != nullptr);
  MS_ASSERT(input_tensors != nullptr);
  auto inputs = cnode->inputs();
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto input = inputs[i];
    if (input == nullptr) {
      MS_LOG(ERROR) << "input is nullptr";
      return RET_ERROR;
    }
    auto tensor = std::make_unique<lite::Tensor>();
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "new input tensor failed";
      return RET_ERROR;
    }

    if (utils::isa<ValueNodePtr>(cnode->input(i))) {
      MS_LOG(ERROR) << "input is value node";
      continue;
    }

    AbstractBasePtr abstract = GetCNodeInputAbstract(cnode, i);
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "Abstract of CNode is nullptr";
      return RET_ERROR;
    }
    if (!utils::isa<abstract::AbstractTensorPtr>(abstract)) {
      MS_LOG(DEBUG) << "Abstract of parameter should be abstract tensor";
      return RET_ERROR;
    }
    auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
    if (!utils::isa<ParamValueLitePtr>(abstract_tensor->GetValueTrack())) {  // input node not complete infershape
      MS_LOG(DEBUG) << "Value of abstract is not ParamValueLite, indicate that infershape has failed";
      return RET_ERROR;
    }
    auto param_value_lite = utils::cast<ParamValueLitePtr>(abstract_tensor->GetValueTrack());
    if (param_value_lite == nullptr) {
      MS_LOG(ERROR) << "ParamValueLite of abstract is nullptr";
      return RET_ERROR;
    }
    tensor->set_shape(param_value_lite->tensor_shape());
    tensor->set_data_type(param_value_lite->tensor_type());
    tensor->SetFormat(schema::Format(param_value_lite->format()));

    if (utils::isa<ParameterPtr>(input)) {
      auto parameter = input->cast<ParameterPtr>();
      if (parameter->has_default()) {
        auto param_value = std::dynamic_pointer_cast<ParamValueLite>(parameter->default_param());
        auto ret = tensor->MallocData();
        if (ret != 0) {
          MS_LOG(ERROR) << "Malloc tensor data failed";
          return RET_ERROR;
        }
        ret = memcpy_s(tensor->MutableData(), tensor->Size(), param_value->tensor_addr(), param_value->tensor_size());
        if (ret != EOK) {
          MS_LOG(ERROR) << "memcpy error: " << ret;
          return RET_ERROR;
        }
      }
    }
    input_tensors->push_back(tensor.release());
  }
  return RET_OK;
}

STATUS InferShapePass::GetCNodeOutputTensors(const CNodePtr &cnode, std::vector<lite::Tensor *> *output_tensors) {
  MS_ASSERT(output_tensors != nullptr);
  auto abstract = cnode->abstract();
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "abstract is nullptr";
    return RET_ERROR;
  }
  size_t num_outputs = 1;
  if (utils::isa<abstract::AbstractTuple>(abstract)) {
    auto abstract_tuple = abstract->cast<abstract::AbstractTuplePtr>();
    num_outputs = abstract_tuple->size();
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    auto output_tensor = std::make_unique<lite::Tensor>();
    if (output_tensor == nullptr) {
      MS_LOG(ERROR) << "new output tensor failed";
      return RET_ERROR;
    }
    output_tensors->push_back(output_tensor.release());
  }
  return RET_OK;
}

STATUS InferShapePass::SetCNodeAbstract(const std::vector<lite::Tensor *> &output_tensors,
                                        const std::shared_ptr<CNode> &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (output_tensors.size() == 0) {
    MS_LOG(ERROR) << "empty output_tensors";
    return RET_ERROR;
  }
  if (output_tensors.size() == 1) {
    auto tensor = output_tensors.front();
    auto new_abstract = ConvertLiteTensorToAbstractTensor(tensor);
    if (new_abstract == nullptr) {
      return RET_ERROR;
    }
    cnode->set_abstract(new_abstract);
  } else {
    AbstractBasePtrList abstract_list;
    for (size_t i = 0; i < output_tensors.size(); i++) {
      auto tensor = output_tensors.front();
      auto new_abstract = ConvertLiteTensorToAbstractTensor(tensor);
      if (new_abstract == nullptr) {
        return RET_ERROR;
      }
      abstract_list.emplace_back(new_abstract);
    }
    cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  }
  return RET_OK;
}

bool InferShapePass::Run(const FuncGraphPtr &func_graph) {
  if (fmk_type != lite::converter::FmkType_TF && fmk_type != lite::converter::FmkType_TFLITE) {
    MS_LOG(INFO) << "The framework type of model should be tf/tflite.";
    return false;
  }
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (utils::isa<ParameterPtr>(node)) {
      int status = SetParameterAbstract(node->cast<ParameterPtr>());
      if (status != RET_OK) {
        return false;
      }
      continue;
    }
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto origin_primc = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(cnode->input(0));
    if (origin_primc == nullptr) {
      MS_LOG(ERROR) << "origin_primc is nullptr";
      return false;
    }
    auto origin_primt = origin_primc->GetPrimitiveT();
    if (origin_primt == nullptr) {
      MS_LOG(ERROR) << "origin_primt is nullptr";
      return false;
    }
    auto type = GetCNodeType(cnode);
    if ((type == schema::PrimitiveType_TupleGetItem) ||
#ifdef SUPPORT_TRAIN
        (type == schema::PrimitiveType_Depend) || (type == schema::PrimitiveType_ControlDepend) ||
#endif
        (type == schema::PrimitiveType_MakeTuple || type == schema::PrimitiveType_Return)) {
      continue;
    }
    std::vector<lite::Tensor *> input_tensors;
    std::vector<lite::Tensor *> output_tensors;
    auto status = GetCNodeInputTensors(cnode, &input_tensors);
    if (status != RET_OK) {
      MS_LOG(DEBUG) << "input shape unknown, infershape can't process cnode " << cnode->fullname_with_scope();
      FreeTensors(&input_tensors);
      continue;
    }
    status = GetCNodeOutputTensors(cnode, &output_tensors);
    if (status != RET_OK) {
      FreeTensors(&input_tensors);
      FreeTensors(&output_tensors);
      continue;
    }
    auto primt = std::make_unique<schema::PrimitiveT>();
    if (primt == nullptr) {
      MS_LOG(ERROR) << "primt is nullptr";
      return false;
    }
    *primt = *origin_primt;
    auto primc = std::shared_ptr<lite::PrimitiveC>(lite::PrimitiveC::Create(primt.release()));
    if (primc == nullptr) {
      MS_LOG(ERROR) << "primc is nullptr";
      return false;
    }
    status = primc->InferShape(input_tensors, output_tensors);
    if (status == RET_OK) {
      status = SetCNodeAbstract(output_tensors, cnode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "set CNode abstract failed: " << cnode->fullname_with_scope();
      }
    }
    FreeTensors(&input_tensors);
    FreeTensors(&output_tensors);
  }
  return true;
}
}  // namespace mindspore::opt
