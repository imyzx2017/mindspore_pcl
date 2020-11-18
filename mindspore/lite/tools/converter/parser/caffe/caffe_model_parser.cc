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

#include "tools/converter/parser/caffe/caffe_model_parser.h"
#include <vector>
#include <iostream>
#include <utility>
#include "tools/converter/parser/caffe/caffe_node_parser_registry.h"
#include "tools/converter/parser/caffe/caffe_inspector.h"
#include "tools/common/graph_util.h"
#include "tools/common/protobuf_utils.h"

namespace mindspore {
namespace lite {
CaffeModelParser::CaffeModelParser() {}

CaffeModelParser::~CaffeModelParser() {}

const std::set<std::string> CaffeModelParser::skipedLayerType = {"Dropout"};

schema::MetaGraphT *CaffeModelParser::ParseToFb(const std::string &modelFile, const std::string &weightFile,
                                                const QuantType &quantType) {
  int status = ValidateFileStr(modelFile, ".prototxt");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: modelFile must be *.prototxt";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  if (weightFile.empty()) {
    MS_LOG(ERROR) << "INPUT MISSING: weightFile is necessary";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_GRAPH_FILE_ERR);
    return nullptr;
  }

  status = ValidateFileStr(weightFile, ".caffemodel");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: weightFile must be *.caffemodel";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  auto metaGraph = std::make_unique<schema::MetaGraphT>();
  TensorCache tensorCache;

  caffe::NetParameter proto;
  status = ReadProtoFromText((const char *)modelFile.c_str(), &proto);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Read prototxt file failed, model path: " << modelFile;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  metaGraph->name = proto.name();

  caffe::NetParameter weight;
  status = ReadProtoFromBinaryFile((const char *)weightFile.c_str(), &weight);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Read caffemodel file failed, model path: " << weightFile;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  status = GetModelInput(proto, &tensorCache);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GetModelInput failed " << status;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  NoSupportOp::GetInstance()->SetFmkType("CAFFE");
  status = ParseLayer(proto, weight, &tensorCache, metaGraph.get(), quantType);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParseLayer failed " << status;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    for (auto &tensor : tensorCache.GetCachedTensor()) {
      delete tensor;
    }
    return nullptr;
  }

  status = SetGraphTensorIndex(proto, &tensorCache, metaGraph.get());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Set inputTensor index and outputTensor index for graph failed!";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  metaGraph->name = GetModelName(modelFile);

  SetAllTensors(tensorCache, metaGraph.get());

  return metaGraph.release();
}

STATUS CaffeModelParser::SetOpInputIdx(const caffe::LayerParameter &layer, schema::CNodeT *op,
                                       TensorCache *tensorCache) {
  for (int i = 0; i < layer.bottom_size(); i++) {
    int index = -1;
    if (splitLayer.find(layer.bottom(i)) != splitLayer.end()) {
      index = tensorCache->FindTensor(splitLayer.find(layer.bottom(i))->second);
    } else {
      index = tensorCache->FindTensor(layer.bottom(i));
    }
    if (index >= 0) {
      op->inputIndex.emplace_back(index);
    } else {
      MS_LOG(ERROR) << "Can't find input layer for " << layer.name().c_str();
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS CaffeModelParser::SetOpOutputIdx(const caffe::LayerParameter &layer, schema::CNodeT *op,
                                        TensorCache *tensorCache) {
  for (int i = 0; i < layer.top_size(); i++) {
    std::unique_ptr<schema::TensorT> msTensor = std::make_unique<schema::TensorT>();
    op->outputIndex.emplace_back(tensorCache->AddTensor(layer.top(i), msTensor.release(), OP_OUTPUT));
  }
  return RET_OK;
}

STATUS CaffeModelParser::SetWeightTensor(const std::vector<schema::TensorT *> &weightVec, schema::CNodeT *op,
                                         TensorCache *tensorCache) {
  for (auto iter : weightVec) {
    op->inputIndex.emplace_back(tensorCache->AddTensor("Weight", iter, CONST));
  }
  return RET_OK;
}

STATUS CaffeModelParser::SetAllTensors(const TensorCache &tensorCache, schema::MetaGraphT *subGraphDef) {
  std::vector<schema::TensorT *> tensors = tensorCache.GetCachedTensor();
  for (auto iter : tensors) {
    std::unique_ptr<schema::TensorT> temp(iter);
    subGraphDef->allTensors.emplace_back(move(temp));
  }
  return RET_OK;
}

STATUS CaffeModelParser::SetGraphTensorIndex(const caffe::NetParameter &proto, TensorCache *tensorCache,
                                             schema::MetaGraphT *subGraphDef) {
  CaffeInspector caffeInspector;
  caffeInspector.InspectModel(proto);
  for (auto iter : caffeInspector.GetGraphInput()) {
    int index = tensorCache->FindTensor(iter);
    if (index >= 0) {
      subGraphDef->inputIndex.emplace_back(index);
    } else {
      MS_LOG(ERROR) << "Can't find input tensor layer for graph.";
      return RET_ERROR;
    }
  }

  for (auto iter : caffeInspector.GetGraphOutput()) {
    int index = -1;
    if (splitLayer.find(iter) != splitLayer.end()) {
      index = tensorCache->FindTensor(splitLayer.find(iter)->second);
    } else {
      index = tensorCache->FindTensor(iter);
    }
    if (index >= 0) {
      subGraphDef->outputIndex.emplace_back(index);
    } else {
      MS_LOG(ERROR) << "Can't find output tensor layer for graph.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS CaffeModelParser::ParseLayer(const caffe::NetParameter &proto, const caffe::NetParameter &weight,
                                    TensorCache *tensorCache, schema::MetaGraphT *subGraphDef,
                                    const QuantType &quantType) {
  static bool interrupt = false;
  int status = RET_OK;
  for (int i = 0; i < proto.layer_size(); i++) {
    auto layer = proto.layer(i);

    caffe::LayerParameter layerP;
    for (int j = 0; j < weight.layer_size(); j++) {
      auto tempLayer = weight.layer(j);
      if (tempLayer.name() == layer.name()) {
        layerP = tempLayer;
        break;
      }
    }
    if (layer.type() == "Input") {
      std::unique_ptr<schema::TensorT> msTensor = std::make_unique<schema::TensorT>();
      for (int j = 0; j < layer.input_param().shape(0).dim_size(); j++) {
        msTensor->dims.push_back(layer.input_param().shape(0).dim(j));
      }
      msTensor->nodeType = schema::NodeType::NodeType_ValueNode;
      msTensor->refCount = 1;
      msTensor->dataType = kNumberTypeFloat32;
      tensorCache->AddTensor(layer.top(0), msTensor.release(), GRAPH_INPUT);
    } else {
      if (skipedLayerType.find(layer.type()) != skipedLayerType.end()) {
        MS_LOG(INFO) << "Skip layer " << layer.name();
        continue;
      }

      // here we only process the bn with phase
      if (layer.type() == "BatchNorm" && layer.include_size() == 1) {
        if (layer.include(0).phase() == caffe::TRAIN) {
          MS_LOG(INFO) << "Skip layer " << layer.name();
          continue;
        }
      }

      std::unique_ptr<schema::CNodeT> op = std::make_unique<schema::CNodeT>();
      op->name = layer.name();
      op->quantType = quantType;
      if (layer.type() == "Split") {
        for (int j = 0; j < layer.top_size(); ++j) {
          splitLayer.emplace(layer.top(j), layer.bottom(0));
        }
        continue;
      }

      auto nodeParser = CaffeNodeParserRegistry::GetInstance()->GetNodeParser(layer.type().c_str());
      if (nodeParser == nullptr || interrupt) {
        interrupt = true;
        if (nodeParser == nullptr) {
          NoSupportOp::GetInstance()->InsertOp(layer.type());
          status = (status == RET_OK ? RET_NOT_FIND_OP : status);
        }
        continue;
      }

      std::vector<schema::TensorT *> weightVec;
      auto status_node = nodeParser->Parse(layer, layerP, op.get(), &weightVec);
      if (status_node != RET_OK) {
        interrupt = true;
        if (status_node == RET_NOT_FIND_OP) {
          NoSupportOp::GetInstance()->InsertOp(layer.type());
        } else {
          MS_LOG(ERROR) << "Parse weight for " << layer.name() << " Failed!";
        }
        status = (status == RET_OK ? status_node : status);
        continue;
      }

      status_node = SetOpInputIdx(layer, op.get(), tensorCache);
      if (status_node != RET_OK) {
        MS_LOG(ERROR) << "Set Op " << layer.name() << " Input Index Failed!";
        status = (status == RET_OK ? status_node : status);
      }
      SetWeightTensor(weightVec, op.get(), tensorCache);

      status_node = SetOpOutputIdx(layer, op.get(), tensorCache);
      if (status_node != RET_OK) {
        interrupt = true;
        MS_LOG(ERROR) << "Set Op " << layer.name() << " Output Index Failed!";
        status = (status == RET_OK ? status_node : status);
        continue;
      }

      // op->fmkType = FmkType_CAFFE;
      subGraphDef->nodes.emplace_back(move(op));
    }
  }
  return status;
}

STATUS CaffeModelParser::GetModelInput(const caffe::NetParameter &proto, TensorCache *tensorCache) {
  for (int i = 0; i < proto.input_size(); i++) {
    if (proto.input_dim_size() <= 0) {
      continue;
    }
    std::unique_ptr<schema::TensorT> msTensor = std::make_unique<schema::TensorT>();
    if (proto.input_dim_size() > 4) {
      int step = proto.input_dim_size() / proto.input_size();
      for (int j = i * step; j < (i + 1) * step; j++) {
        msTensor->dims.push_back(proto.input_dim(j));
      }
    } else {
      for (int j = 0; j < proto.input_dim_size(); j++) {
        msTensor->dims.push_back(proto.input_dim(j));
      }
    }
    msTensor->refCount = schema::NodeType::NodeType_ValueNode;
    msTensor->dataType = kNumberTypeFloat32;
    tensorCache->AddTensor(proto.input(i), msTensor.release(), GRAPH_INPUT);
  }

  for (int i = 0; i < proto.input_shape_size(); i++) {
    auto shape = proto.input_shape(i);
    std::unique_ptr<schema::TensorT> msTensor = std::make_unique<schema::TensorT>();
    for (int j = 0; j < shape.dim_size(); j++) {
      msTensor->dims.push_back(shape.dim(j));
    }
    msTensor->refCount = schema::NodeType::NodeType_ValueNode;
    msTensor->dataType = kNumberTypeFloat32;
    tensorCache->AddTensor(proto.input(i), msTensor.release(), GRAPH_INPUT);
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
