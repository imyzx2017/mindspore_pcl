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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_MODEL_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_MODEL_PARSER_H_

#include <string>
#include <vector>
#include <memory>
#include <set>
#include <unordered_map>
#include "tools/converter/model_parser.h"
#include "proto/caffe.pb.h"
#include "tools/common/tensor_util.h"

namespace mindspore {
namespace lite {
class CaffeModelParser : public ModelParser {
 public:
  CaffeModelParser();

  virtual ~CaffeModelParser();

  schema::MetaGraphT *ParseToFb(const std::string &modelFile, const std::string &weightFile,
                                const QuantType &quantType = QuantType_QUANT_NONE) override;

 private:
  STATUS SetOpInputIdx(const caffe::LayerParameter &layer, schema::CNodeT *op, TensorCache *tensorCache);

  STATUS SetOpOutputIdx(const caffe::LayerParameter &layer, schema::CNodeT *op, TensorCache *tensorCache);

  STATUS SetWeightTensor(const std::vector<schema::TensorT *> &weightVec, schema::CNodeT *op, TensorCache *tensorCache);

  STATUS SetAllTensors(const TensorCache &tensorCache, schema::MetaGraphT *subGraphDef);

  STATUS SetGraphTensorIndex(const caffe::NetParameter &proto, TensorCache *tensorCache,
                             schema::MetaGraphT *subGraphDef);

  STATUS ParseLayer(const caffe::NetParameter &proto, const caffe::NetParameter &weight, TensorCache *tensorCache,
                    schema::MetaGraphT *subGraphDef, const QuantType &quantType);

  STATUS GetModelInput(const caffe::NetParameter &proto, TensorCache *tensorCache);

  static const std::set<std::string> skipedLayerType;

  std::unordered_map<std::string, std::string> splitLayer;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_MODEL_PARSER_H_
