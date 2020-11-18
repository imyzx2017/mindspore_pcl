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

#include "tools/converter/parser/caffe/caffe_batchnorm_parser.h"
#include <cmath>
#include <memory>
#include "tools/common/tensor_util.h"

#define CAFFE_BATCH_NORM_ESP_DEFAULT_FLOAT 0.00001
#define CAFFE_BATCH_NORM_ESP_DEFAULT_DIFF_FLOAT 0.000000001

static const int CAFFE_BATCHNORMAL_BOTTOM_SIZE = 1;
static const int CAFFE_BATCHNORMAL_TOP_SIZE = 1;

namespace mindspore {
namespace lite {
using STATUS = int;

STATUS CaffeBatchNormParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight,
                                   schema::CNodeT *op, std::vector<schema::TensorT *> *weightVec) {
  MS_LOG(DEBUG) << "parse CaffeBatchNormParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::BatchNormT> attr = std::make_unique<schema::BatchNormT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  const caffe::BatchNormParameter batchNormParam = proto.batch_norm_param();
  // check bottom size
  if (proto.bottom_size() != CAFFE_BATCHNORMAL_BOTTOM_SIZE) {
    MS_LOG(ERROR) << "Layer " << proto.name().c_str() << "bottom numbers is error, it must be "
                  << CAFFE_BATCHNORMAL_BOTTOM_SIZE << "but is " << proto.bottom_size();
    return RET_ERROR;
  }

  // check top size
  if (proto.top_size() != CAFFE_BATCHNORMAL_TOP_SIZE) {
    MS_LOG(ERROR) << "Layer " << proto.name().c_str() << "top numbers is error, it must be "
                  << CAFFE_BATCHNORMAL_TOP_SIZE << "but is " << proto.top_size();
    return RET_ERROR;
  }

  if (batchNormParam.has_eps()) {
    if (fabs(CAFFE_BATCH_NORM_ESP_DEFAULT_FLOAT - batchNormParam.eps()) < CAFFE_BATCH_NORM_ESP_DEFAULT_DIFF_FLOAT) {
      attr->epsilon = CAFFE_BATCH_NORM_ESP_DEFAULT_FLOAT;
    } else {
      auto tmpAuto = batchNormParam.eps();
      attr->epsilon = tmpAuto;
    }
  } else {
    attr->epsilon = CAFFE_BATCH_NORM_ESP_DEFAULT_FLOAT;
  }

  const float blob2Data =
    (weight.blobs(2).double_data_size() > 0) ? weight.blobs(2).double_data(0) : weight.blobs(2).data(0);
  const float scaleFactor = blob2Data == 0 ? 0 : 1 / blob2Data;

  // parse weight gamma
  auto gamma = ConvertWeight(weight.blobs(0));
  if (gamma == nullptr) {
    MS_LOG(ERROR) << "Convert blobs(0) for layer " << weight.name().c_str() << " failed";
    return RET_ERROR;
  }

  auto estimatedMean = reinterpret_cast<float *>(gamma->data.data());
  auto estimatedMeanShapeSize = GetShapeSize(*gamma);
  for (size_t i = 0; i < estimatedMeanShapeSize; i++) {
    estimatedMean[i] = estimatedMean[i] * scaleFactor;
  }
  estimatedMean = nullptr;
  weightVec->push_back(gamma);

  // parse weight beta
  auto beta = ConvertWeight(weight.blobs(1));
  if (beta == nullptr) {
    MS_LOG(ERROR) << "Convert blobs(1) for layer " << weight.name().c_str() << " failed";
    return RET_ERROR;
  }

  auto estimatedVariance = reinterpret_cast<float *>(beta->data.data());
  size_t estimatedVarianceShapeSize = GetShapeSize(*beta);
  for (size_t i = 0; i < estimatedVarianceShapeSize; i++) {
    estimatedVariance[i] = estimatedVariance[i] * scaleFactor;
  }
  estimatedVariance = nullptr;
  weightVec->push_back(beta);

  op->name = proto.name();
  op->primitive->value.type = schema::PrimitiveType_BatchNorm;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

CaffeNodeRegistrar g_caffeBatchNormParser("BatchNorm", new CaffeBatchNormParser());
}  // namespace lite
}  // namespace mindspore
