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
#include <memory>
#include <string>
#include "./securec.h"
#include "common/common_test.h"
#include "gtest/gtest.h"

#include "mindspore/ccsrc/minddata/dataset/core/data_type.h"
#include "mindspore/ccsrc/minddata/dataset/core/tensor.h"
#include "mindspore/ccsrc/minddata/dataset/core/tensor_shape.h"
#include "mindspore/ccsrc/minddata/dataset/include/de_tensor.h"
#include "mindspore/lite/src/common/log_adapter.h"
#include "mindspore/lite/src/tensor.h"

using MSTensor = mindspore::tensor::MSTensor;
using DETensor = mindspore::tensor::DETensor;
using LiteTensor = mindspore::lite::Tensor;
using Tensor = mindspore::dataset::Tensor;
using DataType = mindspore::dataset::DataType;
using TensorShape = mindspore::dataset::TensorShape;

class MindDataTestTensorDE : public mindspore::CommonTest {
 public:
  MindDataTestTensorDE() {}
};

TEST_F(MindDataTestTensorDE, MSTensorBasic) {
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({2, 3}), DataType(DataType::DE_FLOAT32));
  auto ms_tensor = std::shared_ptr<MSTensor>(new DETensor(t));
  ASSERT_EQ(t == std::dynamic_pointer_cast<DETensor>(ms_tensor)->tensor(), true);
}

TEST_F(MindDataTestTensorDE, MSTensorConvertToLiteTensor) {
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({2, 3}), DataType(DataType::DE_FLOAT32));
  auto ms_tensor = std::shared_ptr<DETensor>(new DETensor(t));
  std::shared_ptr<MSTensor> lite_ms_tensor =
    std::shared_ptr<MSTensor>(std::dynamic_pointer_cast<DETensor>(ms_tensor)->ConvertToLiteTensor());
  // check if the lite_ms_tensor is the derived LiteTensor
  LiteTensor *lite_tensor = static_cast<LiteTensor *>(lite_ms_tensor.get());
  ASSERT_EQ(lite_tensor != nullptr, true);
}

TEST_F(MindDataTestTensorDE, MSTensorShape) {
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({2, 3}), DataType(DataType::DE_FLOAT32));
  auto ms_tensor = std::shared_ptr<MSTensor>(new DETensor(t));
  ASSERT_EQ(ms_tensor->DimensionSize(0) == 2, true);
  ASSERT_EQ(ms_tensor->DimensionSize(1) == 3, true);
}

TEST_F(MindDataTestTensorDE, MSTensorSize) {
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({2, 3}), DataType(DataType::DE_FLOAT32));
  auto ms_tensor = std::shared_ptr<MSTensor>(new DETensor(t));
  ASSERT_EQ(ms_tensor->ElementsNum() == 6, true);
  ASSERT_EQ(ms_tensor->Size() == 24, true);
}

TEST_F(MindDataTestTensorDE, MSTensorDataType) {
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({2, 3}), DataType(DataType::DE_FLOAT32));
  auto ms_tensor = std::shared_ptr<MSTensor>(new DETensor(t));
  ASSERT_EQ(ms_tensor->data_type() == mindspore::TypeId::kNumberTypeFloat32, true);
}

TEST_F(MindDataTestTensorDE, MSTensorMutableData) {
  std::vector<float> x = {2.5, 2.5, 2.5, 2.5};
  std::shared_ptr<Tensor> t;
  Tensor::CreateFromVector(x, TensorShape({2, 2}), &t);
  auto ms_tensor = std::shared_ptr<MSTensor>(new DETensor(t));
  float *data = static_cast<float *>(ms_tensor->MutableData());
  std::vector<float> tensor_vec(data, data + ms_tensor->ElementsNum());
  ASSERT_EQ(x == tensor_vec, true);
}

TEST_F(MindDataTestTensorDE, MSTensorCreateFromMemory) {
  std::vector<float> x = {2.5, 2.5, 2.5, 2.5};
  auto mem_tensor = DETensor::CreateFromMemory(mindspore::TypeId::kNumberTypeFloat32, {2, 2}, &x[0]);
  ASSERT_EQ(mem_tensor->data_type() == mindspore::TypeId::kNumberTypeFloat32, true);
}
