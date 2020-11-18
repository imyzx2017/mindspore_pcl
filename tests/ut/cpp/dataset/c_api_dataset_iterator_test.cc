
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
#include "common/common.h"
#include "minddata/dataset/include/datasets.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestIteratorEmptyColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIteratorEmptyColumn.";
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create a Rename operation on ds
  ds = ds->Rename({"image", "label"}, {"col1", "col2"});
  EXPECT_NE(ds, nullptr);

  // No columns are specified, use all columns
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);
  TensorShape expect0({32, 32, 3});
  TensorShape expect1({});

  uint64_t i = 0;
  while (row.size() != 0) {
    MS_LOG(INFO) << "row[0]:" << row[0]->shape() << ", row[1]:" << row[1]->shape();
    EXPECT_EQ(expect0, row[0]->shape());
    EXPECT_EQ(expect1, row[1]->shape());
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestIteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIteratorOneColumn.";
  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", RandomSampler(false, 4));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 2;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // Only select "image" column and drop others
  std::vector<std::string> columns = {"image"};
  std::shared_ptr<Iterator> iter = ds->CreateIterator(columns);
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);
  TensorShape expect({2, 28, 28, 1});

  uint64_t i = 0;
  while (row.size() != 0) {
    for (auto &v : row) {
      MS_LOG(INFO) << "image shape:" << v->shape();
      EXPECT_EQ(expect, v->shape());
    }
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestIteratorReOrder) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIteratorReOrder.";
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", SequentialSampler(false, 4));
  EXPECT_NE(ds, nullptr);

  // Create a Take operation on ds
  ds = ds->Take(2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // Reorder "image" and "label" column
  std::vector<std::string> columns = {"label", "image"};
  std::shared_ptr<Iterator> iter = ds->CreateIterator(columns);
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);
  TensorShape expect0({32, 32, 3});
  TensorShape expect1({});

  // Check if we will catch "label" before "image" in row
  std::vector<std::string> expect = {"label", "image"};
  uint64_t i = 0;
  while (row.size() != 0) {
    MS_LOG(INFO) << "row[0]:" << row[0]->shape() << ", row[1]:" << row[1]->shape();
    EXPECT_EQ(expect1, row[0]->shape());
    EXPECT_EQ(expect0, row[1]->shape());
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestIteratorTwoColumns) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIteratorTwoColumns.";
  // Create a VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, false, SequentialSampler(0, 4));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // Only select "image" and "bbox" column
  std::vector<std::string> columns = {"image", "bbox"};
  std::shared_ptr<Iterator> iter = ds->CreateIterator(columns);
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);
  std::vector<TensorShape> expect = {TensorShape({173673}), TensorShape({1, 4}),   TensorShape({173673}),
                                     TensorShape({1, 4}),   TensorShape({147025}), TensorShape({1, 4}),
                                     TensorShape({211653}), TensorShape({1, 4})};

  uint64_t i = 0;
  uint64_t j = 0;
  while (row.size() != 0) {
    MS_LOG(INFO) << "row[0]:" << row[0]->shape() << ", row[1]:" << row[1]->shape();
    EXPECT_EQ(2, row.size());
    EXPECT_EQ(expect[j++], row[0]->shape());
    EXPECT_EQ(expect[j++], row[1]->shape());
    iter->GetNextRow(&row);
    i++;
    j = (j == expect.size()) ? 0 : j;
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIteratorOneColumn.";
  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", RandomSampler(false, 4));
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<Iterator> iter = ds->CreateIterator(columns);
  EXPECT_EQ(iter, nullptr);
}
