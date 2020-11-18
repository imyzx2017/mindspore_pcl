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
#include "minddata/dataset/core/global_context.h"

#include "mindspore/core/ir/dtype/type_id.h"

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestRandomDatasetBasic1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetBasic1.";

  // Create a RandomDataset
  std::shared_ptr<SchemaObj> schema = Schema();
  schema->add_column("image", mindspore::TypeId::kNumberTypeUInt8, {2});
  schema->add_column("label", mindspore::TypeId::kNumberTypeUInt8, {1});
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  ds = ds->Repeat(4);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  // Check if RandomDataOp read correct columns
  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    MS_LOG(INFO) << "Tensor label shape: " << label->shape();

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 200);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetGetters.";

  // Create a RandomDataset
  std::shared_ptr<SchemaObj> schema = Schema();
  schema->add_column("image", mindspore::TypeId::kNumberTypeUInt8, {2});
  schema->add_column("label", mindspore::TypeId::kNumberTypeUInt8, {1});
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  std::vector<std::string> column_names = {"image", "label"};
  EXPECT_EQ(ds->GetDatasetSize(), 50);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

TEST_F(MindDataTestPipeline, TestRandomDatasetBasic2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetBasic2.";

  // Create a RandomDataset
  std::shared_ptr<Dataset> ds = RandomData(10);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(1);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  ds = ds->Repeat(2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  // Check if RandomDataOp read correct columns
  uint64_t i = 0;
  while (row.size() != 0) {
    // If no schema specified, RandomData will generate random columns
    // So we don't check columns here
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomDatasetBasic3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetBasic3.";

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);

  std::string SCHEMA_FILE = datasets_root_path_ + "/testTFTestAllTypes/datasetSchema.json";
  std::shared_ptr<SchemaObj> schema = Schema(SCHEMA_FILE);
  std::shared_ptr<Dataset> ds = RandomData(0, schema);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  ds = ds->Repeat(2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  // Check if RandomDataOp read correct columns
  uint64_t i = 0;
  while (row.size() != 0) {
    auto col_sint16 = row["col_sint16"];
    auto col_sint32 = row["col_sint32"];
    auto col_sint64 = row["col_sint64"];
    auto col_float = row["col_float"];
    auto col_1d = row["col_1d"];
    auto col_2d = row["col_2d"];
    auto col_3d = row["col_3d"];
    auto col_binary = row["col_binary"];

    // validate shape
    ASSERT_EQ(col_sint16->shape(), TensorShape({1}));
    ASSERT_EQ(col_sint32->shape(), TensorShape({1}));
    ASSERT_EQ(col_sint64->shape(), TensorShape({1}));
    ASSERT_EQ(col_float->shape(), TensorShape({1}));
    ASSERT_EQ(col_1d->shape(), TensorShape({2}));
    ASSERT_EQ(col_2d->shape(), TensorShape({2, 2}));
    ASSERT_EQ(col_3d->shape(), TensorShape({2, 2, 2}));
    ASSERT_EQ(col_binary->shape(), TensorShape({1}));

    // validate Rank
    ASSERT_EQ(col_sint16->Rank(), 1);
    ASSERT_EQ(col_sint32->Rank(), 1);
    ASSERT_EQ(col_sint64->Rank(), 1);
    ASSERT_EQ(col_float->Rank(), 1);
    ASSERT_EQ(col_1d->Rank(), 1);
    ASSERT_EQ(col_2d->Rank(), 2);
    ASSERT_EQ(col_3d->Rank(), 3);
    ASSERT_EQ(col_binary->Rank(), 1);

    // validate type
    ASSERT_EQ(col_sint16->type(), DataType::DE_INT16);
    ASSERT_EQ(col_sint32->type(), DataType::DE_INT32);
    ASSERT_EQ(col_sint64->type(), DataType::DE_INT64);
    ASSERT_EQ(col_float->type(), DataType::DE_FLOAT32);
    ASSERT_EQ(col_1d->type(), DataType::DE_INT64);
    ASSERT_EQ(col_2d->type(), DataType::DE_INT64);
    ASSERT_EQ(col_3d->type(), DataType::DE_INT64);
    ASSERT_EQ(col_binary->type(), DataType::DE_UINT8);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 984);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

TEST_F(MindDataTestPipeline, TestRandomDatasetBasic4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetBasic4.";

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);

  std::string SCHEMA_FILE = datasets_root_path_ + "/testTFTestAllTypes/datasetSchema.json";
  std::shared_ptr<Dataset> ds = RandomData(0, SCHEMA_FILE);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  ds = ds->Repeat(2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  // Check if RandomDataOp read correct columns
  uint64_t i = 0;
  while (row.size() != 0) {
    auto col_sint16 = row["col_sint16"];
    auto col_sint32 = row["col_sint32"];
    auto col_sint64 = row["col_sint64"];
    auto col_float = row["col_float"];
    auto col_1d = row["col_1d"];
    auto col_2d = row["col_2d"];
    auto col_3d = row["col_3d"];
    auto col_binary = row["col_binary"];

    // validate shape
    ASSERT_EQ(col_sint16->shape(), TensorShape({1}));
    ASSERT_EQ(col_sint32->shape(), TensorShape({1}));
    ASSERT_EQ(col_sint64->shape(), TensorShape({1}));
    ASSERT_EQ(col_float->shape(), TensorShape({1}));
    ASSERT_EQ(col_1d->shape(), TensorShape({2}));
    ASSERT_EQ(col_2d->shape(), TensorShape({2, 2}));
    ASSERT_EQ(col_3d->shape(), TensorShape({2, 2, 2}));
    ASSERT_EQ(col_binary->shape(), TensorShape({1}));

    // validate Rank
    ASSERT_EQ(col_sint16->Rank(), 1);
    ASSERT_EQ(col_sint32->Rank(), 1);
    ASSERT_EQ(col_sint64->Rank(), 1);
    ASSERT_EQ(col_float->Rank(), 1);
    ASSERT_EQ(col_1d->Rank(), 1);
    ASSERT_EQ(col_2d->Rank(), 2);
    ASSERT_EQ(col_3d->Rank(), 3);
    ASSERT_EQ(col_binary->Rank(), 1);

    // validate type
    ASSERT_EQ(col_sint16->type(), DataType::DE_INT16);
    ASSERT_EQ(col_sint32->type(), DataType::DE_INT32);
    ASSERT_EQ(col_sint64->type(), DataType::DE_INT64);
    ASSERT_EQ(col_float->type(), DataType::DE_FLOAT32);
    ASSERT_EQ(col_1d->type(), DataType::DE_INT64);
    ASSERT_EQ(col_2d->type(), DataType::DE_INT64);
    ASSERT_EQ(col_3d->type(), DataType::DE_INT64);
    ASSERT_EQ(col_binary->type(), DataType::DE_UINT8);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 984);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

TEST_F(MindDataTestPipeline, TestRandomDatasetBasic5) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetBasic5.";

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);

  std::string SCHEMA_FILE = datasets_root_path_ + "/testTFTestAllTypes/datasetSchema.json";
  std::shared_ptr<Dataset> ds = RandomData(0, SCHEMA_FILE, {"col_sint32", "col_sint64", "col_1d"});
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  ds = ds->Repeat(2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  // Check if RandomDataOp read correct columns
  uint64_t i = 0;
  while (row.size() != 0) {
    EXPECT_EQ(row.size(), 3);

    auto col_sint32 = row["col_sint32"];
    auto col_sint64 = row["col_sint64"];
    auto col_1d = row["col_1d"];

    // validate shape
    ASSERT_EQ(col_sint32->shape(), TensorShape({1}));
    ASSERT_EQ(col_sint64->shape(), TensorShape({1}));
    ASSERT_EQ(col_1d->shape(), TensorShape({2}));

    // validate Rank
    ASSERT_EQ(col_sint32->Rank(), 1);
    ASSERT_EQ(col_sint64->Rank(), 1);
    ASSERT_EQ(col_1d->Rank(), 1);

    // validate type
    ASSERT_EQ(col_sint32->type(), DataType::DE_INT32);
    ASSERT_EQ(col_sint64->type(), DataType::DE_INT64);
    ASSERT_EQ(col_1d->type(), DataType::DE_INT64);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 984);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

TEST_F(MindDataTestPipeline, TestRandomDatasetBasic6) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetBasic6.";

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);

  std::string SCHEMA_FILE = datasets_root_path_ + "/testTFTestAllTypes/datasetSchema.json";
  std::shared_ptr<Dataset> ds = RandomData(10, nullptr, {"col_sint32", "col_sint64", "col_1d"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  // Check if RandomDataOp read correct columns
  uint64_t i = 0;
  while (row.size() != 0) {
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

TEST_F(MindDataTestPipeline, TestRandomDatasetBasic7) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetBasic7.";

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);

  std::string SCHEMA_FILE = datasets_root_path_ + "/testTFTestAllTypes/datasetSchema.json";
  std::shared_ptr<Dataset> ds = RandomData(10, "", {"col_sint32", "col_sint64", "col_1d"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  // Check if RandomDataOp read correct columns
  uint64_t i = 0;
  while (row.size() != 0) {
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

TEST_F(MindDataTestPipeline, TestRandomDatasetDuplicateColumnName) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetDuplicateColumnName.";

  // Create a RandomDataset
  std::shared_ptr<SchemaObj> schema = Schema();
  schema->add_column("image", mindspore::TypeId::kNumberTypeUInt8, {2});
  schema->add_column("label", mindspore::TypeId::kNumberTypeUInt8, {1});
  std::shared_ptr<Dataset> ds = RandomData(50, schema, {"image", "image"});
  // Expect failure: duplicate column names
  EXPECT_EQ(ds->CreateIterator(), nullptr);
}
