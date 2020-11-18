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
#include <iostream>
#include <memory>
#include <vector>

#include "minddata/dataset/core/client.h"
#include "common/common.h"
#include "utils/ms_utils.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "minddata/dataset/engine/datasetops/source/clue_op.h"
#include "minddata/dataset/util/status.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestCLUEOp : public UT::DatasetOpTesting {

};

TEST_F(MindDataTestCLUEOp, TestCLUEBasic) {
  // Start with an empty execution tree
  auto tree = std::make_shared<ExecutionTree>();

  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/testCLUE/afqmc/train.json";
  std::map<std::string, std::string> key_map;
  key_map["sentence1"] = "sentence1";
  key_map["sentence2"] = "sentence2";
  key_map["label"] = "label";

  std::shared_ptr<ClueOp> op;
  ClueOp::Builder builder;
  builder.SetClueFilesList({dataset_path})
      .SetRowsPerBuffer(16)
      .SetNumWorkers(16)
      .SetOpConnectorSize(2)
      .SetColsKeyMap(key_map);

  Status rc = builder.Build(&op);
  ASSERT_TRUE(rc.IsOk());

  rc = tree->AssociateNode(op);
  ASSERT_TRUE(rc.IsOk());

  rc = tree->AssignRoot(op);
  ASSERT_TRUE(rc.IsOk());

  MS_LOG(INFO) << "Launching tree and begin iteration.";
  rc = tree->Prepare();
  ASSERT_TRUE(rc.IsOk());

  rc = tree->Launch();
  ASSERT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  int row_count = 0;
  while (!tensor_list.empty()) {
    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << *tensor_list[i] << std::endl;
      MS_LOG(INFO) << "Tensor print: " << ss.str() << ".";
    }

    rc = di.FetchNextTensorRow(&tensor_list);
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }

  ASSERT_EQ(row_count, 3);
}

TEST_F(MindDataTestCLUEOp, TestTotalRows) {
  std::string tf_file1 = datasets_root_path_ + "/testCLUE/afqmc/train.json";
  std::string tf_file2 = datasets_root_path_ + "/testCLUE/afqmc/dev.json";
  std::vector<std::string> files;
  files.push_back(tf_file1);
  int64_t total_rows = 0;
  ClueOp::CountAllFileRows(files, &total_rows);
  ASSERT_EQ(total_rows, 3);
  files.clear();

  files.push_back(tf_file2);
  ClueOp::CountAllFileRows(files, &total_rows);
  ASSERT_EQ(total_rows, 3);
  files.clear();

  files.push_back(tf_file1);
  files.push_back(tf_file2);
  ClueOp::CountAllFileRows(files, &total_rows);
  ASSERT_EQ(total_rows, 6);
  files.clear();
}
