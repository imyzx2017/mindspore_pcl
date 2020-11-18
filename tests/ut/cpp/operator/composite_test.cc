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

#include "common/common_test.h"
#include "ir/anf.h"
#include "ir/value.h"
#include "frontend/operator/composite/composite.h"
#include "frontend/operator/ops.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "abstract/abstract_function.h"
#include "debug/trace.h"

namespace mindspore {
using Shape = abstract::Shape;

using AbstractScalar = abstract::AbstractScalar;
using AbstractScalarPtr = abstract::AbstractScalarPtr;

using AbstractSlice = abstract::AbstractSlice;
using AbstractSlicePtr = abstract::AbstractSlicePtr;

using AbstractTuple = abstract::AbstractTuple;
using AbstractTuplePtr = abstract::AbstractTuplePtr;

using AbstractTensor = abstract::AbstractTensor;
using AbstractTensorPtr = abstract::AbstractTensorPtr;

using AbstractNone = abstract::AbstractNone;
using AbstractAttribute = abstract::AbstractAttribute;
using AnalysisEngine = abstract::AnalysisEngine;
using AnalysisEnginePtr = abstract::AnalysisEnginePtr;

class TestComposite : public UT::Common {
 public:
  virtual void SetUp();
  virtual void TearDown();

  AnalysisEnginePtr engine_;
};

void TestComposite::SetUp() {
  // init resource
  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager();
  engine_ = std::make_shared<AnalysisEngine>(abstract::GetPrimEvaluatorConstructors(), graph_manager);
}

void TestComposite::TearDown() {
  // destroy resource
}

class UTCompositeUtils {
 public:
  static AbstractTensorPtr ArrayInt32Of(std::initializer_list<int64_t> shp) {
    auto ele = std::make_shared<AbstractScalar>(kAnyValue, kInt64);
    return std::make_shared<AbstractTensor>(ele, std::make_shared<Shape>(shp));
  }
  static FuncGraphPtr MakeFuncGraph(const MetaFuncGraphPtr &metaFuncGraphPtr, size_t nparam) {
    FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
    std::vector<AnfNodePtr> inputs;
    inputs.push_back(NewValueNode(metaFuncGraphPtr));
    for (size_t i = 0; i < nparam; i++) {
      inputs.push_back(func_graph->add_parameter());
    }
    CNodePtr cnode_prim = func_graph->NewCNode(inputs);
    inputs.clear();
    inputs.push_back(NewValueNode(prim::kPrimReturn));
    inputs.push_back(cnode_prim);
    CNodePtr cnode_return = func_graph->NewCNode(inputs);
    func_graph->set_return(cnode_return);
    return func_graph;
  }
};

TEST_F(TestComposite, test_TupleSlice_arg_two_numbers) {
  MetaFuncGraphPtr tupleSlicePtr = std::make_shared<prim::TupleSlice>("tuple_slice");
  FuncGraphPtr tupleSliceGraphPtr = UTCompositeUtils::MakeFuncGraph(tupleSlicePtr, 3);

  AbstractBasePtrList eles;
  auto tensor = UTCompositeUtils::ArrayInt32Of({2, 3, 4});
  size_t tuple_size = 6;
  for (size_t i = 0; i < tuple_size; i++) {
    eles.push_back(tensor);
  }
  auto tuple_tensor = std::make_shared<AbstractTuple>(eles);
  auto start_index = std::make_shared<AbstractScalar>(static_cast<int64_t>(1));
  auto stop_index = std::make_shared<AbstractScalar>(static_cast<int64_t>(5));
  AbstractBasePtrList args_spec_list = {tuple_tensor, start_index, stop_index};

  try {
    engine_->Run(tupleSliceGraphPtr, args_spec_list);
    FAIL() << "Excepted exception :Args type is wrong";
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("TupleSlice input args size should be 2, but got 3") != std::string::npos);
  } catch (...) {
    FAIL() << "Excepted exception :Args type is wrong";
  }
}

TEST_F(TestComposite, test_TupleSlice_arg_one_number) {
  MetaFuncGraphPtr tupleSlicePtr = std::make_shared<prim::TupleSlice>("tuple_slice");
  FuncGraphPtr tupleSliceGraphPtr = UTCompositeUtils::MakeFuncGraph(tupleSlicePtr, 2);

  AbstractBasePtrList eles;
  auto tensor = UTCompositeUtils::ArrayInt32Of({2, 3, 4});
  size_t tuple_size = 6;
  for (size_t i = 0; i < tuple_size; i++) {
    eles.push_back(tensor);
  }
  auto tuple_tensor = std::make_shared<AbstractTuple>(eles);
  auto start_index = std::make_shared<AbstractScalar>(static_cast<int64_t>(1));
  AbstractBasePtrList args_spec_list = {tuple_tensor, start_index};

  try {
    trace::ClearTraceStack();
    engine_->Run(tupleSliceGraphPtr, args_spec_list);
    FAIL() << "Excepted exception: Args type is wrong";
  } catch (pybind11::type_error const &err) {
    ASSERT_TRUE(true);
  } catch (std::runtime_error const &err) {
    if (std::strstr(err.what(), "TypeError") != nullptr) {
      ASSERT_TRUE(true);
    } else {
      FAIL() << "Excepted exception: Args type is wrong, message: " << err.what();
    }
  } catch (...) {
    FAIL() << "Excepted exception: Args type is wrong";
  }
}

TEST_F(TestComposite, test_TupleSlice_arg_slice) {
  std::shared_ptr<py::scoped_interpreter> env = parse::python_adapter::set_python_scoped();
  MetaFuncGraphPtr tupleSlicePtr = std::make_shared<prim::TupleSlice>("tuple_slice");
  FuncGraphPtr tupleSliceGraphPtr = UTCompositeUtils::MakeFuncGraph(tupleSlicePtr, 2);

  AbstractBasePtrList eles;
  auto tensor = UTCompositeUtils::ArrayInt32Of({2, 3, 4});
  size_t tuple_size = 6;
  for (size_t i = 0; i < tuple_size; i++) {
    eles.push_back(tensor);
  }
  auto tuple_tensor = std::make_shared<AbstractTuple>(eles);
  auto start_index = std::make_shared<AbstractScalar>(static_cast<int64_t>(1));
  auto stop_index = std::make_shared<AbstractScalar>(static_cast<int64_t>(6));
  auto step = std::make_shared<AbstractScalar>(static_cast<int64_t>(2));
  auto slice = std::make_shared<AbstractSlice>(start_index, stop_index, step);
  AbstractBasePtrList args_spec_list = {tuple_tensor, slice};

  AbstractTuplePtr ret = dyn_cast<AbstractTuple>(engine_->Run(tupleSliceGraphPtr, args_spec_list).inferred->abstract());
  if (ret == nullptr) {
    FAIL() << "Cast ret to abstract tuple failed.";
  }
  size_t real = ret->size();
  size_t expect = 3;
  ASSERT_EQ(real, expect);
}

TEST_F(TestComposite, test_TupleSlice_arg_slice_step_none) {
  MetaFuncGraphPtr tupleSlicePtr = std::make_shared<prim::TupleSlice>("tuple_slice");
  FuncGraphPtr tupleSliceGraphPtr = UTCompositeUtils::MakeFuncGraph(tupleSlicePtr, 2);

  AbstractBasePtrList eles;
  auto tensor = UTCompositeUtils::ArrayInt32Of({2, 3, 4});
  size_t tuple_size = 6;
  for (size_t i = 0; i < tuple_size; i++) {
    eles.push_back(tensor);
  }
  auto tuple_tensor = std::make_shared<AbstractTuple>(eles);
  auto start_index = std::make_shared<AbstractScalar>(static_cast<int64_t>(1));
  auto stop_index = std::make_shared<AbstractScalar>(static_cast<int64_t>(5));
  auto step = std::make_shared<AbstractNone>();
  auto slice = std::make_shared<AbstractSlice>(start_index, stop_index, step);
  AbstractBasePtrList args_spec_list = {tuple_tensor, slice};

  AbstractTuplePtr ret = dyn_cast<AbstractTuple>(engine_->Run(tupleSliceGraphPtr, args_spec_list).inferred->abstract());
  if (ret == nullptr) {
    FAIL() << "Cast ret to abstract tuple failed.";
  }
  size_t real = ret->size();
  size_t expect = 4;
  ASSERT_EQ(real, expect);
}

TEST_F(TestComposite, test_TupleSlice_arg_slice_step_negative) {
  MetaFuncGraphPtr tupleSlicePtr = std::make_shared<prim::TupleSlice>("tuple_slice");
  FuncGraphPtr tupleSliceGraphPtr = UTCompositeUtils::MakeFuncGraph(tupleSlicePtr, 2);

  AbstractBasePtrList eles;
  auto tensor = UTCompositeUtils::ArrayInt32Of({2, 3, 4});
  size_t tuple_size = 6;
  for (size_t i = 0; i < tuple_size; i++) {
    eles.push_back(tensor);
  }
  auto tuple_tensor = std::make_shared<AbstractTuple>(eles);
  auto start_index = std::make_shared<AbstractNone>();
  auto stop_index = std::make_shared<AbstractNone>();
  auto step = std::make_shared<AbstractScalar>(static_cast<int64_t>(-1));
  auto slice = std::make_shared<AbstractSlice>(start_index, stop_index, step);
  AbstractBasePtrList args_spec_list = {tuple_tensor, slice};

  AbstractTuplePtr ret = dyn_cast<AbstractTuple>(engine_->Run(tupleSliceGraphPtr, args_spec_list).inferred->abstract());
  if (ret == nullptr) {
    FAIL() << "Cast ret to abstract tuple failed.";
  }
  size_t real = ret->size();
  size_t expect = 6;
  ASSERT_EQ(real, expect);
}

TEST_F(TestComposite, test_TupleSlice_arg_slice_step_positive) {
  MetaFuncGraphPtr tupleSlicePtr = std::make_shared<prim::TupleSlice>("tuple_slice");
  FuncGraphPtr tupleSliceGraphPtr = UTCompositeUtils::MakeFuncGraph(tupleSlicePtr, 2);

  AbstractBasePtrList eles;
  auto tensor = UTCompositeUtils::ArrayInt32Of({2, 3, 4});
  size_t tuple_size = 6;
  for (size_t i = 0; i < tuple_size; i++) {
    eles.push_back(tensor);
  }
  auto tuple_tensor = std::make_shared<AbstractTuple>(eles);
  auto start_index = std::make_shared<AbstractScalar>(static_cast<int64_t>(-2));
  auto stop_index = std::make_shared<AbstractNone>();
  auto step = std::make_shared<AbstractScalar>(static_cast<int64_t>(-1));
  auto slice = std::make_shared<AbstractSlice>(start_index, stop_index, step);
  AbstractBasePtrList args_spec_list = {tuple_tensor, slice};

  AbstractTuplePtr ret = dyn_cast<AbstractTuple>(engine_->Run(tupleSliceGraphPtr, args_spec_list).inferred->abstract());
  if (ret == nullptr) {
    FAIL() << "Cast ret to abstract tuple failed.";
  }
  size_t real = ret->size();
  size_t expect = 5;
  ASSERT_EQ(real, expect);
}

TEST_F(TestComposite, test_UnpackCall_3args) {
  MetaFuncGraphPtr unPackCallPtr = std::make_shared<prim::UnpackCall>("UnPackCall");
  FuncGraphPtr unPackCallGraphPtr = UTCompositeUtils::MakeFuncGraph(unPackCallPtr, 3);

  auto fn_arg= std::make_shared<abstract::PrimitiveAbstractClosure>(prim::kPrimMakeTuple);
  AbstractTensorPtr tensor = UTCompositeUtils::ArrayInt32Of({2, 3, 4});
  AbstractBasePtrList eles;
  for (size_t i = 0; i < 6; i++) {
    eles.push_back(tensor);
  }
  AbstractTuplePtr tensor_tuple = std::make_shared<AbstractTuple>(eles);
  AbstractTensorPtr arr_x = UTCompositeUtils::ArrayInt32Of({2, 3, 4});
  AbstractTensorPtr arr_y = UTCompositeUtils::ArrayInt32Of({2, 3, 4});
  AbstractTensorPtr arr_z = UTCompositeUtils::ArrayInt32Of({2, 3, 4});
  std::vector<AbstractAttribute> tensor_map{{"x", arr_x}, {"y", arr_y}, {"z", arr_z}};
  abstract::AbstractDictionaryPtr tensor_dict = std::make_shared<abstract::AbstractDictionary>(tensor_map);

  AbstractBasePtrList args_spec_list = {fn_arg, tensor_tuple, tensor_dict};
  AbstractTuplePtr ret = dyn_cast<AbstractTuple>(engine_->Run(unPackCallGraphPtr, args_spec_list).inferred->abstract());
  if (ret == nullptr) {
    FAIL() << "Cast ret to abstract tuple failed.";
  }
  size_t real = ret->size();
  size_t expect = 9;
  ASSERT_EQ(real, expect);
}

TEST_F(TestComposite, test_UnpackCall_5args) {
  MetaFuncGraphPtr unPackCallPtr = std::make_shared<prim::UnpackCall>("UnPackCall");
  FuncGraphPtr unPackCallGraphPtr = UTCompositeUtils::MakeFuncGraph(unPackCallPtr, 5);

  auto fn_arg = std::make_shared<abstract::PrimitiveAbstractClosure>(prim::kPrimMakeTuple);
  AbstractTensorPtr tensor = UTCompositeUtils::ArrayInt32Of({2, 3, 4});
  AbstractBasePtrList eles;
  for (size_t i = 0; i < 6; i++) {
    eles.push_back(tensor);
  }
  AbstractTuplePtr tensor_tuple = std::make_shared<AbstractTuple>(eles);
  AbstractTensorPtr arr_x = UTCompositeUtils::ArrayInt32Of({2, 3, 4});
  AbstractTensorPtr arr_y = UTCompositeUtils::ArrayInt32Of({2, 3, 4});
  AbstractTensorPtr arr_z = UTCompositeUtils::ArrayInt32Of({2, 3, 4});
  std::vector<AbstractAttribute> tensor_map{{"x", arr_x}, {"y", arr_y}, {"z", arr_z}};
  abstract::AbstractDictionaryPtr tensor_dict = std::make_shared<abstract::AbstractDictionary>(tensor_map);

  AbstractBasePtrList args_spec_list = {fn_arg, tensor_dict, tensor_tuple, tensor_dict, tensor_tuple};
  AbstractTuplePtr ret = dyn_cast<AbstractTuple>(engine_->Run(unPackCallGraphPtr, args_spec_list).inferred->abstract());
  if (ret == nullptr) {
    FAIL() << "Cast ret to abstract tuple failed.";
  }
  size_t real = ret->size();
  size_t expect = 18;
  ASSERT_EQ(real, expect);
}

TEST_F(TestComposite, test_ZipOperation) {
  MetaFuncGraphPtr zip_op = std::make_shared<prim::ZipOperation>("zip_op");
  FuncGraphPtr zip_op_graph = UTCompositeUtils::MakeFuncGraph(zip_op, 1);

  AbstractBasePtrList eles;
  auto tensor = UTCompositeUtils::ArrayInt32Of({2, 3, 4});
  size_t tuple_size = 3;
  for (size_t i = 0; i < tuple_size; i++) {
    eles.push_back(tensor);
  }
  auto tuple = std::make_shared<AbstractTuple>(eles);
  AbstractBasePtrList args_spec_list = {tuple};

  AbstractTuplePtr ret = dyn_cast<AbstractTuple>(engine_->Run(zip_op_graph, args_spec_list).inferred->abstract());
  if (ret == nullptr) {
    FAIL() << "Cast ret to abstract tuple failed.";
  }
  size_t real = ret->size();
  size_t expect = 3;
  ASSERT_EQ(real, expect);
}
}  // namespace mindspore
