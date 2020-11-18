/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "utils/convert_utils_py.h"

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <list>
#include <utility>
#include <cfloat>

#include "abstract/abstract_value.h"
#include "pipeline/jit/parse/parse.h"
#include "pipeline/jit/parse/parse_base.h"
#include "ir/value.h"
#include "ir/tensor.h"
#include "ir/param_info.h"
#include "pybind_api/ir/base_ref_py.h"
#include "utils/ms_context.h"

namespace mindspore {
py::object BuiltinsToPyData(const Any &value);
py::object BuiltinsToPyData(const BaseRef &value);
py::object VectorToPyData(const Any &value);
py::object VectorRefToPyData(const VectorRef &value);

py::object TensorToPyData(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->NeedWait()) {
    py::gil_scoped_release release;
    tensor->Wait();
  }
  py::tuple v(1);
  v[0] = tensor;
  return v[0];
}

py::object ValuePtrToPyData(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(EXCEPTION) << "value is null";
  }
  py::object ret;
  if (value->isa<Int8Imm>()) {
    MS_LOG(DEBUG) << "int8";
    py::int_ v = value->cast<Int8ImmPtr>()->value();
    ret = v;
  } else if (value->isa<Int16Imm>()) {
    MS_LOG(DEBUG) << "int16";
    py::int_ v = value->cast<Int16ImmPtr>()->value();
    ret = v;
  } else if (value->isa<Int32Imm>()) {
    MS_LOG(DEBUG) << "int32";
    py::int_ v = value->cast<Int32ImmPtr>()->value();
    ret = v;
  } else if (value->isa<Int64Imm>()) {
    MS_LOG(DEBUG) << "int64";
    py::int_ v = value->cast<Int64ImmPtr>()->value();
    ret = v;
  } else if (value->isa<UInt8Imm>()) {
    MS_LOG(DEBUG) << "uint8";
    py::int_ v = value->cast<UInt8ImmPtr>()->value();
    ret = v;
  } else if (value->isa<UInt16Imm>()) {
    MS_LOG(DEBUG) << "uint16";
    py::int_ v = value->cast<UInt16ImmPtr>()->value();
    ret = v;
  } else if (value->isa<UInt32Imm>()) {
    MS_LOG(DEBUG) << "uint32";
    py::int_ v = value->cast<UInt32ImmPtr>()->value();
    ret = v;
  } else if (value->isa<UInt64Imm>()) {
    MS_LOG(DEBUG) << "uint64";
    py::int_ v = value->cast<UInt64ImmPtr>()->value();
    ret = v;
  } else if (value->isa<BoolImm>()) {
    MS_LOG(DEBUG) << "bool";
    py::bool_ v = value->cast<BoolImmPtr>()->value();
    ret = v;
  } else if (value->isa<FP64Imm>()) {
    MS_LOG(DEBUG) << "double";
    py::float_ v = value->cast<FP64ImmPtr>()->value();
    ret = v;
  } else if (value->isa<FP32Imm>()) {
    MS_LOG(DEBUG) << "float";
    py::float_ v = value->cast<FP32ImmPtr>()->value();
    ret = v;
  } else if (value->isa<StringImm>()) {
    MS_LOG(DEBUG) << "String";
    py::str v = value->cast<StringImmPtr>()->value();
    ret = v;
  } else if (value->isa<tensor::Tensor>()) {
    MS_LOG(DEBUG) << "tensor";
    auto tensor_ptr = value->cast<tensor::TensorPtr>();
    ret = TensorToPyData(tensor_ptr);
  } else if (value->isa<tensor::MetaTensor>()) {
    MS_LOG(DEBUG) << "MetaTensor";
    py::tuple v(1);
    v[0] = value->cast<tensor::MetaTensorPtr>();
    ret = v[0];
  } else if (value->isa<RefKey>()) {
    MS_LOG(DEBUG) << "RefKey";
    py::tuple v(1);
    v[0] = value->cast<RefKeyPtr>();
    ret = v[0];
  } else if (value->isa<ValueTuple>()) {
    MS_LOG(DEBUG) << "tuple";
    auto value_tuple = value->cast<ValueTuplePtr>()->value();
    py::tuple rets(value_tuple.size());

    size_t i = 0;
    for (auto &v : value_tuple) {
      rets[i] = ValuePtrToPyData(v);
      i++;
    }
    ret = rets;
  } else if (value->isa<ValueList>()) {
    MS_LOG(DEBUG) << "list";
    auto value_list = value->cast<ValueListPtr>()->value();
    py::list rets(value_list.size());

    size_t i = 0;
    for (auto &v : value_list) {
      rets[i] = ValuePtrToPyData(v);
      i++;
    }
    ret = rets;
  } else if (value->isa<Ellipsis>()) {
    ret = py::ellipsis();
  } else if (value->isa<ValueSlice>()) {
    auto slice = value->cast<ValueSlicePtr>();
    auto start = ValuePtrToPyData(slice->start());
    auto end = ValuePtrToPyData(slice->stop());
    auto step = ValuePtrToPyData(slice->step());
    ret = parse::python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_PARSE_CLASS_SLICE, start, end,
                                          step);
  } else if (value->isa<Type>()) {
    py::tuple v(1);
    v[0] = value->cast<TypePtr>();
    ret = v[0];
  } else if (value->isa<AnyValue>()) {
    ret = py::none();
  } else if (value->isa<None>()) {
    ret = py::none();
  } else if (value->isa<FuncGraph>()) {
    // FuncGraph is not used in the backend, return None
    ret = py::none();
  } else {
    MS_LOG(EXCEPTION) << "Unsupported convert value: " << value->ToString() << " to a PyData.";
  }
  return ret;
}

py::object AnyToPyData(const Any &value) {
  py::object ret;
  MS_LOG(DEBUG) << "AnyToPyData " << value.GetString();
  if (value.is<int>() || value.is<float>() || value.is<double>() || value.is<bool>()) {
    ret = BuiltinsToPyData(value);
  } else if (value.is<ValuePtr>()) {
    MS_LOG(DEBUG) << "ValuePtr";
    ValuePtr v = value.cast<ValuePtr>();
    ret = ValuePtrToPyData(v);
  } else if (value.is<tensor::TensorPtr>()) {
    MS_LOG(DEBUG) << "tensor";
    auto tensor_ptr = value.cast<tensor::TensorPtr>();
    ret = TensorToPyData(tensor_ptr);
  } else if (value.is<py::object>()) {
    MS_LOG(DEBUG) << "py obj";
    ret = value.cast<py::object>();
  } else if (value.is<std::vector<tensor::TensorPtr>>() || value.is<std::vector<Any>>()) {
    ret = VectorToPyData(value);
  } else if (value.is<std::list<Any>>()) {
    MS_LOG(DEBUG) << "list_any";
    auto value_list = value.cast<std::list<Any>>();
    py::list rets = py::list();
    for (auto &v : value_list) {
      rets.append(AnyToPyData(v));
    }
    ret = rets;
  } else if (value.is<std::vector<Any>>()) {
    auto value_list = value.cast<std::vector<Any>>();
    py::tuple rets(value_list.size());
    for (size_t i = 0; i < value_list.size(); i++) {
      rets[i] = AnyToPyData(value_list[i]);
    }
    ret = rets;
  } else if (value.is<TypePtr>()) {
    py::tuple v(1);
    v[0] = value.cast<TypePtr>();
    ret = v[0];
  } else {
    MS_LOG(EXCEPTION) << "value is not support type";
  }
  return ret;
}

py::object BaseRefToPyData(const BaseRef &value) {
  py::object ret;
  MS_LOG(DEBUG) << "BaseRefToPyData " << value.ToString();
  if (utils::isa<int>(value) || utils::isa<float>(value) || utils::isa<double>(value) || utils::isa<bool>(value)) {
    ret = BuiltinsToPyData(value);
  } else if (utils::isa<ValuePtr>(value)) {
    MS_LOG(DEBUG) << "ValuePtr";
    ValuePtr v = utils::cast<ValuePtr>(value);
    ret = ValuePtrToPyData(v);
  } else if (utils::isa<tensor::TensorPtr>(value)) {
    MS_LOG(DEBUG) << "tensor";
    auto tensor_ptr = utils::cast<tensor::TensorPtr>(value);
    ret = TensorToPyData(tensor_ptr);
  } else if (utils::isa<PyObjectRef>(value)) {
    MS_LOG(DEBUG) << "py obj";
    PyObjectRef py_ref = utils::cast<PyObjectRef>(value);
    ret = py_ref.object_;
  } else if (utils::isa<VectorRef>(value)) {
    auto vec_ref = utils::cast<VectorRef>(value);
    ret = VectorRefToPyData(vec_ref);
  } else if (utils::isa<TypePtr>(value)) {
    py::tuple v(1);
    v[0] = utils::cast<TypePtr>(value);
    ret = v[0];
  } else {
    MS_LOG(EXCEPTION) << "value is not support type";
  }
  return ret;
}

py::object BuiltinsToPyData(const Any &value) {
  if (value.is<int>()) {
    MS_LOG(DEBUG) << "int";
    py::int_ ret = value.cast<int>();
    return std::move(ret);
  } else if (value.is<float>()) {
    MS_LOG(DEBUG) << "float";
    py::float_ ret = value.cast<float>();
    return std::move(ret);
  } else if (value.is<double>()) {
    MS_LOG(DEBUG) << "double";
    py::float_ ret = value.cast<double>();
    return std::move(ret);
  } else {
    MS_LOG(DEBUG) << "bool";
    py::bool_ ret = value.cast<bool>();
    return std::move(ret);
  }
}

py::object BuiltinsToPyData(const BaseRef &value) {
  if (utils::isa<int>(value)) {
    MS_LOG(DEBUG) << "int";
    py::int_ ret = utils::cast<int>(value);
    return std::move(ret);
  } else if (utils::isa<float>(value)) {
    MS_LOG(DEBUG) << "float";
    py::float_ ret = utils::cast<float>(value);
    return std::move(ret);
  } else if (utils::isa<double>(value)) {
    MS_LOG(DEBUG) << "double";
    py::float_ ret = utils::cast<double>(value);
    return std::move(ret);
  } else {
    MS_LOG(DEBUG) << "bool";
    py::bool_ ret = utils::cast<bool>(value);
    return std::move(ret);
  }
}

py::object VectorToPyData(const Any &value) {
  py::object ret;
  if (value.is<std::vector<tensor::TensorPtr>>()) {
    MS_LOG(DEBUG) << "vector_tensor";
    std::vector<tensor::TensorPtr> outputs;
    outputs = value.cast<std::vector<tensor::TensorPtr>>();
    py::tuple tensor_tuple(outputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      tensor_tuple[i] = *outputs[i];
    }
    ret = tensor_tuple;
  } else {
    MS_LOG(DEBUG) << "vector_any";
    auto value_list = value.cast<std::vector<Any>>();
    py::tuple any_tuple = py::tuple(value_list.size());
    size_t i = 0;
    for (auto &v : value_list) {
      any_tuple[i] = AnyToPyData(v);
      i++;
    }
    ret = any_tuple;
  }
  return ret;
}

py::object VectorRefToPyData(const VectorRef &value_list) {
  py::object ret;
  MS_LOG(DEBUG) << "vector_ref";
  size_t value_size = value_list.size();
  auto ref_tuple = py::tuple(value_size);
  for (size_t i = 0; i < value_size; i++) {
    ref_tuple[i] = BaseRefToPyData(value_list[i]);
  }
  ret = ref_tuple;
  return ret;
}

AbstractBasePtr PyListDtype2AbstractTensor(const py::object &shape_obj, const py::object &type_obj,
                                           const py::object &min_shape, const py::object &max_shape) {
  if ((py::isinstance<py::list>(shape_obj) || py::isinstance<py::tuple>(shape_obj)) && py::isinstance<Type>(type_obj)) {
    auto ret_vec = shape_obj.cast<ShapeVector>();
    auto ret_dtype = type_obj.cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(ret_dtype);
    // if the size of shape list is empty, return an scalar abstract
    if (ret_vec.empty() && (!ret_dtype->isa<TensorType>())) {
      abstract::AbstractScalarPtr abs_scalar = std::make_shared<abstract::AbstractScalar>(kAnyValue, ret_dtype);
      return abs_scalar;
    }
    AbstractBasePtr tensor = nullptr;
    ShapeVector min_shape_vec;
    ShapeVector max_shape_vec;
    if (!min_shape.is_none()) {
      min_shape_vec = min_shape.cast<ShapeVector>();
    }
    if (!max_shape.is_none()) {
      max_shape_vec = max_shape.cast<ShapeVector>();
    }
    auto ret_shape = std::make_shared<abstract::Shape>(ret_vec, min_shape_vec, max_shape_vec);
    if (ret_dtype->isa<TensorType>()) {
      auto tensor_type = type_obj.cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(tensor_type);
      auto element = std::make_shared<abstract::AbstractScalar>(kAnyValue, tensor_type->element());
      tensor = std::make_shared<abstract::AbstractTensor>(element, ret_shape);
    } else {
      auto element = std::make_shared<abstract::AbstractScalar>(kAnyValue, ret_dtype);
      tensor = std::make_shared<abstract::AbstractTensor>(element, ret_shape);
    }
    return tensor;
  } else if (py::isinstance<py::tuple>(shape_obj) && py::isinstance<py::tuple>(type_obj)) {
    py::tuple shape_tuple = shape_obj.cast<py::tuple>();
    py::tuple typeid_tuple = type_obj.cast<py::tuple>();
    AbstractBasePtrList ptr_list;
    for (size_t it = 0; it < shape_tuple.size(); ++it) {
      auto tensor_it = PyListDtype2AbstractTensor(shape_tuple[it], typeid_tuple[it]);
      ptr_list.push_back(tensor_it);
    }
    auto tuple = std::make_shared<abstract::AbstractTuple>(ptr_list);
    return tuple;
  } else if (py::isinstance<py::list>(shape_obj) && py::isinstance<py::list>(type_obj)) {
    py::list shape_list = shape_obj.cast<py::list>();
    py::list typeid_list = type_obj.cast<py::list>();
    AbstractBasePtrList ptr_list;
    for (size_t it = 0; it < shape_list.size(); ++it) {
      auto tensor_it = PyListDtype2AbstractTensor(shape_list[it], typeid_list[it]);
      ptr_list.push_back(tensor_it);
    }
    auto list = std::make_shared<abstract::AbstractList>(ptr_list);
    return list;
  } else if (shape_obj.is_none() && type_obj.is_none()) {
    // AbstractNone indicates there is no output for this CNode node.
    auto abstract_none = std::make_shared<abstract::AbstractNone>();
    return abstract_none;
  } else {
    // When sparse enabled, the undetermined might be raised and eliminated in opt passes
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    bool enable_sparse = context->get_param<bool>(MS_CTX_ENABLE_SPARSE);
    if (enable_sparse) {
      return std::make_shared<abstract::AbstractUndetermined>();
    }
    MS_LOG(EXCEPTION) << "Python evaluator return invalid shape or type. " << (std::string)py::str(type_obj);
  }
}
bool IsGraphOutputValueNodeOrParameter(const AnfNodePtr &output, const py::tuple &args,
                                       const std::shared_ptr<py::object> &ret_val) {
  if (output->isa<ValueNode>()) {
    MS_LOG(INFO) << "Graph's output is a constant. No need to execute.";
    ValuePtr value = GetValueNode(output);
    *ret_val = ValuePtrToPyData(value);
    return true;
  }

  // Adapter will transform values in __init__() and construct() to parameters, this could cause
  // inputs (a.k.a args in current function) size less than parameters'.
  if (output->isa<Parameter>()) {
    MS_LOG(INFO) << "Graph's output is a parameter. If all params are inputs, no need to execute.";
    // Find the right parameter as ret_val.
    auto func_graph = output->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    auto params = func_graph->parameters();
    if ((args.size() + func_graph->hyper_param_count()) != params.size()) {
      MS_LOG(EXCEPTION) << "Input size " << args.size() << " add Parameter count " << func_graph->hyper_param_count()
                        << " not equal to graph input size " << params.size() << ", let graph to be executed.";
    }

    auto it = std::find(params.begin(), params.end(), output);
    if (it == params.end()) {
      MS_EXCEPTION(UnknownError) << "When graph output is Parameter,  it should be found in graph parameters";
    }
    size_t index = it - params.cbegin();
    if (index >= args.size() + func_graph->hyper_param_count()) {
      MS_EXCEPTION(UnknownError) << "Index " << index << " equal or larger than args size " << args.size()
                                 << " add Parameter count " << func_graph->hyper_param_count() << ".";
    }
    if (index < args.size()) {
      *ret_val = args[index];
    } else {
      auto param = dyn_cast<Parameter>(params[index]);
      MS_EXCEPTION_IF_NULL(param);
      if (!param->has_default()) {
        MS_LOG(EXCEPTION) << "Can not determine value of Parameter " << index << " (" << param->name() << ")";
      }
      auto tensor = param->default_param();
      *ret_val = py::cast(tensor);
    }
    return true;
  }
  return false;
}
}  // namespace mindspore
