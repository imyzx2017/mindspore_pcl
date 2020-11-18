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

#include "utils/primitive_utils.h"

#include <memory>

#include "pipeline/jit/parse/python_adapter.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "utils/convert_utils_py.h"
#include "pybind_api/ir/base_ref_py.h"

namespace mindspore {
py::function GetBpropFunctionByObj(py::object obj) {
  static const std::string get_bprop_fn = "get_bprop_fn";
  static const std::string ad_module = "mindspore.ops._grad";
  py::function fn = parse::python_adapter::GetPyFn(ad_module, get_bprop_fn)(obj);
  return fn;
}

py::function GetBpropFunction(std::string name) {
  auto fn = GetBpropFunctionByObj(py::str(name));
  return fn;
}

py::function GetComputeFunction(std::string name) {
  static const std::string module = "mindspore._extends.builtin_operations";
  py::module mod = py::module::import(common::SafeCStr(module));
  if (!py::hasattr(mod, common::SafeCStr(name))) {
    PyErr_SetString(PyExc_NotImplementedError, common::SafeCStr(name));
    // If raise AttributeError, user can't understand. This case need raise NotImplementedError.
    throw(py::error_already_set());
  }
  py::object fn = mod.attr(common::SafeCStr(name));
  return fn;
}

py::tuple ConvertDatatoPyTuple(const VectorRef &args) {
  auto py_args = py::tuple(args.size());
  size_t i = 0;
  for (auto &arg : args) {
    py_args[i] = BaseRefToPyData(arg);
    MS_LOG(DEBUG) << "arg:" << i << ":" << arg.ToString();
    i++;
  }
  return py_args;
}

BaseRef RunComputeFunction(const PrimitivePtr &prim, const VectorRef &args) {
  auto func = GetComputeFunction(prim->name());
  if (py::isinstance<py::none>(func)) {
    MS_LOG(EXCEPTION) << prim->name() << " 's compute function run failed, please check whether it is not implemented";
  }
  auto py_args = ConvertDatatoPyTuple(args);
  py::object obj = func(*py_args);
  return std::make_shared<PyObjectRef>(obj);
}
}  // namespace mindspore
