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

#include "pybind_api/ir/cell_py.h"
#include <string>

#include "pybind_api/api_register.h"
#include "abstract/abstract_value.h"
#include "pipeline/jit/parse/python_adapter.h"

namespace mindspore {
void CellPy::AddAttr(CellPtr cell, const std::string &name, const py::object &obj) {
  std::string attr_name = name;
  ValuePtr converted_ret = nullptr;
  if (py::isinstance<py::module>(obj)) {
    MS_LOG(EXCEPTION) << "Cell set_attr failed, attr should not be py::module";
  }
  bool converted = parse::ConvertData(obj, &converted_ret, true);
  if (!converted) {
    MS_LOG(DEBUG) << "Attribute convert error with type: " << std::string(py::str(obj));
  } else {
    MS_LOG(DEBUG) << cell->ToString() << " add attr " << attr_name << converted_ret->ToString();
    cell->AddAttr(attr_name, converted_ret);
  }
}
// Define python 'Cell' class.
REGISTER_PYBIND_DEFINE(Cell, ([](const py::module *m) {
                         (void)py::class_<Cell, std::shared_ptr<Cell>>(*m, "Cell_")
                           .def(py::init<std::string &>())
                           .def("__str__", &Cell::ToString)
                           .def("_add_attr", &CellPy::AddAttr, "Add Cell attr.")
                           .def("_del_attr", &Cell::DelAttr, "Delete Cell attr.")
                           .def(
                             "construct", []() { MS_LOG(EXCEPTION) << "we should define `construct` for all `cell`."; },
                             "construct")
                           .def(py::pickle(
                             [](const Cell &cell) {  // __getstate__
                               /* Return a tuple that fully encodes the state of the object */
                               return py::make_tuple(py::str(cell.name()));
                             },
                             [](const py::tuple &tup) {  // __setstate__
                               if (tup.size() != 1) {
                                 throw std::runtime_error("Invalid state!");
                               }
                               /* Create a new C++ instance */
                               Cell data(tup[0].cast<std::string>());
                               return data;
                             }));
                       }));
}  // namespace mindspore
