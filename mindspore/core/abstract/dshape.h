/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_ABSTRACT_DSHAPE_H_
#define MINDSPORE_CORE_ABSTRACT_DSHAPE_H_

#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <typeindex>
#include <memory>
#include <algorithm>

#include "utils/log_adapter.h"
#include "base/base.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace abstract {
class BaseShape;
using BaseShapePtr = std::shared_ptr<BaseShape>;
using BaseShapePtrList = std::vector<BaseShapePtr>;

class BaseShape : public Base {
 public:
  BaseShape() = default;
  ~BaseShape() override = default;

  MS_DECLARE_PARENT(BaseShape, Base)
  virtual bool operator==(const BaseShape &other) const;
  bool operator!=(const BaseShape &other) const;
  std::size_t hash() const override { return tid(); }

  // return a deep copy
  virtual BaseShapePtr Clone() const = 0;
  virtual void Broaden() {}
};

class NoShape : public BaseShape {
 public:
  MS_DECLARE_PARENT(NoShape, BaseShape)
  BaseShapePtr Clone() const override { return std::make_shared<NoShape>(); }
  std::string ToString() const override { return type_name(); }
};
extern const std::shared_ptr<NoShape> kNoShape;

class Shape : public BaseShape {
 public:
  static const int64_t SHP_ANY = -1;
  Shape() : shape_() {}
  Shape(const std::initializer_list<int64_t> &list) : shape_(list) {}
  explicit Shape(const ShapeVector &list) : shape_(list) {}
  Shape(const ShapeVector &list, const ShapeVector &min_shape, const ShapeVector &max_shape)
      : shape_(list), min_shape_(min_shape), max_shape_(max_shape) {}
  ~Shape() override = default;
  MS_DECLARE_PARENT(Shape, BaseShape)
  std::string ToString() const override;
  std::string DumpText() const override;
  bool operator==(const BaseShape &other) const override;
  BaseShapePtr Clone() const override { return std::make_shared<Shape>(shape_, min_shape_, max_shape_); }
  void Broaden() override;
  ShapeVector &shape() { return shape_; }
  ShapeVector &min_shape() { return min_shape_; }
  ShapeVector &max_shape() { return max_shape_; }

  ShapeVector shape_;      // use SHP_ANY to implement the any shape in python
  ShapeVector min_shape_;  // record mininum length for each dynamic dimention
  ShapeVector max_shape_;  // record maximum length for each dynamic dimention
};
using ShapePtr = std::shared_ptr<Shape>;
using ShapePtrList = std::vector<ShapePtr>;

class SequeueShape : public BaseShape {
 public:
  SequeueShape() : p_shapes_() {}
  explicit SequeueShape(const BaseShapePtrList &shapes) : p_shapes_(shapes) {}
  ~SequeueShape() override = default;
  MS_DECLARE_PARENT(SequeueShape, BaseShape)

  std::string ToString() const override;
  BaseShapePtrList ElementsClone() const;

  template <typename T>
  bool SequeueEqual(const BaseShape &other) const;

  const BaseShapePtrList &shape() const { return p_shapes_; }
  size_t size() const { return p_shapes_.size(); }
  const BaseShapePtr operator[](std::size_t dim) const { return p_shapes_[dim]; }

 protected:
  BaseShapePtrList p_shapes_;  // shape list of each elements
};
using SequeueShapePtr = std::shared_ptr<SequeueShape>;

class TupleShape : public SequeueShape {
 public:
  TupleShape() : SequeueShape() {}
  explicit TupleShape(const BaseShapePtrList &shapes) : SequeueShape(shapes) {}
  ~TupleShape() override = default;
  MS_DECLARE_PARENT(TupleShape, SequeueShape)

  std::string ToString() const override { return type_name() + "(" + SequeueShape::ToString() + ")"; }

  BaseShapePtr Clone() const override { return std::make_shared<TupleShape>(ElementsClone()); }

  bool operator==(const BaseShape &other) const override { return SequeueEqual<TupleShape>(other); }
};
using TupleShapePtr = std::shared_ptr<TupleShape>;

class ListShape : public SequeueShape {
 public:
  ListShape() : SequeueShape() {}
  explicit ListShape(const BaseShapePtrList &shapes) : SequeueShape(shapes) {}
  ~ListShape() override = default;
  MS_DECLARE_PARENT(ListShape, SequeueShape)

  std::string ToString() const override { return type_name() + "[" + SequeueShape::ToString() + "]"; }

  BaseShapePtr Clone() const override { return std::make_shared<ListShape>(SequeueShape::ElementsClone()); }

  bool operator==(const BaseShape &other) const override { return SequeueEqual<ListShape>(other); }
};
using ListShapePtr = std::shared_ptr<ListShape>;
}  // namespace abstract
}  // namespace mindspore

#endif  // MINDSPORE_CORE_ABSTRACT_DSHAPE_H_
