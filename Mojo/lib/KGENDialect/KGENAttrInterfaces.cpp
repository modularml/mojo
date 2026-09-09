//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// ParameterAttr
//===----------------------------------------------------------------------===//

namespace {
struct IntegerParameterAttr
    : public ParameterAttr::ExternalModel<IntegerParameterAttr, IntegerAttr> {
  bool isConstant(Attribute attr) const { return true; }
  bool isLessThan(Attribute attr, Attribute rhs) const {
    auto intAttr = cast<IntegerAttr>(rhs);
    return cast<IntegerAttr>(attr).getValue().slt(intAttr.getValue());
  }
};

struct FloatParameterAttr
    : public ParameterAttr::ExternalModel<FloatParameterAttr, FloatAttr> {
  bool isConstant(Attribute attr) const { return true; }
  bool isLessThan(Attribute attr, Attribute rhs) const {
    auto fpAttr = cast<FloatAttr>(rhs);
    return cast<FloatAttr>(attr).getValue() < fpAttr.getValue();
  }
};

struct StringParameterAttr
    : public ParameterAttr::ExternalModel<StringParameterAttr, StringAttr> {
  bool isConstant(Attribute attr) const { return true; }
  bool isLessThan(Attribute attr, Attribute rhs) const {
    auto strAttr = cast<StringAttr>(rhs);
    return cast<StringAttr>(attr).getValue() < strAttr.getValue();
  }
};

struct TypeParameterAttr
    : public ParameterAttr::ExternalModel<TypeParameterAttr, TypeAttr> {
  bool isConstant(Attribute attr) const {
    return !isParameterizedType(cast<TypeAttr>(attr).getValue());
  }
};

struct PointerParameterAttr
    : public ParameterAttr::ExternalModel<PointerParameterAttr, PointerAttr> {
  bool isConstant(Attribute attr) const {
    return !isParameterizedType(cast<PointerAttr>(attr).getType());
  }
};

struct MemRefParameterAttr
    : public ParameterAttr::ExternalModel<MemRefParameterAttr, MemRefAttr> {
  bool isConstant(Attribute attr) const {
    return !isParameterizedType(cast<MemRefAttr>(attr).getType());
  }
};

struct StoreToMemParameterAttr
    : public ParameterAttr::ExternalModel<StoreToMemParameterAttr,
                                          StoreToMemAttr> {
  bool isConstant(Attribute attr) const {
    // If the value is concrete, then the type must be too.
    return ParameterAttr::isSimpleConstant(
        cast<StoreToMemAttr>(attr).getValue());
  }
  bool isLessThan(Attribute attr, Attribute rhs) const {
    auto storeToMem = cast<StoreToMemAttr>(rhs);
    return ParameterAttr::compare(cast<StoreToMemAttr>(attr).getValue(),
                                  storeToMem.getValue());
  }
};
} // namespace

void KGENDialect::injectAttrInterfaces() {
  IntegerAttr::attachInterface<IntegerParameterAttr>(*getContext());
  FloatAttr::attachInterface<FloatParameterAttr>(*getContext());
  StringAttr::attachInterface<StringParameterAttr>(*getContext());
  TypeAttr::attachInterface<TypeParameterAttr>(*getContext());
  PointerAttr::attachInterface<PointerParameterAttr>(*getContext());
  MemRefAttr::attachInterface<MemRefParameterAttr>(*getContext());
  StoreToMemAttr::attachInterface<StoreToMemParameterAttr>(*getContext());
}

bool ParameterAttr::isSimpleConstant(Attribute attr) {
  // Check for an interface.
  if (auto itf = ::dyn_cast<ParameterAttr>(attr))
    return itf.isConstant();

  // Handle UninitMemAttr.  It cannot conform to ParameterAttr because it is
  // KGEN level and the interpreter is a lower level dialect.
  if (auto uninitMem = ::dyn_cast<UninitMemAttr>(attr))
    return !isParameterizedType(uninitMem.getType());

  // Otherwise, assume the attribute is not a simple constant.
  return false;
}

// Compare sub-elements of a type or attribute lexicographically (first all the
// attributes, then all the types).
template <typename T>
static bool compareSubElementsImpl(T lhs, T rhs) {
  SmallVector<Attribute> lhsSubAttrs;
  SmallVector<Type> lhsSubTypes;
  SmallVector<Attribute> rhsSubAttrs;
  SmallVector<Type> rhsSubTypes;
  lhs.walkImmediateSubElements(
      [&](Attribute attr) { lhsSubAttrs.push_back(attr); },
      [&](Type type) { lhsSubTypes.push_back(type); });
  rhs.walkImmediateSubElements(
      [&](Attribute attr) { rhsSubAttrs.push_back(attr); },
      [&](Type type) { rhsSubTypes.push_back(type); });
  // If the attr lists aren't equal, the shorter list is "less than" the longer
  // list.
  if (lhsSubAttrs.size() != rhsSubAttrs.size())
    return lhsSubAttrs.size() < rhsSubAttrs.size();
  // Similarly for the type lists.
  if (lhsSubTypes.size() != rhsSubTypes.size())
    return lhsSubTypes.size() < rhsSubTypes.size();

  // If all equal, perform element-wise comparison.
  // First compare the attributes pairwise using ParameterAttr::compare.
  for (auto [lhsAttr, rhsAttr] : llvm::zip(lhsSubAttrs, rhsSubAttrs))
    if (lhsAttr != rhsAttr)
      return ParameterAttr::compare(lhsAttr, rhsAttr);

  // Then compare the types by recursing until we see attributes.
  for (auto [lhsType, rhsType] : llvm::zip(lhsSubTypes, rhsSubTypes)) {
    if (lhsType == rhsType)
      continue;

    // If the types are not even the same kind, order by kind name first.
    const mlir::AbstractType &lhsAbs = lhsType.getAbstractType();
    const mlir::AbstractType &rhsAbs = rhsType.getAbstractType();
    if (&lhsAbs != &rhsAbs)
      return lhsAbs.getName() < rhsAbs.getName();

    return compareSubElementsImpl(lhsType, rhsType);
  }

  // At this point, we bottomed out at leaf attributes that didn't implement
  // `isLessThan`, or we bottomed out at atomic types (no sub-elements).
  return false;
}

bool ParameterAttr::compareSubElements(Attribute lhs, Attribute rhs) {
  return compareSubElementsImpl(lhs, rhs);
}

bool ParameterAttr::compareSubElements(Type lhs, Type rhs) {
  return compareSubElementsImpl(lhs, rhs);
}

bool ParameterAttr::compare(Attribute lhs, Attribute rhs) {
  // Simplify the code below - we never have to care about exactly equal values.
  if (lhs == rhs)
    return false;

  // Look through sugar - it shouldn't affect ordering.
  lhs = SugarAttr::strip(lhs);
  rhs = SugarAttr::strip(rhs);

  // All non-constant expressions are "less than" a constant, since they appear
  // on the right. We handle all simple constants consistently here: they can
  // never occur in the same expression since they have different types.
  if (isSimpleConstant(rhs)) {
    if (!isSimpleConstant(lhs))
      return true;
  } else if (isSimpleConstant(lhs)) {
    return false;
  }

  // Parameter operator expressions are always on the left.
  if (::isa<ParamOperatorAttr>(lhs)) {
    if (!::isa<ParamOperatorAttr>(rhs))
      return true;
  } else if (::isa<ParamOperatorAttr>(rhs)) {
    return false;
  }

  // If the attributes are not even the same kind, order by kind name first.
  const mlir::AbstractAttribute &lhsAbs = lhs.getAbstractAttribute();
  const mlir::AbstractAttribute &rhsAbs = rhs.getAbstractAttribute();
  if (&lhsAbs != &rhsAbs)
    return lhsAbs.getName() < rhsAbs.getName();

  // Check for an interface.
  if (auto itf = ::dyn_cast<ParameterAttr>(lhs))
    return itf.isLessThan(rhs);

  // Otherwise, compare sub-elements lexicographically.
  return compareSubElementsImpl(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/KGENAttrInterfaces.cpp.inc"
