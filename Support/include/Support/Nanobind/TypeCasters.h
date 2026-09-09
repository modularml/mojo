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

#ifndef SUPPORT_NANOBIND_TYPECASTERS_H
#define SUPPORT_NANOBIND_TYPECASTERS_H

#include "Support/AssertStream.h"
#include "Support/ErrorOr.h"
#include "Support/ML/DType.h"
#include "Support/Nanobind/SequenceView.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"      // IWYU pragma: keep (type caster)
#include "nanobind/stl/string_view.h" // IWYU pragma: keep (type caster)
#include "nanobind/stl/unique_ptr.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <nanobind/stl/detail/nb_list.h>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace M::Graph::Python {

/// Let bound types register themselves for type hooks.
void registerTypeID(mlir::TypeID, const std::type_info *);
const std::type_info *lookupTypeID(mlir::TypeID);

/// Wrap an OpBuilder->create call.
/// - We can't bind templatized functions on OpBuilder
/// - Instead, each Op has their own constructors that take a builder
/// - Internally, these constructors use `create_op` to delegate to the builder
template <typename Op, typename... Args>
auto create_op(nanobind::handle_t<mlir::OpBuilder> builder, Args... args) {
  return Op::create(*nanobind::cast<mlir::OpBuilder *>(builder),
                    std::forward<Args>(args)...);
}

/// Nanobind doesn't support multiple inheritance, but we want to correctly
/// model MLIR Interface types. Conveniently, MLIR interfaces are basically
/// Python protocols already.
///
/// Rather than bind them directly, we create custom type casters and type hooks
/// that check whether a type matches an interface, and then rely on type hooks
/// for  downcasting.
///
/// For stub eneration, we bind a `Protocol` wrapped type, which then can't
/// actually be created, but will have the correct types and definitions for the
/// interface.
template <typename T>
class Protocol {};

/// A non-implemented method with a specific signature for nanobind typing.
template <typename Return, typename... Args>
Return not_implemented(Args &&...) {
  throw nb::type_error("not implemented");
}
} // namespace M::Graph::Python

namespace NB_NAMESPACE {
namespace detail {

namespace {
//===----------------------------------------------------------------------===//
// is_attribute_interface
//===----------------------------------------------------------------------===//

/// Trait for detecting subclasses of `mlir::AttributeBase
template <typename T>
struct is_attribute_interface_base {
private:
  template <typename C, typename Concrete, typename... Traits>
  static std::true_type
  test(const mlir::AttributeInterface<Concrete, Traits...> *);

  template <typename C>
  static std::false_type test(...);

public:
  using type = decltype(test<T>(std::declval<T *>()));
  static constexpr bool value = type::value;
};

/// Trait function for detecting subclasses of `mlir::AttributeBase
template <typename T>
constexpr bool is_attribute_interface() {
  return is_attribute_interface_base<T>::value;
}

/// Trait for detecting subclasses of `mlir::AttributeBase
template <typename T>
struct is_type_interface_base {
private:
  template <typename C, typename Concrete, typename... Traits>
  static std::true_type test(const mlir::TypeInterface<Concrete, Traits...> *);

  template <typename C>
  static std::false_type test(...);

public:
  using type = decltype(test<T>(std::declval<T *>()));
  static constexpr bool value = type::value;
};

/// Trait function for detecting subclasses of `mlir::AttributeBase
template <typename T>
constexpr bool is_type_interface() {
  return is_type_interface_base<T>::value;
}
} // namespace

//===----------------------------------------------------------------------===//
// TypeCasters for LLVM and MLIR types
//===----------------------------------------------------------------------===//

template <typename T, typename Delegate>
struct delegate_caster : type_caster_base<T> {
  using Caster = make_caster<Delegate>;
  NB_TYPE_CASTER(T, Caster::Name)
  Caster caster;

  static T convert_to(Delegate &d) {
    static_assert(std::is_convertible_v<Delegate, T>);
    return d;
  }

  static Delegate convert_from(T &t) noexcept {
    static_assert(std::is_convertible_v<T, Delegate>);
    return t;
  }

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (caster.from_python(src, flags, cleanup)) {
      value = convert_to(caster.value);
      return true;
    }
    return false;
  }
  static handle from_cpp(T t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return Caster::from_cpp(convert_from(t), policy, cleanup);
  }
};

/// Casts object <-> MLIRContext.
/// Only passes by pointer; we never want to take or store an MLIRContext by
/// value, which will deep copy all of the storage data.
template <>
struct type_caster<::mlir::MLIRContext> {
protected:
  ::mlir::MLIRContext *value;

public:
  static constexpr auto Name = const_name("Context");

  template <typename T>
  using Cast = ::mlir::MLIRContext *;

  operator ::mlir::MLIRContext *() { return value; }

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    std::optional<nb::object> capsule = mlirApiObjectToCapsule(src);
    if (!capsule)
      return false;
    value = unwrap(mlirPythonCapsuleToContext(capsule->ptr()));
    return !mlirContextIsNull(wrap(value));
  }

  static handle from_cpp(::mlir::MLIRContext *t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    static nb::handle contextCAPICreate =
        nb::module_::import_("max.mlir")
            .attr("Context")
            .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR);
    nb::handle capsule = mlirPythonContextToCapsule(wrap(t));
    return contextCAPICreate(capsule).release();
  }
};

/// Casts MlirLocation <-> mlir::Location.
/// Delegate to the `MlirLocation` upstream type caster.
template <>
struct type_caster<::mlir::Location> {
  static constexpr auto Name = const_name("Location");
  template <typename T>
  using Cast = movable_cast_t<::mlir::Location>;
  using Caster = make_caster<MlirLocation>;
  Caster caster;

  /// Since `mlir::Location` doesn't have a default constructor, store the value
  /// as an `optional` until it is successfully cast.
  std::optional<::mlir::Location> value;

  explicit operator ::mlir::Location *() { return &*value; }
  explicit operator ::mlir::Location &() { return *value; }
  explicit operator ::mlir::Location &&() { return std::move(*value); }

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (!caster.from_python(src, flags, cleanup)) {
      return false;
    }
    value = unwrap(caster.value);
    return true;
  }

  template <typename T>
  static constexpr bool can_cast() {
    return Caster::can_cast<T>();
  }

  static handle from_cpp(::mlir::Location t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return Caster::from_cpp(wrap(t), policy, cleanup);
  }
};

/// Casts str <-> llvm::StringRef.
/// Delegate implementation to the `std::string_view` caster.
template <>
struct type_caster<::llvm::StringRef>
    : delegate_caster<llvm::StringRef, std::string_view> {};

/// Casts int <-> llvm::APInt.
/// There's likely a _much_ better way to do this by directly passing
/// the int bytes between bignum implementations. For now stringifying and
/// parsing.
template <>
struct type_caster<::llvm::APInt> {
  NB_TYPE_CASTER(::llvm::APInt, const_name("int"))

  bool from_python(handle_t<nb::int_> src, uint8_t flags,
                   cleanup_list *cleanup) noexcept {
    auto base10 = nb::cast<std::string>(nb::str(src));
    llvm::StringRef(base10).getAsInteger(10, value);
    return true;
  }

  static handle from_cpp(::llvm::APInt t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    std::string base10;
    llvm::raw_string_ostream(base10) << t;
    // _Very_ important to release here.
    // - In most of our type casters we defer to another caster, or explicitly
    // pass ownership of a C++ type via a `unique_ptr`
    // - Here we are returning an `nb::object` directly. `release` gives
    // ownership to python.
    // - Otherwise, you can end up double-freeing memory from Python's interned
    // longs, which is a spooky and very hard to track down memory safety bug.
    return nb::int_(make_caster<std::string>::from_cpp(base10, policy, cleanup))
        .release();
  }
};

/// Casts float <-> llvm::APFloat.
template <>
struct type_caster<::llvm::APFloat> {
  static constexpr auto Name = const_name("float");
  template <typename T>
  using Cast = movable_cast_t<llvm::APFloat>;
  using Caster = make_caster<double>;
  Caster caster;

  /// Since `APFloat` doesn't have a default constructor, store the value
  /// as an `optional` until it is successfully cast.
  std::optional<::llvm::APFloat> value;

  explicit operator ::llvm::APFloat *() { return &*value; }
  explicit operator ::llvm::APFloat &() { return *value; }
  explicit operator ::llvm::APFloat &&() { return std::move(*value); }

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (!caster.from_python(src, flags, cleanup)) {
      return false;
    }
    value = llvm::APFloat(caster.value);
    return true;
  }

  template <typename T>
  static constexpr bool can_cast() {
    return Caster::can_cast<T>();
  }

  static handle from_cpp(::llvm::APFloat f, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return Caster::from_cpp(f.convertToDouble(), policy, cleanup);
  }
};

template <typename Entry>
struct type_caster<::llvm::SmallVector<Entry>>
    : list_caster<::llvm::SmallVector<Entry>, Entry> {};

// Type caster to convert from list <-> SmallVectorImpl
template <typename Entry>
struct type_caster<::llvm::SmallVectorImpl<Entry>> {
  using Underlying = ::llvm::SmallVector<Entry>;
  using Value = ::llvm::SmallVectorImpl<Entry>;

  template <typename T_>
  using Cast = movable_cast_t<T_>;

  static constexpr auto Name = make_caster<Underlying>::Name;

  using Caster = make_caster<Underlying>;
  Caster caster;

  explicit operator Value *() { return &caster.value; }
  explicit operator Value &() { return (Value &)caster.value; }
  explicit operator Value &&() { return (Value &&)caster.value; }

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    return !caster.from_python(src, flags, cleanup);
  }

  static handle from_cpp(Underlying &&underlying, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return make_caster<Underlying>::from_cpp(Underlying(std::move(underlying)),
                                             policy, cleanup);
  }
};

/// Casts str <-> llvm::Twine.
/// Twines are meant to be temporary values, so we treat them
/// more like values than stringrefs. We don't support taking pointers to them.
/// - Strings passed from Python -> C++ as a Twine will not copy since we can
/// expose a single stringref as a Twine
/// - Twines passed from C++ -> Python copy into a contiguous string and are
/// passed to Python.
template <>
struct type_caster<::llvm::Twine> {
protected:
  std::string_view value;

public:
  static constexpr auto Name = const_name("str");

  template <typename T>
  using Cast = ::llvm::Twine;

  operator ::llvm::Twine() { return value; }
  using Caster = make_caster<std::string_view>;
  Caster caster;

  bool from_python(handle_t<nb::str> src, uint8_t flags,
                   cleanup_list *cleanup) noexcept {
    if (!caster.from_python(src, flags, cleanup)) {
      return false;
    }
    value = caster.value;
    return true;
  }

  static handle from_cpp(::llvm::Twine t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return make_caster<std::string>::from_cpp(t.str(), policy, cleanup);
  }
};

/// Casts object <-> DType.
template <>
struct type_caster<::M::DType> {
  NB_TYPE_CASTER(::M::DType, const_name("max._core.dtype.DType"))
  using DTypeCaster = make_caster<M::DType::Cases>;
  DTypeCaster dtypeCaster;

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (dtypeCaster.from_python(src, flags, cleanup)) {
      value = M::DType(dtypeCaster.value);
      return true;
    }
    return false;
  }
  static handle from_cpp(::M::DType t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return DTypeCaster::from_cpp(static_cast<M::DType::Cases>(t.getValue()),
                                 policy, cleanup);
  }
};

/// Casts AttributeInterface <-> python.
/// - General type caster for any attribute interfaces
/// - Allows any type implementing the interface to be passed Python -> C++
/// - Downcasts to the concrete attribute implementation type when passed C++ ->
/// Python
template <typename AttributeInterface>
struct type_caster<
    AttributeInterface,
    std::enable_if_t<is_attribute_interface<AttributeInterface>(), int>> {
  NB_TYPE_CASTER(
      AttributeInterface,
      make_caster<M::Graph::Python::Protocol<AttributeInterface>>::Name)

  using Caster = make_caster<mlir::Attribute>;
  Caster caster;

  bool from_python(handle_t<::mlir::Attribute> src, uint8_t flags,
                   cleanup_list *cleanup) noexcept {
    if (!caster.from_python(src, flags, cleanup))
      return false;
    value =
        ::mlir::dyn_cast_or_null<AttributeInterface>(mlir::Attribute(caster));
    return bool(value);
  }

  static handle from_cpp(AttributeInterface ar, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return make_caster<std::unique_ptr<::mlir::Attribute>>::from_cpp(
        std::make_unique<::mlir::Attribute>(ar), policy, cleanup);
  }
};

/// Casts TypeInterface <-> python.
/// - General type caster for any type interfaces
/// - Allows any type implementing the interface to be passed Python ->
/// C++
/// - Downcasts to the concrete type implementation type when passed C++
/// -> Python
template <typename TypeInterface>
struct type_caster<TypeInterface,
                   std::enable_if_t<is_type_interface<TypeInterface>(), int>> {
  NB_TYPE_CASTER(TypeInterface,
                 make_caster<M::Graph::Python::Protocol<TypeInterface>>::Name)

  using Caster = make_caster<mlir::Type>;
  Caster caster;

  bool from_python(handle_t<::mlir::Type> src, uint8_t flags,
                   cleanup_list *cleanup) noexcept {
    if (!caster.from_python(src, flags, cleanup))
      return false;
    value = ::mlir::dyn_cast_or_null<TypeInterface>(mlir::Type(caster));
    return bool(value);
  }

  static handle from_cpp(TypeInterface t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return make_caster<std::unique_ptr<::mlir::Type>>::from_cpp(
        std::make_unique<::mlir::Type>(t), policy, cleanup);
  }
};

template <>
struct type_caster<mlir::Operation> {
protected:
  mlir::Operation *value;

public:
  using Caster = make_caster<mlir::OpState>;
  static constexpr auto Name = Caster::Name;

  template <typename T>
  using Cast = ::mlir::Operation *;

  operator mlir::Operation *() { return value; }
  operator mlir::Operation **() { return &value; }

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    return false;
  }

  template <typename T>
  static constexpr bool can_cast() {
    return false;
  }

  static handle from_cpp(mlir::Operation *v, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    // There's no way I can see to do this safely. MLIR expects
    // you're working in C++ and can use mlir::dyn_cast on a subclass,
    // but we downcast to the subclass via a nanobind type hook.
    auto op = std::make_unique<mlir::OpState>(*(mlir::OpState *)&v);
    return Caster::from_cpp(op, policy, cleanup);
  }
};

template <typename Type>
struct type_caster<mlir::detail::TypedValue<Type>> {
  using Caster = make_caster<mlir::Value>;
  NB_TYPE_CASTER(mlir::detail::TypedValue<Type>,
                 Caster::Name + const_name("[") + make_caster<Type>::Name +
                     const_name("]"))
  Caster caster;

  bool from_python(handle_t<::mlir::Value> src, uint8_t flags,
                   cleanup_list *cleanup) noexcept {
    if (!caster.from_python(src, flags, cleanup))
      return false;
    if (!*caster || mlir::isa<Type>((*caster).getType())) {
      value = mlir::cast<mlir::detail::TypedValue<Type>>(*caster);
      return true;
    }
    return false;
  }

  static handle from_cpp(mlir::TypedValue<Type> v, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return Caster::from_cpp(v, policy, cleanup);
  }
};

template <>
struct type_caster<mlir::BlockArgument>
    : delegate_caster<mlir::BlockArgument, mlir::Value> {};

template <typename Entry>
struct type_caster<llvm::FailureOr<Entry>>
    : delegate_caster<llvm::FailureOr<Entry>, std::optional<Entry>> {};

/// Downcast known Attributes to their bound type object.
/// If we don't know it, return the base Attribute.
template <>
struct type_hook<::mlir::Attribute> {
  static const std::type_info *get(::mlir::Attribute *attr) {
    if (attr && *attr) {
      if (auto info = M::Graph::Python::lookupTypeID(attr->getTypeID()))
        return info;
    }
    return &typeid(::mlir::Attribute);
  }
};

/// Downcast known Types to their bound type object.
/// If we don't know it, return the base Type.
template <>
struct type_hook<::mlir::Type> {
  static const std::type_info *get(::mlir::Type *type) {
    if (type && *type) {
      if (auto info = M::Graph::Python::lookupTypeID(type->getTypeID()))
        return info;
    }
    return &typeid(::mlir::Type);
  }
};

/// Downcast known Ops to their bound type object.
/// If we don't know it, return the base Type.
template <>
struct type_hook<::mlir::OpState> {
  static const std::type_info *get(::mlir::OpState *op) {
    if (op && *op) {
      if (auto info = M::Graph::Python::lookupTypeID(
              op->getOperation()->getRegisteredInfo()->getTypeID()))
        return info;
    }
    return &typeid(::mlir::OpState);
  }
};

/// Casts sequence <-> ArrayRef.
/// This currently copies in each direction.
/// - For Python -> C++ it's unlikely we could improve this, except in the case
/// where the Python value already represents a contiguous C++ array.
/// - For C++ -> Python, we return a special SequenceView type which wraps the
/// ArrayRef as a Sequence type of type-erased references.
template <typename Entry>
struct type_caster<::llvm::ArrayRef<Entry>> {
  using Caster = make_caster<Entry>;
  using VecCaster = make_caster<llvm::SmallVector<Entry>>;
  NB_TYPE_CASTER(::llvm::ArrayRef<Entry>,
                 const_name("Sequence[") + Caster::Name + const_name("]"))

  VecCaster caster;
  bool used = false;

  bool from_python(handle_t<nb::sequence> src, uint8_t flags,
                   cleanup_list *cleanup) noexcept {
    // Nanobind's built in typecasters for collections re-use
    // their internal typecasters, which isn't safe for array refs.
    // We can do something smart like specialize the caster for
    // std::vector<ArrayRef<T>> to hold a vector of typecaster instances.
    ASSERT_STREAM(!used, "ArrayRef typecasters cannot be reused.");
    if (!caster.from_python(src, flags, cleanup))
      return false;
    used = true;
    const Entry *start = caster.value.data();
    value = ::llvm::ArrayRef<Entry>(start, caster.value.size());
    return true;
  }
  static handle from_cpp(::llvm::ArrayRef<Entry> ar, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    // HACK: Somehow ArrayViews are causing reference leaks at shutdown
    // (nanobind warnings) These are likely caused by references used in default
    // arguments. We circumvent the problem by avoiding the ArrayView caster
    // when the list is empty and just creating an empty list.
    if (ar.empty())
      return PyList_New(0);
    return make_caster<M::Graph::Python::SequenceView>::from_cpp(
        M::Graph::Python::SequenceView(ar), policy, cleanup);
  }
};

template <typename Entry>
struct type_caster<::llvm::MutableArrayRef<Entry>>
    : delegate_caster<::llvm::MutableArrayRef<Entry>, ::llvm::ArrayRef<Entry>> {
};

/// Casts object <-> mlir::TypeRange.
/// This makes a copy of the type pointers passing either direction.
template <>
struct type_caster<::mlir::TypeRange> {
  using Caster = make_caster<llvm::ArrayRef<mlir::Type>>;
  NB_TYPE_CASTER(::mlir::TypeRange, Caster::Name)
  Caster caster;

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (caster.from_python(src, flags, cleanup)) {
      value = mlir::TypeRange(caster.value);
      return true;
    }
    return false;
  }
  static handle from_cpp(::mlir::TypeRange t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return Caster::VecCaster::from_cpp(t, policy, cleanup);
  }
};

/// Casts object <-> mlir::ValueRange.
/// This makes a copy of the value pointers passing either direction.
template <>
struct type_caster<::mlir::ValueRange> {
  using TypeCaster = make_caster<mlir::Type>;
  using ValueCaster = make_caster<mlir::Value>;
  using Caster = make_caster<llvm::ArrayRef<mlir::TypedValue<mlir::Type>>>;
  NB_TYPE_CASTER(::mlir::ValueRange, const_name("Sequence[") +
                                         ValueCaster::Name + const_name("[") +
                                         TypeCaster::Name + const_name("]]"))
  Caster caster;

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (caster.from_python(src, flags, cleanup)) {
      value = mlir::ValueRange(caster.value);
      return true;
    }
    return false;
  }
  static handle from_cpp(::mlir::ValueRange t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return Caster::VecCaster::from_cpp(t, policy, cleanup);
  }
};

/// Casts object <-> mlir::ValueRange.
/// This makes a copy of the value pointers passing either direction.
template <>
struct type_caster<::mlir::ResultRange> {
  using Caster = make_caster<::mlir::ValueRange>;
  NB_TYPE_CASTER(::mlir::ResultRange, Caster::Name)

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    return false;
  }

  static handle from_cpp(::mlir::ResultRange t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    std::vector<mlir::Value> vec(t.begin(), t.end());
    return make_caster<std::vector<mlir::Value>>::from_cpp(std::move(vec),
                                                           policy, cleanup);
  }
};

/// Casts object <-> mlir::OperandRange.
/// This makes a copy of the value pointers passing either direction.
template <>
struct type_caster<::mlir::OperandRange> {
  using Caster = make_caster<mlir::ValueRange>;
  NB_TYPE_CASTER(::mlir::ValueRange, Caster::Name)
  Caster caster;

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    return false;
  }
  static handle from_cpp(::mlir::OperandRange t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    std::vector<mlir::Value> vec(t.begin(), t.end());
    return make_caster<std::vector<mlir::Value>>::from_cpp(std::move(vec),
                                                           policy, cleanup);
  }
};

template <typename Return, typename... Args>
struct type_caster<::llvm::function_ref<Return(Args...)>>
    : delegate_caster<::llvm::function_ref<Return(Args...)>,
                      std::function<Return(Args...)>> {};

template <>
struct type_caster<::llvm::LogicalResult> {
  using Caster = make_caster<bool>;
  NB_TYPE_CASTER(::llvm::LogicalResult, Caster::Name)
  static handle from_cpp(::llvm::LogicalResult t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return Caster::from_cpp(t.succeeded(), policy, cleanup);
  }
};

template <>
struct type_caster<::M::ErrorOrSuccess> {
  // void is a capsule, void_type is actual void which translates to None
  using Caster = make_caster<void_type>;
  NB_TYPE_CASTER(::M::ErrorOrSuccess, Caster::Name)
  static handle from_cpp(::M::ErrorOrSuccess &&t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    if (t.isError()) {
      PyErr_SetString(PyExc_RuntimeError, t.getError());
      return nullptr;
    }
    return nb::none().release();
  }
};

template <typename T>
struct type_caster<::M::ErrorOr<T>> {
  using Caster = make_caster<T>;
  NB_TYPE_CASTER(T, Caster::Name)
  static handle from_cpp(::M::ErrorOr<T> &&result, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    if (result.isError()) {
      PyErr_SetString(PyExc_RuntimeError, result.getError());
      return nullptr;
    }

    return make_caster<T>::from_cpp(result.takeValue(), policy, cleanup);
  }
};

template <>
struct type_caster<::llvm::function_ref<mlir::InFlightDiagnostic()>> {
  NB_TYPE_CASTER(::llvm::function_ref<mlir::InFlightDiagnostic()>,
                 const_name("DiagnosticHandler"))

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    return false;
  }
};

} // namespace detail
} // namespace NB_NAMESPACE

#endif // SUPPORT_NANOBIND_TYPECASTERS_H
