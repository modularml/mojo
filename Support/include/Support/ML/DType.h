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
//
// This file declares `DType` which is a standardized type to represent the
// storage format of things like tensors.  This is intended to fit in a single
// byte and be extensible by clients with new enumerators, but isn't suitable
// for things like quantization information.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_DTYPE_H
#define SUPPORT_ML_DTYPE_H

#include "Support/FunctionExtras.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/FloatTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <limits>
#include <tuple>

namespace M {

template <typename ResultType, typename... ParamPtrTypes>
class DTypeSwitch;

/// This class represents common datatypes, e.g. for clients like the storage
/// format of elements in a tensor.  This is encoded in a specific way intended
/// to allow efficient analysis and transformation.
///
/// Note that this class is designed to allow open extension by clients, which
/// can subclass it and adding new enumerators. For example, frameworks want to
/// do weird things within their type system, e.g. have string or ragged tensors
/// (which ...aren't tensors!), have "resources", etc.  Given this reality, we
/// don't want to have conversions back and forth between enum types all the
/// time.  Extensibility allows use of this generic type for those framework
/// defined cases as well.
class DType {
public:
  /// Note that many of these should be interpreted as type classes, i.e. the
  /// enum represents a mask, not a concrete enumerator.  These enum values
  /// are prefixed with `m` to designate this.
  enum Cases : uint8_t {
    invalid = 0,

    //------ Encoding for ordinary primitives --------------------------------//

    // Bit 7 encode primitive category: 0 = Float/Other, 1 = SInt/UInt
    mIsInteger = 1 << 7,

    // Bit 6 encode for Float/Other category encodes "isFloat".
    mIsFloat = 1 << 6,

    // Bit 5 for integer and floating point types indicate if type is complex.
    // This keeps the element types densely packed, allowing table lookups.
    // Note that we support many integer and floating point element types in
    // complex number, but they must be at least a byte in size:
    //   `complex si1` is not supported (but `complex kBool` is).
    mIsComplex = 1 << 5,

    //===--- Signed and Unsigned Integer Types ----------------------------===//
    // This supports any power-of-two integer type up to a larger width than
    // MLIR supports.  The width is encoded in logarithmic form, which enables
    // small lookup tables indexed by the enum value.

    /// Bit 0 encodes "isSigned".
    mIsSigned = 1,

    /// Bits 1 thru 4 indicate the bit width in log base 2.
    kIntWidthShift = 1,
    // i1's densely packed in memory.
    si1 = (0 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui1 = (0 << kIntWidthShift) | mIsInteger,
    si2 = (1 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui2 = (1 << kIntWidthShift) | mIsInteger,
    si4 = (2 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui4 = (2 << kIntWidthShift) | mIsInteger,
    si8 = (3 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui8 = (3 << kIntWidthShift) | mIsInteger,
    si16 = (4 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui16 = (4 << kIntWidthShift) | mIsInteger,
    si32 = (5 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui32 = (5 << kIntWidthShift) | mIsInteger,
    si64 = (6 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui64 = (6 << kIntWidthShift) | mIsInteger,
    si128 = (7 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui128 = (7 << kIntWidthShift) | mIsInteger,
    si256 = (8 << kIntWidthShift) | mIsInteger | mIsSigned,
    ui256 = (8 << kIntWidthShift) | mIsInteger,

    //===--- Floating point types -----------------------------------------===//

    /// Bits 0 through 3 indicate the kind of FP value.
    f4e2m1fn = 0 | mIsFloat,
    /// The OCP MX FP6 encodings. Both are finite-only (no inf/NaN), hence the
    /// `fn` suffix, matching LLVM's `Float6E2M3FN` / `Float6E3M2FN`.
    f6e2m3fn = 1 | mIsFloat,
    f6e3m2fn = 2 | mIsFloat,
    /// Some slots are left blank here to enable us to support more lower
    /// precision types in the future.
    f8e8m0fnu = 9 | mIsFloat,
    f8e3m4 = 10 | mIsFloat,
    f8e4m3fn = 11 | mIsFloat,
    f8e4m3fnuz = 12 | mIsFloat,
    f8e5m2 = 13 | mIsFloat,
    f8e5m2fnuz = 14 | mIsFloat,
    f16 = 15 | mIsFloat,
    bf16 = 16 | mIsFloat,
    f32 = 17 | mIsFloat,
    f64 = 18 | mIsFloat,

    // Helpers for common complex types.  We don't need to enumerate all of
    // them, though, use DType::getComplex(eltType) for generality.
    complex_f32 = f32 | mIsComplex,
    complex_f64 = f64 | mIsComplex,

    //===--- Encodings for other types ------------------------------------===//

    // kBool != ui1.  Like it, this only contains 1-bit of data, but it occupies
    // 1-byte of storage.  The rest of the byte is guaranteed to be zeros.
    kBool = 1,

    kFirstExtendedOption = 2,

    // Derived enums may add their own types into the "Other" category.  This
    // should always be done by starting with `kFirstExtendedOption`, e.g.
    // like this:
    //    kYourThing  = kFirstExtendedOption,
    //    kYourOtherThing,
  };

  // Related constants.
  enum {
    // Complex i256 is the largest size thing this type enum can represent.
    // Knowing this allows kernels to use fixed-size on-stack buffers.
    // When increasing this value, be sure to check users of this constant to
    // ensure they are still correct & sufficiently efficient.
    kMaxElementSizeInBits = 256 * 2,
    kMaxElementSizeInBytes = kMaxElementSizeInBits / 8
  };

  /*implicit*/ constexpr DType() : value(invalid) {}
  /*implicit*/ constexpr DType(Cases v) : value(v) {}
  explicit constexpr DType(uint8_t v) : value((Cases)v) {}

  /// This turns the printed form of a dtype back into a DType or
  /// returns None if it is an unrecognized name.
  static FailureOr<DType> getFromString(StringRef str);

  /// This returns the underlying integer value for the enum.
  constexpr uint8_t getValue() const { return value; }

  // Categorization.
  constexpr bool isValid() const { return !isInvalid(); }
  constexpr bool isInvalid() const { return value == invalid; }
  constexpr bool isBool() const { return value == DType::kBool; }
  constexpr bool isBoolLike() const {
    return value == DType::kBool || value == DType::ui8;
  }
  constexpr bool isInt() const { return (value & mIsInteger) != 0; }
  constexpr bool isSInt() const { return isInt() & ((value & mIsSigned) != 0); }
  constexpr bool isUInt() const { return isInt() & ((value & mIsSigned) == 0); }
  constexpr bool isFloat() const {
    return !isInt() && ((value & mIsFloat) != 0);
  }
  bool isFloat8() const;

  constexpr bool isArithmetic() const { return isInt() || isFloat(); }
  constexpr bool isOther() const { return !isArithmetic(); }

  // Complex number handling.
  constexpr bool isComplex() const { return value & mIsComplex; }

  /// If the current type is a complex type, remove the complex marker to get
  /// the underlying element type, otherwise return it unmodified.
  constexpr DType stripComplex() const { return DType(value & ~mIsComplex); }

  /// Given a valid element type for a complex number, return the complex type.
  /// We do not support sub-byte element types in order to simplify clients.
  static DType getComplex(DType eltType) {
    assert(eltType.getWidthInBits() >= 8 && "invalid element type for complex");
    assert(!eltType.isComplex() &&
           "cannot construct a complex type with complex type as element");
    return DType(eltType.getValue() | mIsComplex);
  }

  /// Return a complex type if it is valid, otherwise fail.
  static FailureOr<DType> getComplexChecked(DType eltType);

  // Integer handling.

  /// This returns a DType for an integer with the specified width and
  /// signedness, or `invalid` if it cannot be represented.
  static FailureOr<DType> getInt(unsigned widthInBits, bool isSigned) {
    if (!llvm::isPowerOf2_32(widthInBits) ||
        // Disallow large numbers by policy, because we don't want clients to
        // have to worry about memory allocation for arbitrarily large values.
        widthInBits > kMaxElementSizeInBits / 2)
      return failure();
    unsigned widthEncoding = llvm::countr_zero(widthInBits);
    return DType((widthEncoding << kIntWidthShift) | mIsInteger |
                 (isSigned ? mIsSigned : 0));
  }

  /// Return the width in bits of this type, assuming it is an integer type.
  size_t getIntegerWidthInBits() const {
    return size_t(1) << getIntegerWidthInLogBits();
  }

  /// Return the width in bits of this type, assuming it is an integer type.
  /// This returns the amount in power of two, e.g. it returns:
  ///    0 == 1, 1 == 2, 2 == 4, 3 == 8, etc.
  size_t getIntegerWidthInLogBits() const {
    assert(isInt() && "Can only get the width of an integer type");
    return (value & ~(mIsInteger | mIsComplex)) >> kIntWidthShift;
  }

  // Type generic methods.

  /// Return the width of this element in bits. This returns -1 for unknown
  /// width values.
  ssize_t getWidthInBits() const;

  // Returns the alignment of this element in bytes.
  size_t getAlignment() const;

  /// This method returns the LLVM floating point semantics for the given DType,
  /// or nullptr if the DType is not a floating point type LLVM knows about
  /// (e.g. TF32).
  const llvm::fltSemantics *getFloatSemantics() const;

  /// Return the in-memory size for an array of the specified type with the
  /// specified number of elements, or -1 for non-numeric types or too large
  /// values.  This supports densely packed sub-byte types like i1, i2, i4.
  ssize_t getSizeInBytes(size_t numElements = 1) const;

  /// Return the in-memory size for an array of the specified type with the
  /// specified number of elements, or Error for non-numeric types or too large
  /// values.  This supports densely packed sub-byte types like i1, i2, i4.
  FailureOr<size_t> getSizeInBytesChecked(size_t numElements = 1) const;

  constexpr bool operator==(uint8_t v) const { return value == v; }
  constexpr bool operator!=(uint8_t v) const { return value != v; }
  constexpr bool operator==(DType v) const { return value == v.value; }
  constexpr bool operator!=(DType v) const { return value != v.value; }

  /// Perform a eltType dispatch to delegate to a lambda or other callable, see
  /// the definition of `DTypeSwitch` below.
  template <typename ResultType, typename... ParamPtrTypes>
  DTypeSwitch<ResultType, ParamPtrTypes...>
  dispatch(ParamPtrTypes... paramPtrs) const;

  /// Return a string form of this eltType suitable for printing and error
  /// messages.
  std::string getAsString() const;

  void print(raw_ostream &os) const;
  void dump() const;

  // Get the maximum and minimum representable value for the given dtype.
  ErrorOr<std::pair<int32_t, int32_t>> getMaxAndMinValue() const;

private:
  Cases value;
};

static_assert(sizeof(DType) == 1, "DType should not grow");

inline raw_ostream &operator<<(raw_ostream &os, DType value) {
  value.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// DTypeForCXXType CXXTypeForDType
//===----------------------------------------------------------------------===//

/// Provide a mapping from C++ types to the corresponding EltType kinds.
template <typename CXXType>
struct DTypeForCXXType {
  // Default mapping allows a static_assert instead of template substitution
  // failure to catch invalid cases.
  static constexpr DType kind = DType::invalid;
};

/// Provide a mapping from DType enums to C++ types.
template <unsigned DType>
struct CXXTypeForDType {
  using CXXType = void;
};

#define DECLARE_TYPE_MAPPING(ELTTYPE_KIND, CXXTYPE)                            \
  template <>                                                                  \
  struct DTypeForCXXType<CXXTYPE> {                                            \
    static constexpr DType kind = DType::ELTTYPE_KIND;                         \
  };                                                                           \
  template <>                                                                  \
  struct CXXTypeForDType<DType::ELTTYPE_KIND> {                                \
    using CXXType = CXXTYPE;                                                   \
  }

DECLARE_TYPE_MAPPING(kBool, bool);
DECLARE_TYPE_MAPPING(si8, int8_t);
DECLARE_TYPE_MAPPING(ui8, uint8_t);
DECLARE_TYPE_MAPPING(si16, int16_t);
DECLARE_TYPE_MAPPING(ui16, uint16_t);
DECLARE_TYPE_MAPPING(si32, int32_t);
DECLARE_TYPE_MAPPING(ui32, uint32_t);
DECLARE_TYPE_MAPPING(si64, int64_t);
DECLARE_TYPE_MAPPING(ui64, uint64_t);

#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, M_TYPE, MLIR_TYPE, CXX_TYPE, ...) \
  DECLARE_TYPE_MAPPING(SHORT_NAME, CXX_TYPE);

#include "Support/ML/FloatTypes.def"

#undef DECLARE_FLOAT

/// TODO: Add long double when sizeof(long double) != sizeof(double).
#undef DECLARE_TYPE_MAPPING

//===----------------------------------------------------------------------===//
// DTypeSwitch
//===----------------------------------------------------------------------===//

/// This class is used to implement switch-like dispatch for DType
/// values. In addition to allowing checks of enumerator values, it provides
/// convenient helpers for things like "C++ floating point types" and "integer
/// types ignoring sign" etc.  This should be used like:
///
///   someDType.dispatch<>(paramPtrs...)  // pass in void* / const void*
///      .when<DType::f32>([](void *... buf) { ... invoked when f32 ...
///      }) .when([](bool *... bufPtr) { ... invoked when kBool ... })
///      .whenCXXFP([](auto *... bufPtr) { ... invoked with correct pointer type
///      ... .otherwise([]() { ... invoked otherwise ... });
///
template <typename ResultType, typename... ParamPtrTypes>
class DTypeSwitch {
public:
  DTypeSwitch(DType value, ParamPtrTypes... paramPtrs)
      : paramPtrs(std::forward_as_tuple(paramPtrs...)), value(value) {
    static_assert(
        ((std::is_same_v<ParamPtrTypes, void *> ||
          std::is_same_v<ParamPtrTypes, const void *>) &&
         ...),
        "Input pointers to type dispatch should be void*/const void*");
  }
  ~DTypeSwitch() = default;

  /// Add a case on the given type.
  template <uint8_t CaseValue, typename CallableT>
  DTypeSwitch &when(CallableT &&caseFn) {
    using CXXType = typename CXXTypeForDType<CaseValue>::CXXType;
    // Check to see if any of the types apply to 'value'.
    if (!result && this->value.getValue() == CaseValue) {
      CallableReturnType callableRes = std::apply(
          [&](ParamPtrTypes... args) {
            return invokeWithDefaultResultType<EmptyReturnType>(
                std::forward<CallableT>(caseFn),
                constPreservingCast<CXXType *>(
                    std::forward<ParamPtrTypes>(args))...);
          },
          paramPtrs);
      result = std::move(callableRes);
    }
    return *this;
  }

  /// Invoke a case on the derived class, inferring the type of the Case from
  /// the first input of the given callable.
  /// Note: This inference rules for this overload are very simple: strip
  ///       pointers and references.
  template <typename CallableT>
  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG DTypeSwitch &
  when(CallableT &&caseFn) {
    using Traits = llvm::function_traits<std::decay_t<CallableT>>;
    using ElementType = std::remove_cv_t<std::remove_pointer_t<
        std::remove_reference_t<typename Traits::template arg_t<0>>>>;
    constexpr auto kind = DTypeForCXXType<ElementType>::kind.getValue();
    static_assert(kind != DType::invalid, "unknown C++ pointer type in lambda");
    return this->when<kind>(std::forward<CallableT>(caseFn));
  }

  /// Invoke a case on the derived class with multiple case types.
  template <uint8_t CaseV1, uint8_t CaseV2, uint8_t... CaseVs,
            typename CallableT>
  // This is marked always_inline and nodebug so it doesn't show up in stack
  // traces at -O0 (or other optimization levels).  Large DTypeSwitch's
  // are common, are equivalent to a switch, and don't add any value to stack
  // traces.
  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG DTypeSwitch &
  when(CallableT &&caseFn) {
    return when<CaseV1>(caseFn).template when<CaseV2, CaseVs...>(caseFn);
  }

  /// As a default, invoke the given callable within the root value.
  template <typename CallableT>
  DTypeSwitch &otherwise(CallableT &&defaultFn) {
    if (!result) {
      CallableReturnType callableRes =
          invokeWithDefaultResultType<EmptyReturnType>(
              std::forward<CallableT>(defaultFn));
      result = std::move(callableRes);
    }
    return *this;
  }

  /// Invoke the specified lambda with all the standard C++ integer types from 8
  /// to 64 bits in both signed and unsigned forms.  This passes the pointer in
  /// with the correct C++ type, so it is usually best to use a generic lambda:
  ///
  ///  eltType.dispatch<ResultType>(ptr...)
  ///     .whenCXXInt([&](auto *... ptr) { use ptr generically })
  ///
  template <typename CallableT>
  DTypeSwitch &whenCXXInt(CallableT &&elementFn) {
    return when<DType::si8, DType::ui8, DType::si16, DType::ui16, DType::si32,
                DType::ui32, DType::si64, DType::ui64>(
        std::forward<CallableT>(elementFn));
  }

  /// Invoke the specified lambda with `float` and `double`.  This passes the
  /// pointer in with the correct C++ type, so it is usually best to use a
  /// generic lambda:
  ///
  ///  eltType.dispatch<ResultType>(ptr...)
  ///     .whenCXXFP([&](auto *... ptr) { use ptr generically })
  ///
  /// TODO: Add long double when sizeof(long double) != sizeof(double).
  template <typename CallableT>
  DTypeSwitch &whenCXXFP(CallableT &&elementFn) {
    return when<DType::f32, DType::f64>(std::forward<CallableT>(elementFn));
  }

  /// Invoke the specified lambda with integer and float element types.
  /// This passes the pointer in with the correct C++ type, so it is usually
  /// best to use a generic lambda:
  ///
  ///  eltType.dispatch<ResultType>(ptr...)
  ///     .whenCXXArithmeticType([&](auto *... ptr) { use ptr generically })
  ///
  /// TODO: Add long double when sizeof(long double) != sizeof(double).
  template <typename CallableT>
  DTypeSwitch &whenCXXArithmeticType(CallableT &&elementFn) {
    return this->whenCXXInt(std::forward<CallableT>(elementFn))
        .whenCXXFP(std::forward<CallableT>(elementFn));
  }

  /// Invoke the specified lambda with integer, float and bool element types.
  /// This passes the pointer in with the correct C++ type, so it is usually
  /// best to use a generic lambda:
  ///
  ///  eltType.dispatch<ResultType>(ptr...)
  ///     .whenCXXType([&](auto *... ptr) { use ptr generically })
  ///
  /// TODO: Add long double when sizeof(long double) != sizeof(double).
  template <typename CallableT>
  DTypeSwitch &whenCXXType(CallableT &&elementFn) {
    return this->whenCXXArithmeticType(std::forward<CallableT>(elementFn))
        .template when<DType::kBool>(std::forward<CallableT>(elementFn));
  }

  [[nodiscard]] operator ResultType() {
    assert(result && "Fell off the end of a DTypeSwitch");
    return std::move(*result);
  }

private:
  template <typename To, typename From>
  decltype(auto) constPreservingCast(From *arg) {
    if constexpr (std::is_const_v<From>)
      return static_cast<const To>(const_cast<std::decay_t<From> *>(arg));
    else
      return static_cast<To>(arg);
  }

  // EmptyReturnType is used when the result type is void.
  struct EmptyReturnType {
    static EmptyReturnType get() { return EmptyReturnType{}; }
  };
  using CallableReturnType = std::conditional_t<std::is_void_v<ResultType>,
                                                EmptyReturnType, ResultType>;

  /// DTypeSwitch is not a value.
  DTypeSwitch(const DTypeSwitch &) = delete;
  DTypeSwitch(DTypeSwitch &&other) = delete;
  void operator=(const DTypeSwitch &) = delete;
  void operator=(DTypeSwitch &&other) = delete;

  /// The parameter pointers that we're casting.
  std::tuple<ParamPtrTypes...> paramPtrs;

  const DType value; /// The value we are switching on.

  /// The result of this switch statement, once known, None before that.
  std::optional<CallableReturnType> result;
};

/// Perform a eltType dispatch to delegate to a lambda or other callable, see
/// the definition of `DTypeSwitch` below.
template <typename ResultType, typename... ParamPtrTypes>
inline DTypeSwitch<ResultType, ParamPtrTypes...>
DType::dispatch(ParamPtrTypes... paramPtrs) const {
  return DTypeSwitch<ResultType, ParamPtrTypes...>(
      *this, std::forward<ParamPtrTypes>(paramPtrs)...);
}

/// Hash a DType.
inline llvm::hash_code hash_value(DType dtype) {
  return llvm::hash_value(dtype.getValue());
}

} // namespace M

// Provide the DenseMapInfo for DType so we can use it in llvm::DenseMaps.
namespace llvm {
template <>
struct DenseMapInfo<M::DType> {
  static M::DType getEmptyKey() {
    return M::DType(std::numeric_limits<uint8_t>::max());
  }
  static M::DType getTombstoneKey() {
    return M::DType(std::numeric_limits<uint8_t>::max() - 1);
  }
  static unsigned getHashValue(const M::DType &dtype) {
    return M::hash_value(dtype);
  }

  static bool isEqual(const M::DType &LHS, const M::DType &RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif // SUPPORT_ML_DTYPE_H
