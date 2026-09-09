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
// Provides TypeID, a globally unique, 2-byte identifier for any C++ type which
// can be by computed as 'TypeID::get<TheType>()'. TypeIDs can be retrieved
// and used across dynamic library / executable boundaries via inclusion of
// this header file.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TYPEID_H
#define SUPPORT_TYPEID_H

#include "Support/ADT/ConcurrentAppendingVector.h"
#include "Support/Globals/Globals.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Compiler.h"

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string_view>
#include <utility>

namespace M {

/// Type of destructor functions of arbitrary type.
using ValueDestructorFn = void (*)(void *);

namespace Detail {

//===----------------------------------------------------------------------===//
// Compile time type to string conversion
//===----------------------------------------------------------------------===//

/// Unfortunately there is no way to build a constexpr string specifically in
/// C++17, and `basic_fixed_string` never made it into C++20.  Just do it the
/// old-fashioned way with a char array rather than implementing a full-fledged
/// `basic_fixed_string` type.
template <std::size_t... Indices>
constexpr auto stringToArray(std::string_view str,
                             std::index_sequence<Indices...>) {
  return std::array{str[Indices]...};
}

template <class T>
constexpr auto typeNameArray() {
#if defined(__clang__)
  constexpr std::string_view prefix = "[T = ";
  constexpr std::string_view suffix = "]";
#elif defined(__GNUC__)
  constexpr std::string_view prefix = "with T = ";
  constexpr std::string_view suffix = "]";
#elif defined(_MSC_VER)
  constexpr std::string_view prefix = "typeNameArray<";
  constexpr std::string_view suffix = ">(void)";
#else
#error                                                                         \
    "Modular Runtime built with a toolchain not supporting type introspection."
#endif

  constexpr std::string_view function = LLVM_PRETTY_FUNCTION;

  // The algorithm is straightforward:
  // Find where the prefix starts and record the index at the end of it
  // Find where the suffix ends and record the index at the beginning of it
  // Create a substring between the two indices

  constexpr auto start = function.find(prefix) + prefix.size();
  constexpr auto end = function.rfind(suffix);

  static_assert(start < end,
                "Invalid assumptions about parsing type_name for a type.");

  constexpr auto name = function.substr(start, end - start);
  return stringToArray(name, std::make_index_sequence<name.size()>{});
}

/// In C++17, we can't define an object with static storage duration inside of a
/// `constexpr` function.  However, we can define a `constexpr` object with
/// static storage duration as a member and access through that from within a
/// `constexpr` function.
template <class T>
struct TypeNameHolder {
  static inline constexpr auto value = typeNameArray<T>();
};

/// Returns the compile-time name of the type T for compilers that support the
/// 'pretty' name representation for type T.  If the compiler does not support
/// it, an error is given at build time.
///
/// Currently this only supports getting the demangled type name for a type, and
/// so you cannot specify a non-type (e.g. an NTTP, enum class, etc.) right now.
///
/// TODO: Should we encounter issues with non-uniqueness of type names (eg
/// because of types in anonymous namespaces) then, we can introduce a
/// customization point (such as a class template with static function) that we
/// can specialize for certain types.  We don't currently have this use case
/// though.
///
/// TODO: Should we need to build with toolchains which do not support
/// the PRETTY_FUNCTION machinery then we'll need to pre-register every type
/// manually. See third-party/llvm-project/mlir/include/mlir/Support/TypeID.h
/// for a macro style to mimic.
template <class T>
constexpr std::string_view typeNameFor() {
  constexpr auto &value = TypeNameHolder<T>::value;
  return std::string_view{value.data(), value.size()};
}

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

/// The ValueDestructorFn for values of type T.
template <typename T>
static void valueDestructorFn(void *pointer) {
  std::destroy_at(static_cast<T *>(pointer));
}

/// The underlying unique 2-byte identifier for a type.
using RawTypeID = uint16_t;

/// The distinguished invalid raw type id.
constexpr RawTypeID kInvalidRawTypeID = RawTypeID(~0);

//===----------------------------------------------------------------------===//
// Caching
//===----------------------------------------------------------------------===//

/// A 'cache' for the raw type id for T. Ok if ends up with
/// compiler-instantiated definitions in multiple dynamic libraries /
/// executable due to template instantiation since the true synchronization is
/// done by the singleton type info table.
///
/// For internal use only.
template <typename T>
struct TypeIDCache {
  static std::atomic<RawTypeID> cachedID;
};

template <typename T>
std::atomic<RawTypeID> TypeIDCache<T>::cachedID = kInvalidRawTypeID;

//===----------------------------------------------------------------------===//
// TypeInfoTable
//===----------------------------------------------------------------------===//

/// Pair the destructor and type name used at registration time. The latter
/// is very handy for debugging, eg see AsyncValue::printDebug.
struct TypeInfo {
  std::string_view typeName;
  ValueDestructorFn destructorFn;

  TypeInfo(std::string_view typeName, ValueDestructorFn destructorFn)
      : typeName(typeName), destructorFn(destructorFn) {}
};

/// The globally unique type info table. The string -> id mapping uses
/// heavyweight mutex synchronization, but see TypeIDCache for how that cost is
/// amortized. The id -> property mapping only needs atomic synchronization and
/// is very cheap.
class TypeInfoTable {
public:
  TypeInfoTable(size_t initialCapacity) : entries(initialCapacity) {}

  Detail::RawTypeID getSlow(std::string_view typeName,
                            ValueDestructorFn destructor);
  std::string_view getTypeName(Detail::RawTypeID id) const {
    return id == Detail::kInvalidRawTypeID ? std::string_view{"unk"}
                                           : entries[id].typeName;
  }
  ValueDestructorFn getValueDestructor(Detail::RawTypeID id) const {
    return id == Detail::kInvalidRawTypeID ? nullptr : entries[id].destructorFn;
  }

  static TypeInfoTable &getSingleton() {
    return Globals::getTypeInfoTableSingleton(
        [] { return new TypeInfoTable(64); });
  }

private:
  mutable std::mutex mu; // protects ids
  llvm::StringMap<Detail::RawTypeID> ids;
  ConcurrentAppendingVector<TypeInfo> entries;
};

} // namespace Detail

//===----------------------------------------------------------------------===//
// TypeID
//===----------------------------------------------------------------------===//

/// A globally unique, 2-byte identifier for a type.
///
/// TypeIds for any C++ type can be calculated by:
///    TypeID::get<TheType>()
/// The result will be globally unique, even when the get expression is used
/// across dynamic library / executable boundaries via this header file.
/// There is no need to pre-register types.
///
/// To ensure uniqueness a global type info table is created, and heavyweight
/// synchronization is needed when a type is first encountered. However,
/// thereafter the get expression makes use of a templated static cache
/// and is fast.
///
/// The 'root of uniqueness' for types is their 'pretty' names. It is not
/// recommended to use types from anonymous namespaces to avoid accidental
/// name collision.
/// TODO: If this becomes an issue, we'll need to introduce a customization
/// point for types to specify their 'pretty' name that is unique.
class TypeID {
public:
  /// Constructs the 'invalid' type id.
  TypeID() = default;

  /// Returns the unique type id for T. Thread safe. Can be called from
  /// multiple dynamic libraries / executables using this header file.
  /// Fast after the first call per dynamic library / executable per T.
  template <typename T>
  static TypeID get() {
    /// Fast path: we've already cached an id in the caller's dynamic library
    /// / executable. Ok to use relaxed consistency since we only care if the
    /// value has already been set.
    Detail::RawTypeID id =
        Detail::TypeIDCache<T>::cachedID.load(std::memory_order_relaxed);
    if (id != Detail::kInvalidRawTypeID)
      return TypeID(id);

    /// Slow path: We'll use the string name of T to ensure key uniqueness,
    /// and heavyweight synchronization in fn over the global type info table
    /// to ensure id uniqueness.
    id = getSlow(Detail::typeNameFor<T>(), Detail::valueDestructorFn<T>);
    /// Cache the id. We don't care if we are not the first to make the store
    /// since the underlying id will be consistent over all threads.
    Detail::TypeIDCache<T>::cachedID.store(id, std::memory_order_relaxed);
    return TypeID(id);
  }

  /// Assert equality of this type id (assumed to represent the actual runtime
  /// type id of an object of interest) and expected type id (assumed to
  /// represent the expected static type assumed for the object of interest).
  ///
  /// In debug builds, a failure produces a human readable description of the
  /// actual and expected type names, augmented with the context string.
  void assertEqual(TypeID expected, StringRef context) const {
#ifdef MODULAR_DEBUG
    printErrorIfNotEqual(expected, context);
#endif
    assert(id == expected.id &&
           "mismatch between actual and expected type ids");
  }

  /// Returns a 'signature' for the type id subsystem which is expected to
  /// be unique for the running process. This can be used to catch, at runtime,
  /// accidental multiple definitions for Modular runtime statics across
  /// dynamic libraries / executables.
  ///
  /// (This is just the address of the underlying table info singleton, but
  /// please don't depend on that.)
  static intptr_t getSignature() {
    return reinterpret_cast<intptr_t>(&Detail::TypeInfoTable::getSingleton());
  }

  inline bool operator==(const TypeID &other) const { return id == other.id; }
  inline bool operator!=(const TypeID &other) const {
    return !(*this == other);
  }

  /// Returns the name for this type id, or "unk" if invalid.
  std::string_view getTypeName() const {
    return Detail::TypeInfoTable::getSingleton().getTypeName(id);
  }

  /// Returns the destructor function for this type id, or null if invalid.
  ///
  /// CAUTION: The ValueDestructorFn will destroy the object in-place, but
  ///          will not attempt to delete the object's memory. If the object
  ///          has been new allocated then the underlying memory must be
  ///          deleted by the caller.
  ValueDestructorFn getValueDestructor() const {
    return Detail::TypeInfoTable::getSingleton().getValueDestructor(id);
  }

  /// Returns the underlying index representing the type id. This is in the
  /// range [0..2^16-2].
  uint16_t getDenseIndex() { return id; }

private:
  explicit TypeID(Detail::RawTypeID id) : id(id) {}

  /// Slow path for get. Will force global synchronization on global type
  /// info table.
  LLVM_ATTRIBUTE_NOINLINE
  static Detail::RawTypeID getSlow(std::string_view typeName,
                                   ValueDestructorFn destructorFn);

#ifdef MODULAR_DEBUG
  /// Slow path for assertEqual.
  LLVM_ATTRIBUTE_NOINLINE void printErrorIfNotEqual(TypeID expected,
                                                    StringRef context) const;
#endif

  Detail::RawTypeID id = Detail::kInvalidRawTypeID;
};

} // namespace M

#endif // SUPPORT_TYPEID_H
