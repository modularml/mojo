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

#ifndef KGEN_KGENDIALECT_KGENENUMS_H
#define KGEN_KGENDIALECT_KGENENUMS_H

#include "mlir/IR/BuiltinAttributes.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Enum Declarations
//===----------------------------------------------------------------------===//

// Pull in all enum type definitions and utility function declarations.
#include "Mojo/KGENDialect/KGENEnums.h.inc"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// ArgConvention
//===----------------------------------------------------------------------===//

/// Determine whether an argument with the given input convention expects to
/// have a pointer or reference type.
static inline bool hasAddress(ArgConvention conv) {
  return conv != ArgConvention::OwnedReg && conv != ArgConvention::ImmReg;
}

/// Determine whether an argument with the given input convention expects to
/// have an implicit origin.
static inline bool hasImplicitOrigin(ArgConvention conv) {
  switch (conv) {
  case ArgConvention::Ref:
  case ArgConvention::MutRef:
  case ArgConvention::OwnedReg:
  case ArgConvention::ImmReg:
    return false;
  case ArgConvention::OwnedMem:
  case ArgConvention::DeinitMem:
  case ArgConvention::ImmMem:
  case ArgConvention::Mut:
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError:
    return true;
  }
  llvm_unreachable("invalid argument convention");
}

/// Return true if this is an memory location for a normal or error result.
static inline bool isResultSlot(ArgConvention conv) {
  return conv == ArgConvention::ByRefResult ||
         conv == ArgConvention::ByRefError;
}

//===----------------------------------------------------------------------===//
// FnEffects
//===----------------------------------------------------------------------===//

/// This class represents the effects of a callable. A callable can throw an
/// error, be an async function, etc. The effect of a callable is load-bearing
/// on its type.
class FnEffects {
  using Impl = impl::FnEffects;

public:
  FnEffects(Impl impl = Impl::None) : impl(impl) {}

  bool isThrows() const { return get(Impl::Throws); }
  bool isAsync() const { return get(Impl::Async); }
  bool isCapturing() const { return get(Impl::Capturing); }
  bool isCABI() const { return get(Impl::CABI); }
  bool isRefResult() const { return get(Impl::RefResult); }

  FnEffects setThrows(bool throws = true) { return set(Impl::Throws, throws); }
  FnEffects setAsync(bool async = true) { return set(Impl::Async, async); }
  FnEffects setCapturing(bool capturing = true) {
    return set(Impl::Capturing, capturing);
  }
  FnEffects setCABI(bool value = true) { return set(Impl::CABI, value); }
  FnEffects setRefResult(bool value = true) {
    return set(Impl::RefResult, value);
  }

  bool operator==(FnEffects rhs) const { return getImpl() == rhs.getImpl(); }
  bool operator!=(FnEffects rhs) const { return getImpl() != rhs.getImpl(); }

  Impl getImpl() const { return impl; }

private:
  FnEffects set(Impl bit, bool value) {
    impl = impl::bitEnumSet(impl, bit, value);
    return *this;
  }
  bool get(Impl bit) const { return impl::bitEnumContainsAny(impl, bit); }

  Impl impl;
};

template <typename StreamT>
inline StreamT &operator<<(StreamT &os, FnEffects effects) {
  os << impl::stringifyFnEffects(effects.getImpl());
  return os;
}

/// Found by ADL so that `FnEffects` can be used as an attribute or type
/// parameter, whose uniquer needs to hash it.
inline llvm::hash_code hash_value(FnEffects effects) {
  return llvm::hash_value(static_cast<uint16_t>(effects.getImpl()));
}

//===----------------------------------------------------------------------===//
// ArgConvention
//===----------------------------------------------------------------------===//

/// Return a string like "imm" or "mut".
const char *getUserSyntax(ArgConvention convention);

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENENUMS_H
