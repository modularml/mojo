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
// This file provides information for working with 'special functions' in Lit
// like the __new__ function.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_SPECIAL_FUNCTIONS_H
#define KGEN_KGENDIALECT_SPECIAL_FUNCTIONS_H

#include "llvm/ADT/StringRef.h"

namespace M::KGEN::LIT {

enum class SpecialFunctionKind : uint8_t {
  // This is not a special function.  This enumerator should always have value
  // zero so it can be used as a false condition in an if.
  kNormal = 0,

#define SF(ENUM, NAME, MINOPERANDS, MAXOPERANDS, EXPRNODE, FLAGS) ENUM,
#include "Mojo/LITDialect/SpecialFunctions.def"
};

class SpecialFunctionInfo {
public:
  const char *name = nullptr;
  SpecialFunctionKind kind = SpecialFunctionKind::kNormal;

  /// The minimum number of arguments that this special function requires.
  unsigned minNumArguments = 0;

  /// The maximum number of arguments that this special function requires, or -1
  /// if variadic.
  int maxNumArguments = -1;

  unsigned flags = 0;

  /// This is a bitmask of flags that describes requirements of the special
  /// function.
  enum {
    /// This is an implicitly static method like __new__ even if not declared
    /// as such.
    kImplicitlyStaticMethod = 1 << 0,

    /// This must be an instance method of a type.
    kInstMethod = 1 << 1,

    /// This is true when this represents a "reversed" operator like __radd__.
    kReversedOperator = 1 << 2,

    /// This is true when the operation is supposed to return None.
    kNoneResult = 1 << 3,

    /// This method must return Self.
    kSelfResult = 1 << 4,

    /// This method is a struct initializer, it is a static method that returns
    /// 'Self', which is typically spelled with an "out self" first argument.
    kInitializer = (1 << 5) | kImplicitlyStaticMethod | kSelfResult,

    /// This method cannot be declared to raise an error.
    kCannotRaise = 1 << 6,
  };

  /// Return true if this is any kind of instance method.
  bool isInstMethod() const { return (flags & kInstMethod) != 0; }

  /// Return true if this special function is implicitly static, like __init__.
  bool isImplicitlyStatic() const {
    return (flags & kImplicitlyStaticMethod) != 0;
  }

  /// Return true if this is a reversed operator.
  bool isReversed() const { return (flags & kReversedOperator) != 0; }

  /// Return true if this special function must return None.
  bool hasNoneResult() const { return (flags & kNoneResult) != 0; }

  /// Return true if this special function is an initializer.
  bool isInitializer() const { return (flags & kInitializer) == kInitializer; }

  /// Return true if this special function returns Self
  bool hasSelfResult() const { return (flags & kSelfResult) != 0; }

  /// Return a record that describes special functions like __init__.  The
  /// kind field identifies it.
  static const SpecialFunctionInfo &get(SpecialFunctionKind kind);

  /// Given a function name like "__init__" return the special function kind
  /// that corresponds to it.
  static SpecialFunctionKind lookupKind(llvm::StringRef name);

  /// If `name` is a deprecated spelling of a special function that has a
  /// canonical replacement (currently just the destructor's '__del__', whose
  /// canonical spelling is '__deinit__'), returns the canonical spelling.
  /// Otherwise returns `name` unchanged.
  static llvm::StringRef getCanonicalSpelling(llvm::StringRef name);
};

} // namespace M::KGEN::LIT

#endif // KGEN_KGENDIALECT_SPECIAL_FUNCTIONS_H
