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

#ifndef SUPPORT_ML_COMPILEDFRAMEWORKLABEL_H
#define SUPPORT_ML_COMPILEDFRAMEWORKLABEL_H

#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"

#include <cassert>

namespace M {

/// Indicates the 'framework' needed to execute a compiled model.
///
/// This is an enum coupled with helpers needed at compile-time, at
/// runtime, and within tools to indicate which overall framework
/// should be used to execute a compiled model. The framework influences:
///  - how the model's inputs and outputs need to be adjusted to mediate
///    between our internal implementation and the frameworks API.
///  - which additional shared libraries need to be loaded to support
///    execution.
///
/// Note that this cannot be used to represent source-level model types
/// since most frameworks have multiple source representations.
///
/// In MLIR the top-level 'model' op determines the framework, currently by
/// a mixture of op name and, for mgp.model, 'frameworkName' string attribute.
///
/// In MEF the top-level invokable can be decoded to recover the op name
/// and (if needed) frameworkName attribute to match MLIR.
///
/// Some tools need to 'sniff' formats from textual files. Some of the op
/// name and framework name conventions encoded here can help with that.
class CompiledFrameworkLabel {
public:
  enum Cases {
    /// Framework is unknown.
    kUnknown,
    // A compiled Modular model.
    kModularModel,
  } value;

  constexpr CompiledFrameworkLabel() : value(kUnknown) {}

  /* implicit */ constexpr CompiledFrameworkLabel(Cases value) : value(value) {}

  /// Returns the top-level operator name representing this framework, or
  /// null if no such representation is possible.
  const char *getAsOpNameOrNull() const;

  /// Returns the frameworkName representing this framework, or null
  /// if no such representation is possible.
  const char *getAsFrameworkNameOrNull() const;

  /// Returns true if opName is a valid top-level operator name.
  static bool isValidOpName(StringRef opName);

  /// Returns true if opName is a valid framework name.
  static bool isValidFrameworkName(StringRef frameworkName);

  /// Returns the framework describing the framework appropriate for the
  /// given top-level opName with an optional frameworkName if it is known.
  /// Returns unknown if framework cannot be determined.
  static CompiledFrameworkLabel getLabelForOpName(StringRef opName,
                                                  StringRef frameworkName = {});

  /// Return a printable string identifying this framework label.
  const char *getAsString() const;

  /// Returns a unique label string such as `pytorch` identifying the framework.
  /// Differs from `getAsString` in that this is an ID, whereas `getAsString` is
  /// more human readable.
  static const char *asLabelString(Cases label);

  constexpr bool operator==(CompiledFrameworkLabel other) const {
    return value == other.value;
  }
  constexpr bool operator!=(CompiledFrameworkLabel other) const {
    return !(*this == other);
  }
};

} // namespace M

#endif // SUPPORT_ML_COMPILEDFRAMEWORKLABEL_H
