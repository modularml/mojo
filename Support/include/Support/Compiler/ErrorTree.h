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

#ifndef SUPPORT_COMPILER_ERRORTREE_H
#define SUPPORT_COMPILER_ERRORTREE_H

#include "Support/ADT/SmartVariant.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace M {

/// This error class is a complex error consisting of various possible nested
/// causes at certain IR locations. The error consists of a single top-level
/// simple error and potential nested errors.
class ErrorTree {
public:
  /// Construct an error tree with just a main error.
  template <typename U>
  ErrorTree(Location loc, U &&error)
      : loc(loc), error(std::forward<U>(error)) {}

  /// Construct an error tree with a main error and a nested tree of causes.
  ErrorTree(Location loc, Error error, ErrorTree causes);

  /// Construct an error with causes.
  ErrorTree(Location loc, Error error, MutableArrayRef<ErrorTree> causes);

  /// Get the location of the error.
  Location getLoc() const { return loc; }

  /// Get the main error.
  const Error &getError() const { return error; }

  /// Take the main error.
  Error takeError() { return std::move(error); }

  /// Get the causes of the main error.
  std::vector<ErrorTree> &getCauses() { return causes; }

  /// Get the main error message.
  StringRef getMessage() const { return error.get(); }

  /// Add a cause to the error. Return a reference to the current error tree.
  ErrorTree &addCause(ErrorTree cause) & {
    causes.push_back(std::move(cause));
    return *this;
  }
  ErrorTree &&addCause(ErrorTree cause) && {
    return std::move(addCause(std::move(cause)));
  }

  /// Add a cause to the error. Return a reference to the current error tree.
  ErrorTree &addCause(Location loc, Error cause) & {
    causes.emplace_back(loc, std::move(cause));
    return *this;
  }
  ErrorTree &&addCause(Location loc, Error cause) && {
    return std::move(addCause(loc, std::move(cause)));
  }

  /// Add a collection of causes to the error. Return a reference to the current
  /// error tree.
  ErrorTree &addCauses(MutableArrayRef<ErrorTree> causes) {
    for (ErrorTree &cause : causes)
      this->causes.push_back(std::move(cause));
    return *this;
  }

  /// Check if this error is equal to another error in contents.
  bool operator==(const ErrorTree &other) const {
    return loc == other.loc && getMessage() == other.getMessage() &&
           ArrayRef(causes) == ArrayRef(other.causes);
  }

  /// Explicitly copy this error.
  ErrorTree copy() const;

  /// Emit this error to an MLIR diagnostic. The main error is emitted as a
  /// diagnostic error. Any causes are emitted as notes.
  InFlightDiagnostic
  emit(function_ref<InFlightDiagnostic(Location)> emitError,
       StringRef callSiteMsg, bool emitPrelude,
       std::optional<mlir::DiagnosticEngine::HandlerID> diagHandlerID = {}) &&;

private:
  /// Emit nested errors to an MLIR diagnostic as notes.
  static void emit(std::optional<InFlightDiagnostic> &diag,
                   ArrayRef<ErrorTree> errors, StringRef callSiteMsg,
                   bool emitPrelude);

  /// The location of the main error.
  Location loc;

  /// The top-level error.
  Error error;

  /// The nested causes of the main error.
  std::vector<ErrorTree> causes;
};

/// This class represents an error tree or a value.
template <typename T>
class ErrorTreeOr {
public:
  /// Create an error value.
  ErrorTreeOr(ErrorTree &&error) : value(std::move(error)) {}

  /// Create a value.
  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U, T>>>
  ErrorTreeOr(U &&value) : value(T(std::forward<U>(value))) {}

  /// Returns true if there is an error.
  bool isError() const { return isa<ErrorTree>(value); }

  /// Get a reference to the error, assuming there is one.
  const ErrorTree &getError() const { return cast<ErrorTree>(value); }

  /// Take the underlying error, assuming there is one.
  ErrorTree takeError() { return std::move(cast<ErrorTree>(value)); }

  /// Returns true if there is a valid value.
  bool hasValue() const { return isa<T>(value); }

  /// Get a reference to the value, assuming there is one.
  const T &getValue() const { return cast<T>(value); }

  /// Take the underlying value, assuming there is one.
  T takeValue() { return std::move(cast<T>(value)); }

  /// Allow implicit conversion to bool. Returns true if is this an error.
  explicit operator bool() const { return isError(); }

  /// Allow the dereference operator to access the underlying value.
  const T &operator*() const { return getValue(); }

  /// Allow the arrow operator to access the underlying value.
  const T *operator->() const { return &getValue(); }

  /// Try to get a valid value. This method requires `T` to be
  /// default-constructible.
  T tryGetValue() const { return hasValue() ? getValue() : T(); }

  /// Explicitly copy this error or value. The value must have a copy
  /// constructor.
  ErrorTreeOr<T> copy() const {
    if (isError())
      return getError().copy();
    return getValue();
  }

private:
  /// This type is backed by a variant of an error and the value type.
  SmartVariant<ErrorTree, T> value;
};

/// This type is used for APIs that either succeed (with no result value) or can
/// return an ErrorTree.
class [[nodiscard]] ErrorTreeOrSuccess : public ErrorTreeOr<M::Detail::Empty> {
public:
  using ErrorTreeOr::ErrorTreeOr;
  /// This allows initialization from success().
  /*implicit*/ ErrorTreeOrSuccess(SuccessType success)
      : ErrorTreeOr(M::Detail::Empty()) {}

  /// Allow default initialization to success.
  ErrorTreeOrSuccess() : ErrorTreeOr(M::Detail::Empty()) {}
};

} // namespace M

#endif // SUPPORT_COMPILER_ERRORTREE_H
