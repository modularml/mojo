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
// This file declares the M::Error type and related support logic.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ERROR_H
#define SUPPORT_ERROR_H

#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "llvm/ADT/Twine.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string.h>
#include <utility>

namespace M {

/// This is a lightweight error class that holds a nul-terminated string, with a
/// static string optimization that does not allocate.  This is not implicitly
/// copyable because that may invoke allocation, use the `copy()` method to make
/// this explicit if you want that.
///
/// By convention, error strings do not include a trailing \n in the string,
/// but do include a trailing period or other terminator.
///
class [[nodiscard]] Error final {
  enum StorageMode : uint8_t {
    kStaticError, // This contains a pointer to static data.  No allocation.
    kMallocError, // This contains a malloc'd string.
    kValue,       // This is a normal value (used by ErrorOr).
  };

public:
  /// Construct an Error with a static error string.
  template <size_t n>
  Error(const char (&message)[n]) : value(message), storageMode(kStaticError) {
    static_assert(n > 1, "empty errors are not allowed");
  }

  /// Construct an Error with a dynamic Twine value (including std::string,
  /// const char *, etc).
  Error(const Twine &message);

  /// Construct an Error with a known-static string that doesn't need lifetime
  /// management.
  static Error getStaticString(const char *message) {
    assert(message && *message && "empty error strings are not allowed");
    Error result;
    result.value = message;
    result.storageMode = kStaticError;
    return result;
  }

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
  Error(Error &&other) : value(other.value), storageMode(other.storageMode) {
    other.value = nullptr;
    other.storageMode = kStaticError;
  }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

  ~Error() {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif
    if (storageMode == kMallocError)
      free(const_cast<void *>(static_cast<const void *>(value)));
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
  }

  /// Return the message this contains as a nul-terminated string.
  const char *get() const { return value; }

  // Explicit copy operation.
  Error copy() const {
    Error result;
    result.storageMode = storageMode;
    if (storageMode == kMallocError)
      result.value = strdup(value);
    else
      result.value = value;
    return result;
  }

  /// Convert this Error into a LogicalResult.
  /*implicit*/ operator LogicalResult() const { return failure(); }

  Error &operator=(Error &&other) {
    if (&other != this) {
      this->~Error();
      new (this) Error(std::move(other));
    }
    return *this;
  }

  /// Support [raw]ostream insertion.
  template <typename Stream>
  friend Stream &operator<<(Stream &os, const Error &error) {
    os << error.get();
    return os;
  }

private:
  Error() = default;
  Error(const Error &) = delete;                 // use copy() explicitly.
  Error &operator=(const Error &other) = delete; // use copy() explicitly.
  template <typename T>
  friend class M::ErrorOr;

  // Stored state.
  const char *value;
  StorageMode storageMode : 2;
};

bool operator==(const Error &, const Error &);
inline bool operator!=(const Error &a, const Error &b) { return !(a == b); }

/// Convert an LLVM error (which must be in error state) to a Modular error.
/// Note that while llvm::Error has a "success"/"no-error" state, M::Error does
/// not.  If you are unsure whether or not the llvm::Error is in error state,
/// use toModularErrorOr to get an M::ErrorOrSuccess instead.
Error toModularError(llvm::Error llvmError);

} // namespace M

#endif // SUPPORT_ERROR_H
