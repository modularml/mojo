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
// Assertions that can output values
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ASSERT_STREAM_H
#define SUPPORT_ASSERT_STREAM_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <stdlib.h>
#include <string>

namespace M {

/// Structure holding information about a failed assertion. Allows user to
/// append information to the error message.
struct FailedAssertion {
  LLVM_ATTRIBUTE_NOINLINE FailedAssertion(llvm::StringRef file_path,
                                          int64_t lineno) {
    getStorage().message << file_path << ":" << lineno << ": ";
  }
  [[noreturn]] ~FailedAssertion() {
    llvm::errs() << getStorage().message.str() << "\n";
    abort();
  }
  // Store error message info as a thread local struct. Avoids creating data for
  // each assertion which can blow the stack.
  struct Storage {
    std::string message_storage;
    llvm::raw_string_ostream message;
    Storage() : message_storage(), message(message_storage) {}
  };
  LLVM_ATTRIBUTE_NOINLINE static Storage &getStorage() {
    static thread_local Storage storage;
    return storage;
  }
};
} // namespace M

/// Assert that a condition holds. Users can append additional information to
/// the error message by using the << operator.
/// ```
///   ASSERT_STREAM(false, << "This condition is always false!");
/// ```
/// Like `assert`, this does nothing if the NDEBUG symbol is defined.
///
#ifdef NDEBUG
#define ASSERT_STREAM(condition, message_stream)                               \
  {                                                                            \
    do {                                                                       \
    } while (0);                                                               \
  }
#else
#define ASSERT_STREAM(condition, message_stream)                               \
  if (LLVM_UNLIKELY(!(condition)))                                             \
  ::M::FailedAssertion(__FILE__, __LINE__).getStorage().message                \
      << "Assertion failed: (" << #condition << ") is false.\n" message_stream
#endif // NDEBUG

#endif // SUPPORT_ASSERT_STREAM_H
