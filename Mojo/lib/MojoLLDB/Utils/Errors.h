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

#ifndef KGEN_LIB_MOJOLLDB_UTILS_ERRORS_H
#define KGEN_LIB_MOJOLLDB_UTILS_ERRORS_H

#include "llvm/Support/Error.h"

namespace M::KGEN::Mojo {

/// If \p expected does not contain an error, return the value.
/// Otherwise, consume the error and return \p defaultVal.
///
/// This calls llvm::consumeError() in the error path and therefore is subject
/// to the same constraints and warnings as llvm::consumeError(). Use of this
/// wrapper are potentially indicative of design issues, and hopefully should
/// be temporary. See llvm::consumeError() for further details.
template <class T>
T getExpectedValueOr(llvm::Expected<T> expected, T defaultVal) {
  if (expected)
    return expected.get();
  llvm::consumeError(expected.takeError());
  return defaultVal;
}
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_UTILS_ERRORS_H
