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

#include "Support/Error.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <utility>

using namespace M;

/// Construct an Error with a dynamic Twine value (including std::string,
/// const char *, etc).
///
/// This is intentionally out of line, because we don't want error handling
/// logic bloating out libraries that produce the errors.
Error::Error(const llvm::Twine &message) : storageMode(kMallocError) {
  llvm::SmallVector<char, 128> tmp;
  llvm::StringRef str = message.toStringRef(tmp);
  assert(!str.empty() && "empty error strings are not allowed");
  auto *ptr = (char *)malloc(str.size() + 1);
  if (ptr == nullptr)
    std::abort();
  memcpy(ptr, str.data(), str.size());
  ptr[str.size()] = 0;
  value = ptr;
}

bool M::operator==(const Error &a, const Error &b) {
  return strcmp(a.get(), b.get()) == 0;
}

Error M::toModularError(llvm::Error error) {
  assert(error && "Successful (non-error) llvm::Error values do not have an "
                  "M::Error equivalent");
  return Error(llvm::toString(std::move(error)));
}
