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

#ifndef SUPPORT_RANDOM_H
#define SUPPORT_RANDOM_H

#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include <cstdint>

namespace M {
/// Generate `numBytes` cryptographically-secure random numbers use them to
/// completely fill `buf`. Returns an error if we were unable to generate random
/// numbers for whatever reason.
struct SecureRandomBytesGenerator {
  /// Non-trivial constructors and destructors for Windows.
  SecureRandomBytesGenerator();
  ~SecureRandomBytesGenerator();

  /// Non-copyable, but move-able.
  SecureRandomBytesGenerator(const SecureRandomBytesGenerator &other) = delete;
  SecureRandomBytesGenerator(SecureRandomBytesGenerator &&other) = default;

  /// Actually perform the random number generation.
  ErrorOrSuccess getRandomBytes(MutableArrayRef<uint8_t> buf);

  /// Needed for Windows. On Windows, this is an HCRYPTPROV, which is a typedef
  /// for a pointer.
  void *ctx = nullptr;
};
} // namespace M

#endif // SUPPORT_RANDOM_H
