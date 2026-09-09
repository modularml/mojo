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

#ifndef SUPPORT_BINARY_H
#define SUPPORT_BINARY_H

#include "Support/Buffer.h"
#include <memory>

namespace M {
/// This class provides a convenient wrapper around an owned binary whose
/// underlying buffer is tied to a buffer ref.
template <typename T>
class OwningBinary {
  std::unique_ptr<T> binary;
  BufferRef buffer;

public:
  OwningBinary() = default;
  OwningBinary(std::unique_ptr<T> binary, BufferRef buffer)
      : binary(std::move(binary)), buffer(std::move(buffer)) {}
  OwningBinary(OwningBinary<T> &&rhs)
      : binary(std::move(rhs.binary)), buffer(std::move(rhs.buffer)) {}
  OwningBinary<T> &operator=(OwningBinary<T> &&rhs) {
    binary = std::move(rhs.binary);
    buffer = std::move(rhs.buffer);
    return *this;
  }

  std::pair<std::unique_ptr<T>, BufferRef> takeBinary() {
    return std::make_pair(std::move(binary), std::move(buffer));
  }

  T *getBinary() { return binary.get(); }
  const T *getBinary() const { return binary.get(); }
};
} // namespace M

#endif // SUPPORT_BINARY_H
