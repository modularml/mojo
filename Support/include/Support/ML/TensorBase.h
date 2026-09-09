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
// TensorBase combines a TensorSpec with an templated 'Buffer'.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_TENSORBASE_H
#define SUPPORT_ML_TENSORBASE_H

#include "Support/ML/DType.h"
#include "Support/ML/TensorShape.h"
#include "Support/ML/TensorSpec.h"
#include <cassert>
#include <cstddef>
#include <utility>

namespace M {

/// A TensorBase is a pair of a 'BufferRefType' and TensorSpec.
///
/// BufferRefType must be moveable, and implement:
///    // Returns byte size of buffer held by buffer reference.
///    size_t getSize() const
template <typename BufferRefType>
class TensorBase {
public:
  TensorBase(BufferRefType &&bufferRef, TensorSpec &&spec)
      : bufferRef(std::move(bufferRef)), spec(std::move(spec)) {
    assert(this->spec.getSizeInBytes() == this->bufferRef.getSize() &&
           "tensor buffer and spec disagree on byte size");
  }

  /// Returns the underlying buffer reference.
  const BufferRefType &getBufferRef() const { return bufferRef; }
  BufferRefType &getBufferRef() { return bufferRef; }

  /// Returns the tensor spec/shape/element type.
  const TensorSpec &getSpec() const { return spec; }
  TensorSpec &getSpec() { return spec; }
  const TensorShape &getShape() const { return spec; }
  DType getEltType() const { return spec.getEltType(); }

  /// Returns the size of the underlying buffer in bytes.
  size_t getSizeInBytes() const { return bufferRef.getSize(); }

  /// Tensors cannot be implicitly copied.
  void operator=(const TensorBase<BufferRefType> &) = delete;
  TensorBase(const TensorBase<BufferRefType> &rhs) = delete;

  /// Tensors can be moved.
  TensorBase(TensorBase<BufferRefType> &&rhs) = default;
  TensorBase &operator=(TensorBase<BufferRefType> &&rhs) = default;

protected:
  BufferRefType bufferRef;
  TensorSpec spec;
};

} // namespace M

#endif // SUPPORT_ML_TENSORBASE_H
