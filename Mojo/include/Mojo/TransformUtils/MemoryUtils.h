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

#ifndef KGEN_TRANSFORMUTILS_MEMORYUTILS_H
#define KGEN_TRANSFORMUTILS_MEMORYUTILS_H

#include "mlir/IR/Value.h"

namespace M::KGEN {
/// This CRTP class defines a simple recursive iterator over the uses of a
/// "projection" of a value, where the derived class defines how values are
/// projected.
///
/// For example, we may consider the extent of uses of a projection over a stack
/// allocation to apply side effects to the underlying stack allocation. One can
/// use this type to iterate over the uses of a stack allocation while "seeing
/// through" trivially aliasing operations by defining the `project` function in
/// the derived class.
template <typename DerivedT>
class ProjectionUseIterator {
public:
  explicit ProjectionUseIterator(Value value) {
    stack.push_back(Frame(value));
    // Start by moving the iterator to the first use or to the end.
    advance();
  }

  /// Return true if the iterator is at the end.
  bool isAtEnd() const { return stack.empty(); }

  /// Access the current projection use.
  OpOperand &operator*() {
    assert(!isAtEnd());
    return *stack.back().it;
  }

  /// Advance the iterator.
  void operator++() {
    assert(!isAtEnd());
    // We know the current frame iterator is not at its end. Advance it and then
    // try to advance the overall projection iterator.
    ++stack.back().it;
    advance();
  }

private:
  DerivedT &getDerived() { return *static_cast<DerivedT *>(this); }

  void advance() {
    assert(!isAtEnd());
    do {
      Frame &frame = stack.back();

      // If the current frame is at the end of its user range, pop up.
      if (frame.it == frame.e) {
        stack.pop_back();
        continue;
      }

      // Check if the current user is a projection.
      OpOperand &use = *frame.it;
      Value proj = getDerived().project(use);

      // If not, then this is the next use we will visit.
      if (!proj)
        return;

      // We have to recurse on this value. Advance the current iterator first,
      // since we will pop back to this.
      ++frame.it;
      stack.push_back(Frame(proj));

    } while (!isAtEnd());
  }

  struct Frame {
    explicit Frame(Value value)
        : value(value), it(value.use_begin()), e(value.use_end()) {}

    Value value;
    Value::use_iterator it;
    Value::use_iterator e;
  };

  SmallVector<Frame> stack;
};
} // namespace M::KGEN

#endif // KGEN_TRANSFORMUTILS_MEMORYUTILS_H
