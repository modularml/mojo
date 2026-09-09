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

#ifndef SUPPORT_ADT_CONCATENATIONTREE_H
#define SUPPORT_ADT_CONCATENATIONTREE_H

#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace M {
class ConcatTreeBaseNode;

/// This data structure provides the ability to concatenate (potentially large)
/// strings/arrays together without ever copying the underlying data around.
/// This is implemented by building a tree of nodes representing the underlying
/// concatenated result.
///
/// When the final structure is finished, you can use the `traverse` method to
/// emit the data to a file or flatten it into a buffer of your choice once.
class ConcatenationTree {
public:
  ConcatenationTree() : node(nullptr) {}
  ConcatenationTree(ConcatenationTree &&rhs);
  ~ConcatenationTree();

  ConcatenationTree &operator=(ConcatenationTree &&rhs);

  /// Get an empty ConcatenationTree.
  static ConcatenationTree getEmpty() { return ConcatenationTree(); }

  /// Get a ConcatenationTree with the specified vector data.
  static ConcatenationTree takeVector(std::vector<uint8_t> data);

  /// Get a ConcatenationTree with the specified array data, which must be
  /// guaranteed to live beyond the lifetime of this ConcatenationTree.
  static ConcatenationTree getImmortalData(llvm::ArrayRef<uint8_t> data);

  /// Concatenate and return two trees of data.
  static ConcatenationTree concat(ConcatenationTree lhs, ConcatenationTree rhs);

  /// This returns the size in bytes of the collection of data that this
  /// represents.  This is O(1).
  size_t getSize() const;

  /// Iterate through this structure walking over the leaf node data in-order.
  void traverse(llvm::function_ref<void(ArrayRef<uint8_t>)> traversalFn);

private:
  ConcatenationTree(ConcatTreeBaseNode *nodePtr);
  ConcatTreeBaseNode *node;
};

} // namespace M

#endif // SUPPORT_ADT_CONCATENATIONTREE_H
