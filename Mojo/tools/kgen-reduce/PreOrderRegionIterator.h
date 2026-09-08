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

#ifndef KGEN_REDUCE_PREORDERREGIONITERATOR_H
#define KGEN_REDUCE_PREORDERREGIONITERATOR_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"

namespace M {
struct PreOrderRegionIterator {
  static PreOrderRegionIterator begin(Operation *op) {
    return PreOrderRegionIterator{&op->getRegion(0), op};
  }
  static PreOrderRegionIterator end(Operation *op) {
    return PreOrderRegionIterator{nullptr, op};
  }

  bool operator==(PreOrderRegionIterator rhs) const {
    assert(top == rhs.top && "different iterator family");
    return region == rhs.region;
  }
  bool operator!=(PreOrderRegionIterator rhs) const { return !(*this == rhs); }

  PreOrderRegionIterator operator++() {
    assert(region && "incrementing end iterator");
    // Find an op in the region that has a child region.
    for (Operation &op : region->getOps()) {
      if (op.getNumRegions()) {
        region = &op.getRegion(0);
        return *this;
      }
    }

    // Otherwise, try the next sibling region.
    Operation *parent = region->getParentOp();
    unsigned nextIdx = region->getRegionNumber() + 1;
    if (nextIdx != parent->getNumRegions()) {
      region = &parent->getRegion(nextIdx);
      return *this;
    }

    // Otherwise, find a sibling op with a region.
    while (parent != top) {
      for (Operation *next = parent->getNextNode(); next;
           next = next->getNextNode()) {
        if (next->getNumRegions()) {
          region = &next->getRegion(0);
          return *this;
        }
      }
      // Otherwise, go back up a level and continue.
      parent = parent->getParentOp();
    }

    region = nullptr;
    return *this;
  }

  Region &operator*() const { return *region; }

  Region *region;
  Operation *top;
};
} // namespace M

#endif // KGEN_REDUCE_PREORDERREGIONITERATOR_H
