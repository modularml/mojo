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

#include "Support/ADT/ConcatenationTree.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

using namespace M;

//===----------------------------------------------------------------------===//
// ConcatTreeBaseNode
//===----------------------------------------------------------------------===//

/// This is an abstract base class for all concatenation tree nodes.
namespace M {
class ConcatTreeBaseNode {
public:
  const enum NodeKind {
    kVector,       // a node holding an std::vector of data.
    kImmortalData, // a node holding an immortal ArrayRef of data.
    kBranch,       // a node with two or more subnodes.
  } nodeKind;

  void destroy();

protected:
  ConcatTreeBaseNode(NodeKind kind) : nodeKind(kind) {}
};
} // namespace M

//===----------------------------------------------------------------------===//
// ConcatTreeVectorNode
//===----------------------------------------------------------------------===//

namespace {
/// This is a leaf node that holds an std::vector of data.
class ConcatTreeVectorNode : public ConcatTreeBaseNode {
public:
  static bool classof(const ConcatTreeBaseNode *base) {
    return base->nodeKind == kVector;
  }

  ConcatTreeVectorNode(std::vector<uint8_t> data)
      : ConcatTreeBaseNode(kVector), data(std::move(data)) {}

  std::vector<uint8_t> data;
};
} // namespace

//===----------------------------------------------------------------------===//
// ConcatTreeImmortalDataNode
//===----------------------------------------------------------------------===//

namespace {
/// This is a leaf node that holds an immortal ArrayRef of data.
class ConcatTreeImmortalDataNode : public ConcatTreeBaseNode {
public:
  static bool classof(const ConcatTreeBaseNode *base) {
    return base->nodeKind == kImmortalData;
  }

  ConcatTreeImmortalDataNode(ArrayRef<uint8_t> data)
      : ConcatTreeBaseNode(kImmortalData), data(data) {}

  ArrayRef<uint8_t> data;
};
} // namespace

//===----------------------------------------------------------------------===//
// ConcatTreeBranchNode
//===----------------------------------------------------------------------===//

namespace {
/// This class is an interior node in the tree, which has pointers to other
/// nodes.  This can store multiple children to reduce allocations of the branch
/// nodes themselves.
class ConcatTreeBranchNode : public ConcatTreeBaseNode {
public:
  ConcatTreeBranchNode(ConcatenationTree lhs, ConcatenationTree rhs)
      : ConcatTreeBaseNode(kBranch) {
    child[0] = std::move(lhs);
    child[1] = std::move(rhs);
    totalSize = child[0].getSize() + child[1].getSize();
  }

  static bool classof(const ConcatTreeBaseNode *base) {
    return base->nodeKind == kBranch;
  }

  // We support up to four children in this, reducing the number of tree nodes
  // that get created.
  ConcatenationTree child[4];
  size_t totalSize;
};
} // namespace

//===----------------------------------------------------------------------===//
// ConcatenationTree implementation logic
//===----------------------------------------------------------------------===//

ConcatenationTree::ConcatenationTree(ConcatTreeBaseNode *node) : node(node) {}
ConcatenationTree::ConcatenationTree(ConcatenationTree &&rhs) : node(rhs.node) {
  rhs.node = nullptr;
}

ConcatenationTree &ConcatenationTree::operator=(ConcatenationTree &&rhs) {
  if (node)
    node->destroy();
  node = rhs.node;
  rhs.node = nullptr;
  return *this;
}

ConcatenationTree::~ConcatenationTree() {
  if (node)
    node->destroy();
}

void ConcatTreeBaseNode::destroy() {
  if (auto *vec = dyn_cast<ConcatTreeVectorNode>(this))
    delete vec;
  else if (auto *immData = dyn_cast<ConcatTreeImmortalDataNode>(this))
    delete immData;
  else
    delete cast<ConcatTreeBranchNode>(this);
}

/// Get a ConcatenationTree with the specified vector data.
ConcatenationTree ConcatenationTree::takeVector(std::vector<uint8_t> data) {
  if (data.empty())
    return getEmpty();

  return new ConcatTreeVectorNode(std::move(data));
}

/// Get a ConcatenationTree with the specified array data, which must be
/// guaranteed to live beyond the lifetime of this ConcatenationTree.
ConcatenationTree ConcatenationTree::getImmortalData(ArrayRef<uint8_t> data) {
  if (data.empty())
    return getEmpty();

  return new ConcatTreeImmortalDataNode(data);
}

/// Concatenate and return two trees of data.
ConcatenationTree ConcatenationTree::concat(ConcatenationTree lhs,
                                            ConcatenationTree rhs) {
  // Collapse null trees away.
  if (!lhs.node)
    return rhs;
  if (!rhs.node)
    return lhs;

  // Concat nodes have extra space in them that we can fill up to avoid
  // allocating new concat nodes.

  // If the left side is a branch node with space, we can add nodes to it
  // instead of allocating another branch.
  if (auto *lhsConcat = dyn_cast<ConcatTreeBranchNode>(lhs.node)) {
    if (!lhsConcat->child[2].node) {
      lhsConcat->totalSize += rhs.getSize();
      lhsConcat->child[2] = std::move(rhs);
      return lhs;
    }
    if (!lhsConcat->child[3].node) {
      lhsConcat->totalSize += rhs.getSize();
      lhsConcat->child[3] = std::move(rhs);
      return lhs;
    }
  }

  // If the right side is a concat node with space, we can push into it.
  if (auto *rhsConcat = dyn_cast<ConcatTreeBranchNode>(rhs.node)) {
    if (!rhsConcat->child[2].node) {
      rhsConcat->totalSize += lhs.getSize();
      // Move everything down so we can insert to the left of them.
      rhsConcat->child[2] = std::move(rhsConcat->child[1]);
      rhsConcat->child[1] = std::move(rhsConcat->child[0]);
      rhsConcat->child[0] = std::move(lhs);
      return rhs;
    }
    if (!rhsConcat->child[3].node) {
      rhsConcat->totalSize += lhs.getSize();
      // Move everything down so we can insert to the left of them.
      rhsConcat->child[3] = std::move(rhsConcat->child[2]);
      rhsConcat->child[2] = std::move(rhsConcat->child[1]);
      rhsConcat->child[1] = std::move(rhsConcat->child[0]);
      rhsConcat->child[0] = std::move(lhs);
      return rhs;
    }
  }

  return new ConcatTreeBranchNode(std::move(lhs), std::move(rhs));
}

/// This returns the size in bytes of the collection of data that this
/// represents.  This is O(1).
size_t ConcatenationTree::getSize() const {
  if (node == nullptr)
    return 0;

  if (auto *vec = dyn_cast<ConcatTreeVectorNode>(node))
    return vec->data.size();
  if (auto *immData = dyn_cast<ConcatTreeImmortalDataNode>(node))
    return immData->data.size();

  return cast<ConcatTreeBranchNode>(node)->totalSize;
}

/// Iterate through this structure walking over the leaf node data in-order.
void ConcatenationTree::traverse(
    llvm::function_ref<void(ArrayRef<uint8_t>)> fn) {
  // Null is an empty tree.
  if (node == nullptr)
    return;

  // Vector and immortal nodes are leaves.
  if (auto *vec = dyn_cast<ConcatTreeVectorNode>(node))
    return fn(vec->data);
  if (auto *immData = dyn_cast<ConcatTreeImmortalDataNode>(node))
    return fn(immData->data);

  // Otherwise walk through a branch.
  auto *branch = cast<ConcatTreeBranchNode>(node);
  branch->child[0].traverse(fn);
  branch->child[1].traverse(fn);

  // Child #2/3 are optional, traverse if present.
  if (branch->child[2].node) {
    branch->child[2].traverse(fn);
    if (branch->child[3].node)
      branch->child[3].traverse(fn);
  }
}
