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

#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_UPDATEMATERIALIZATION
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct UpdateMaterializationPass
    : impl::UpdateMaterializationBase<UpdateMaterializationPass> {
  void runOnOperation() override;
};
} // namespace

/// A pointer inside a comptime value that refers to the value's own interpreter
/// storage: `offset` is where the pointer sits in the value, `target` is the
/// byte it points at. Both are relative to the start of the value.
struct SelfReference {
  int64_t offset;
  int64_t target;
};

/// Whether `ref` points into the storage the value carrying it was read out of.
/// The marker is only recorded on a reference whose own blob is that storage,
/// so a reference carrying it is self-referential regardless of which memory
/// model it came from.
static bool isSelfReference(MemRefAttr ref) {
  return ref.getModel().getSelfBlobIndex() == ref.getIndex();
}

/// The blob a comptime value was read out of, if the value references it.
static std::optional<int64_t> getSelfBlobIndex(TypedAttr value) {
  std::optional<int64_t> result;
  value.walk([&](MemRefAttr ref) {
    if (!result && isSelfReference(ref))
      result = ref.getIndex();
    return WalkResult::advance();
  });
  return result;
}

/// The self-referential pointers of `value`, taken from the pointer regions its
/// own storage blob records. A region targeting that same blob is a pointer
/// from the value into itself.
static SmallVector<SelfReference> getSelfReferences(TypedAttr value,
                                                    int64_t selfBlobIndex) {
  SmallVector<SelfReference> refs;
  MemorySpaceAttr space;
  value.walk([&](MemRefAttr ref) {
    if (!space)
      space = ref.getModel().getMemory();
    return WalkResult::advance();
  });

  if (!space || selfBlobIndex < 0 ||
      static_cast<size_t>(selfBlobIndex) >= space.size())
    return refs;

  for (const PointerRegion &region : space[selfBlobIndex].getPointerRegions())
    if (region.blobIndex == selfBlobIndex)
      refs.push_back({region.offset, region.blobOffset});
  return refs;
}

/// Replace references to the value's own storage with null. The pointers are
/// written afterwards, relative to where the value actually lands, so leaving
/// the references in would make the materializer rebuild that storage as a
/// second object.
static TypedAttr dropSelfReferences(TypedAttr value) {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([](MemRefAttr ref) -> Attribute {
    if (!isSelfReference(ref))
      return ref;
    return PointerAttr::get(ref.getContext(), 0, ref.getType());
  });
  return cast<TypedAttr>(replacer.replace(value));
}

/// Write the self-referential pointers of a stored value relative to the
/// destination it was stored to.
static void patchStore(POP::StoreOp store, ArrayRef<SelfReference> refs) {
  mlir::OpBuilder b(store);
  Location loc = store.getLoc();

  b.setInsertionPointAfter(store);
  Value dest = store.getPtr();
  auto byteType = PointerType::get(b.getI8Type());
  Value bytes = POP::PointerBitcastOp::create(b, loc, byteType, dest);

  for (const SelfReference &ref : refs) {
    Value target = POP::OffsetOp::create(
        b, loc, bytes, mlir::index::ConstantOp::create(b, loc, ref.target));
    Value slot = POP::OffsetOp::create(
        b, loc, bytes, mlir::index::ConstantOp::create(b, loc, ref.offset));
    Value slotAsPtr =
        POP::PointerBitcastOp::create(b, loc, PointerType::get(byteType), slot);
    POP::StoreOp::create(b, loc, target, slotAsPtr);
  }
}

/// Aim the pointers of materialized comptime values that refer to their own
/// comptime storage at the storage they are materialized into.
static void resolveSelfReferentialMaterialize(Operation *root) {
  root->walk([](ParamMaterializeOp materialize) {
    TypedAttr value = materialize.getValue();
    std::optional<int64_t> selfBlobIndex = getSelfBlobIndex(value);
    if (!selfBlobIndex)
      return;

    // The references are dropped from the value itself, so every use has to be
    // a store: a use that consumes the value in register would see the null
    // pointers left behind instead of an address of its own.
    SmallVector<POP::StoreOp> stores;
    for (Operation *user : materialize->getUsers()) {
      auto store = dyn_cast<POP::StoreOp>(user);
      if (!store ||
          store.getArg().getDefiningOp<ParamMaterializeOp>() != materialize)
        return;
      stores.push_back(store);
    }
    if (stores.empty())
      return;

    SmallVector<SelfReference> refs = getSelfReferences(value, *selfBlobIndex);
    if (refs.empty())
      return;

    // Patch every destination before dropping the references, since dropping
    // them erases what the remaining stores would be patched from.
    for (POP::StoreOp store : stores)
      patchStore(store, refs);
    materialize.setValueAttr(dropSelfReferences(value));
  });
}

void UpdateMaterializationPass::runOnOperation() {
  resolveSelfReferentialMaterialize(getOperation());
}
