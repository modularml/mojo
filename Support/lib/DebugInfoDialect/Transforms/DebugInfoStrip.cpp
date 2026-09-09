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

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "Support/DebugInfoDialect/Transforms/StripDebugInfo.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Visitors.h"
#include <optional>

using namespace M;
using namespace M::DebugInfo;

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

namespace M::DebugInfo {
#define GEN_PASS_DEF_DEBUGINFOSTRIP
#include "Support/DebugInfoDialect/Transforms/Transforms.h.inc"
} // namespace M::DebugInfo

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

void DebugInfo::stripDebugInfo(Operation *scope, bool preserveLineTables) {
  mlir::AttrTypeReplacer replacer;

  // If we're preserving line tables, we need to replace the compile unit
  // attribute with one that only contains line tables.
  if (preserveLineTables) {
    // Strip argument/result types from every DISubroutineType. In line-tables
    // mode only file/line data is needed; retaining full type metadata
    // (which can be large for parametric types) wastes space.
    //
    // The two replacers below work independently: AttrTypeReplacer recurses
    // into composite attributes before applying replacements, so both
    // DISubroutineType (nested inside DISubprogramAttr.type) and
    // DICompileUnitAttr (nested inside DISubprogramAttr.compileUnit) are
    // reached and updated without needing a replacer on DISubprogramAttr
    // itself.
    replacer.addReplacement(
        [](DebugInfo::DISubroutineType type) -> std::optional<mlir::Type> {
          return DISubroutineType::get(type.getContext(), {}, {});
        });
    replacer.addReplacement(
        [](DebugInfo::DICompileUnitAttr cu) -> std::optional<Attribute> {
          if (!cu || cu.getEmissionKind() != DebugInfo::EmissionKind::Full)
            return std::nullopt;
          return Attribute(DebugInfo::DICompileUnitAttr::get(
              cu.getSourceLanguage(), cu.getFile(), cu.getProducer(),
              cu.getIsOptimized(), DebugInfo::EmissionKind::LineTablesOnly,
              cu.getNameTableKind()));
        });

    // Otherwise, we strip debug info from locations.
  } else {
    replacer.addReplacement(
        [&](mlir::FusedLocWith<DIAttr> diLoc) -> LocationAttr {
          return FusedLoc::get(diLoc.getContext(), diLoc.getLocations());
        });
  }

  scope->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Drop all debug info operations.
    if (isa_and_nonnull<DebugInfoDialect>(op->getDialect())) {
      op->erase();
      return WalkResult::skip();
    }

    // For everything else, update the location.
    replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                               /*replaceLocs=*/true);
    return WalkResult::advance();
  });
}

namespace {
struct DebugInfoStrip : public impl::DebugInfoStripBase<DebugInfoStrip> {
  using Base::Base;

  void runOnOperation() override {
    stripDebugInfo(getOperation(), preserveLineTables);
  }
};
} // namespace
