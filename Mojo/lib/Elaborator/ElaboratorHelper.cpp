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

#include "ElaboratorHelper.h"
#include "IREvaluatorContext.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/Support/NameMangling.h"
#include "Mojo/TransformUtils/ManglingUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace KGEN;

static void
replaceSymNames(Operation *op,
                const DenseMap<SymbolRefAttr, StringAttr> &symToRename) {
  if (symToRename.empty())
    return;

  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&symToRename](SymbolConstantAttr attr) {
    auto iter = symToRename.find(attr.getSymbol());
    if (iter != symToRename.end())
      return SymbolConstantAttr::get(iter->second, attr.getType(),
                                     attr.getParamValues());
    return attr;
  });

  replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/true,
                                        /*replaceTypes=*/true);
}

/// Apply the linkage name formula.
///
/// This function is called from two places that must produce identical names:
///
///   1. renameFunctions (post-elaboration, both host & offload side): renames
///      the concrete FuncOps in the elaborated module to their final symbols.
///
///   2. evaluateMangledName (during parameter evaluation, host side): computes
///      the name that get_linkage_name will return. This is typically used so
///      the host runtime knows what kernel entry point symbol to dispatch to.
///
/// Centralizing the formula here ensures the kernel entry point symbol written
/// into the binary and the name the host looks up are always identical.
StringAttr KGEN::applyLinkageName(StringAttr resolved, LinkageNameAttr lna,
                                  bool sanitize, StringRef symName,
                                  mlir::FunctionType funcType) {
  bool mangle = lna.getMangle().getValue();
  // First sanitize the linkage name if asked. If we're mangling the name, keep
  // all characters. Else, produce a short mangled name of 32 characters.
  // FIXME: Perhaps split apart the concepts of sanitization and name
  // shortening, and stop conflating it to exclusively offload targets.
  if (sanitize) {
    size_t charToKeep = !mangle ? 32 : std::numeric_limits<size_t>::max();
    resolved = sanitizeSymbolToUnderscores(resolved, charToKeep);
  }

  // If we're not mangling, we're done
  if (!mangle)
    return resolved;
  // Else append a unique suffix derived from the symName and the printed
  // function type. This matches the symbol produced by the offload rename loop
  // (renameFunctions) and the name looked up by the host (get_linkage_name /
  // evaluateMangledName).
  std::string funcTypeStr;
  llvm::raw_string_ostream os(funcTypeStr);
  funcType.print(os);
  return appendAutoMangledSuffix(resolved, symName, funcTypeStr);
}

void KGEN::renameFunctions(mlir::ModuleOp theModule, bool isGPU, bool &failed) {
  DenseMap<SymbolRefAttr, StringAttr> symToRename;
  DenseMap<StringAttr, FuncOp> renamedTo;

  for (FuncOp func : theModule.getOps<FuncOp>()) {
    StringAttr newName;

    if (auto linkageNameAttr = func.getLinkageNameAttr()) {
      auto prefixAttr = dyn_cast<StringAttr>(linkageNameAttr.getName());
      if (!prefixAttr) {
        failed = true;
        mlir::emitError(func.getLoc())
            << "unable to resolve `linkageName` '" << linkageNameAttr.getName()
            << "' to a string before renaming";
        func.removeLinkageNameAttr();
        continue;
      }
      StringRef prefix = prefixAttr.getValue();
      StringAttr userName = StringAttr::get(theModule.getContext(), prefix);
      newName = applyLinkageName(userName, linkageNameAttr, /*sanitize=*/isGPU,
                                 func.getSymName(), func.getFunctionType());
      func.removeLinkageNameAttr();
    }

    // When a linkage name is set, newName is already set above.
    // When a linkage name is absent, sanitize the auto-mangled sym_name as
    // before.
    if (isGPU && !newName)
      newName = sanitizeSymbolToAlnum(func.getSymNameAttr());

    if (!newName || newName == func.getSymNameAttr())
      continue;

    auto [it, inserted] = renamedTo.try_emplace(newName, func);
    if (!inserted) {
      failed = true;
      mlir::emitError(func.getLoc()) << "duplicate functions named " << newName;
      mlir::emitRemark(it->second.getLoc()) << "existing function here";
      continue;
    }
    StringAttr oldSym = func.getSymNameAttr();
    symToRename[FlatSymbolRefAttr::get(oldSym)] = newName;
    func.setSymName(newName);
    DebugInfo::updateSubprogram(func, newName);

    // On host targets, also rename the companion populate_captures stub when
    // a host function carries an explicit linkage name. Offload kernels live
    // only in standalone offload modules and never have stubs in the host
    // module, so this is a no-op for offload targets.
    if (!isGPU) {
      MLIRContext *ctx = theModule.getContext();
      SmallString<128> stubOldStr(oldSym.getValue());
      stubOldStr += "_populate_captures";
      SmallString<128> stubNewStr(newName.getValue());
      stubNewStr += "_populate_captures";
      if (Operation *stubOp = mlir::SymbolTable::lookupSymbolIn(
              theModule, StringRef(stubOldStr))) {
        // populate_captures stubs are always FuncOps — created as such in
        // evaluateCompileOffloadClosureAttr. A non-FuncOp here is a bug.
        auto stubFunc = cast<FuncOp>(stubOp);
        StringAttr stubOldAttr = StringAttr::get(ctx, StringRef(stubOldStr));
        StringAttr stubNewAttr = StringAttr::get(ctx, StringRef(stubNewStr));
        symToRename[FlatSymbolRefAttr::get(stubOldAttr)] = stubNewAttr;
        stubFunc.setSymName(stubNewAttr);
      }
    }
  }

  replaceSymNames(theModule, symToRename);
}

ErrorTreeOrSuccess KGEN::sortAndBundleOffloadOps(
    const llvm::SetVector<CompileOffloadOp> &offloadOps,
    mlir::SymbolTable &symTab,
    llvm::function_ref<ErrorTreeOrSuccess(CompileOffloadOp, StringAttr)>
        bundleOp) {
  struct BundleItem {
    StringAttr name;
    CompileOffloadOp op;
  };
  SmallVector<BundleItem> items;
  items.reserve(offloadOps.size());
  for (CompileOffloadOp op : offloadOps) {
    ErrorTreeOr<OffloadFunc> offloadFuncOr = extractOffloadFunc(op, symTab);
    if (offloadFuncOr.isError())
      return offloadFuncOr.takeError();
    auto [symbol, func] = offloadFuncOr.takeValue();

    // Use mangleParameterValues for the bundling name so it matches the name
    // evaluateCompileOffloadClosureAttr uses for the _populate_captures stub.
    StringAttr name = StringAttr::get(
        op.getContext(), mangleParameterValues(func, symbol.getParamValues()));
    items.push_back({name, op});
  }

  // The same kernel can be offloaded more than once (e.g. for different targets
  // or emission kinds), so tie-break equal names on the target and emission
  // attributes. The separators keep the key legible if it is ever dumped while
  // debugging a bundling-order question.
  auto emissionKey = [](CompileOffloadOp op) {
    std::string key;
    llvm::raw_string_ostream os(key);
    os << op.getTargetTypeAttr() << ';' << op.getEmissionKindAttr() << ';'
       << op.getEmissionOptionAttr() << ';' << op.getEmissionLinkOptionAttr();
    return key;
  };
  llvm::sort(items, [&](const BundleItem &lhs, const BundleItem &rhs) {
    if (lhs.name != rhs.name)
      return lhs.name.getValue() < rhs.name.getValue();
    return emissionKey(lhs.op) < emissionKey(rhs.op);
  });

  for (const BundleItem &item : items) {
    ErrorTreeOrSuccess result = bundleOp(item.op, item.name);
    if (result.isError())
      return result.takeError();
  }

  return success();
}
