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

#include "Mojo/TransformUtils/SlicingUtils.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

using namespace M;
using namespace KGEN;

/// Dig into an attribute or type to find references to other symbols. If we see
/// a new one, copy it over.
template <typename AttrOrType>
static void sliceDependenciesFrom(AttrOrType value, SymbolTable &sliceSymtab,
                                  const SymbolTable &symtab,
                                  DenseSet<const void *> &visited,
                                  IRMapping &reusedMapping,
                                  std::vector<Operation *> &worklist) {
  // If we've already visited this value, we know we've extracted all
  // dependencies from it and its subtree.
  if (!visited.insert(value.getAsOpaquePointer()).second)
    return;

  // Check if this is a symbol reference.
  if constexpr (std::is_same_v<AttrOrType, Attribute>) {
    // Decorators are currently hacky and won't appear in the symbol table, so
    // we skip them.
    if (isa<KGEN::DecoratorsAttr>(value))
      return;

    if (auto ref = dyn_cast<FlatSymbolRefAttr>(value)) {
      // We know this is a new symbol because this is the first time we've
      // visited this attribute.
      StringAttr name = ref.getAttr();
      Operation *symbol = symtab.lookup(name);
      // If the symbol reference attribute doesn't reference a symbol, it must
      // be a reference that is not relative to `symtab`. Do not clone it.
      if (!symbol)
        return;

      // Clone the symbol into the new symbol table. Reuse an IRMapping to save
      // memory pressure.
      reusedMapping.clear();
      Operation *copy = symbol->clone(reusedMapping);
      sliceSymtab.insert(copy);

      // We need to recurse on this newly cloned operation.
      worklist.push_back(copy);

      // There are no further subelements. We can exit.
      return;
    }
  }

  // Recurse into subelements.
  value.walkImmediateSubElements(
      [&](Attribute attr) {
        sliceDependenciesFrom(attr, sliceSymtab, symtab, visited, reusedMapping,
                              worklist);
      },
      [&](Type type) {
        sliceDependenciesFrom(type, sliceSymtab, symtab, visited, reusedMapping,
                              worklist);
      });
}

/// Slice the dependencies of an operation out of the existing module into the
/// self-contained slice module.
void M::sliceDependencies(Operation *op, SymbolTable &sliceSymtab,
                          const SymbolTable &symtab, IRMapping &reusedMapping,
                          DenseSet<const void *> &visited) {
  std::vector<Operation *> worklist;
  auto visit = [&](auto value) {
    sliceDependenciesFrom(value, sliceSymtab, symtab, visited, reusedMapping,
                          worklist);
  };
  auto extractDependencies = [&](Operation *op) {
    // Extract references to type declarations.
    visit(op->getAttrDictionary());
    for (Type type : op->getResultTypes())
      visit(type);
    for (Region &region : op->getRegions())
      for (Type type : region.getArgumentTypes())
        visit(type);
  };

  worklist.push_back(op);
  while (!worklist.empty()) {
    Operation *op = worklist.back();
    worklist.pop_back();
    op->walk(extractDependencies);
  }
}

OwningOpRef<ModuleOp>
M::produceStandaloneModule(const SymbolTable &symtab,
                           const ExportMap &exportedSymbols,
                           bool overrideExported) {
  IRMapping unused;
  return produceStandaloneModule(symtab, exportedSymbols, unused,
                                 overrideExported);
}

OwningOpRef<ModuleOp>
M::produceStandaloneModule(const SymbolTable &symtab,
                           const ExportMap &exportedSymbols, IRMapping &mapping,
                           bool overrideExported) {

  DenseSet<const void *> visited;
  CompilerTimeTraceScope traceScope("produceStandaloneModule");
  auto module = cast<ModuleOp>(symtab.getOp());
  // Create a new module for these funcs. This will go away at the end
  // of this function.
  OwningOpRef<ModuleOp> singleModule = ModuleOp::create(module->getLoc());
  singleModule.get()->setAttrs(module->getAttrDictionary());

  // Create a new symbol table for the sliced module.
  SymbolTable sliceSymtab(*singleModule);

  IRMapping reusedMapping;
  DenseSet<ExportInterface> exported;

  for (auto [sym, exportValKind] : exportedSymbols) {
    auto func = symtab.lookup<ExportInterface>(sym);
    assert(func && "Unknown exported symbol");

    // Traverse the call graph and clone all the callees into this module.
    sliceDependencies(func, sliceSymtab, symtab, reusedMapping, visited);

    // Clone the func into this new module. We don't want to remove it from
    // the current module. Make sure the function is also exported in the slice.
    auto sliceFn = sliceSymtab.lookup<ExportInterface>(sym);
    if (!sliceFn) {
      sliceFn = cast<ExportInterface>(func->clone(mapping));
      sliceSymtab.insert(sliceFn);
    } else {
      mapping.map(func.getOperation(), sliceFn.getOperation());
    }
    ExportKind kind = func.getExportKind();
    sliceFn.setExportKind(kind == ExportKind::NotExported ? exportValKind
                                                          : kind);
    exported.insert(sliceFn);
  }

  if (overrideExported) {
    // Override exported info here for some cases.
    // For example:
    // GPU kernel should only have kernel entry function as exported.
    // Mark anyone else to not be exported here.
    // Some of they may be marked as exported because graph compiler needs
    // them to be or are needed as CPU functions .
    singleModule->walk([&](ExportInterface func) {
      if (!exported.contains(func) &&
          func.getExportKind() == ExportKind::Exported)
        func.setExportKind(ExportKind::NotExported);
    });
  }

  return singleModule;
}
