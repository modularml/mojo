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

#include "Mojo/KGENDialect/AttrTypeMangler.h"
#include "Mojo/Support/CompilerProfiling.h"

using namespace M;
using namespace KGEN;

/// Split the name into the base name and a trailing id. If there is not
/// trailing number, -1 is returned.
static std::pair<StringRef, ssize_t> split(StringRef name) {
  // We first
  StringRef key = name.rtrim("0123456789");
  size_t splitIdx = key.size();

  // -1 means no number suffix.
  ssize_t id = -1;
  name.substr(splitIdx).getAsInteger(/*Radix=*/10, id);

  return std::make_pair(key, id);
}

NameUniquer::NameUniquer(const ParameterUseDefGraph &scope,
                         const ParameterUseDefGraph &topLevelGraph)
    : topLevelGraph(topLevelGraph) {
  updateMaxIds(scope);
}

/// Check if the name needs mangling.
bool NameUniquer::needsMangling(StringAttr name) {
  auto [key, id] = split(name);
  if (auto it = maxIds.find(key); it != maxIds.end())
    return id <= it->second;
  return false;
}

/// Uniquely mangle a parameter name. Returns the original name if mangling is
/// not needed.
StringAttr NameUniquer::mangle(StringAttr name) {
  if (!needsMangling(name))
    return name;
  auto [key, _] = split(name);
  ssize_t newId = ++maxIds[key];
  return StringAttr::get(name.getContext(), key + Twine(newId));
}

/// Update the uniquer with a new name.
void NameUniquer::updateWith(StringRef name) {
  auto [key, id] = split(name);
  ssize_t &max = maxIds.try_emplace(key, -1).first->second;
  max = std::max(max, id);
}

/// Update the ids we are tracking with the declarations (including those
/// nested) in the given scope.
void NameUniquer::updateMaxIds(const ParameterUseDefGraph &scope) {
  for (auto [declName, _] : scope.decls)
    updateWith(declName);
  for (Region *nestedRegion : scope.nestedDecls)
    updateMaxIds(topLevelGraph.nestedScopes.at(nestedRegion));
}

Attribute AttrTypeMangler::mangleRefsIn(Attribute attr, bool &hasRefs) {
  if (auto ref = dyn_cast<ParamDeclRefAttr>(attr)) {
    hasRefs = true;
    if (StringAttr mangled = mangledDecls.lookup(ref.getName()))
      return ParamDeclRefAttr::get(mangled,
                                   mangleRefsIn(ref.getType(), hasRefs));
  }
  return mangleRefsInImpl(attr, hasRefs);
}

bool AttrTypeMangler::populate(Builder &b, NameUniquer &uniquer,
                               const llvm::SetVector<StringAttr> &calleeDecls,
                               const ParameterUseDefGraph &topLevelGraph) {
  VerboseCompilerTimeTraceScope traceScope("AttrTypeMangler::populate");

  // `curScope` contains all declarations visible in the scope of the call,
  // including those defined in higher scopes. When the function is inlined,
  // these are the declarations that will project into the inlined body. We need
  // to mangle parameters in the inlined body such that they do not collide with
  // any declarations visible in the call scope, or in any nested scopes.
  // Populate `mangleIdx` with the indices of the declarations that need
  // mangling. This ensures we rename as few declarations as possible.

  SmallVector<size_t> mangleIdx;
  for (auto [i, decl] : llvm::enumerate(calleeDecls)) {
    if (uniquer.needsMangling(decl))
      mangleIdx.push_back(i);
    uniquer.updateWith(decl);
  }

  if (mangleIdx.empty())
    return false;

  for (size_t i : mangleIdx) {
    StringAttr decl = calleeDecls[i];
    auto mangled = uniquer.mangle(decl);
    mangledDecls.try_emplace(decl, mangled);
  }

  return true;
}

ParamDeclAttr AttrTypeMangler::mangleDecl(ParamDeclAttr decl,
                                          bool needsMangling) {
  if (!needsMangling)
    return decl;
  Type type = mangleRefsIn(decl.getType());
  if (StringAttr mangled = mangledDecls.lookup(decl.getName()))
    return ParamDeclAttr::get(mangled, type);
  if (type == decl.getType())
    return decl;
  return ParamDeclAttr::get(decl.getName(), type);
}

void AttrTypeMangler::mangleElementsIn(Operation *op) {
  op->setAttrs(cast<DictionaryAttr>(mangleRefsIn(op->getAttrDictionary())));

  for (OpResult result : op->getResults())
    result.setType(mangleRefsIn(result.getType()));

  for (Region &region : op->getRegions())
    for (BlockArgument arg : region.front().getArguments())
      arg.setType(mangleRefsIn(arg.getType()));
}

void AttrTypeMangler::recursivelyMangle(Region *scope,
                                        const ParameterUseDefGraph &graph) {
  VerboseCompilerTimeTraceScope traceScope(
      "AttrTypeMangler::recursivelyMangle");

  const ParameterUseDefGraph &uses = graph.nestedScopes.find(scope)->second;

  for (Operation *op : uses.paramOps)
    if (op != scope->getParentOp())
      mangleElementsIn(op);
  for (auto &[_, decl] : uses.decls)
    if (scope->getParentOp()->isProperAncestor(decl.declOp))
      mangleElementsIn(decl.declOp);

  for (Region *nestedScope : uses.nestedDecls)
    recursivelyMangle(nestedScope, graph);
}
