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

#include "SingletonTypeHelper.h"

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Support/Compiler/OperationUtils.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

FlatSymbolRefAttr M::KGEN::LIT::flattenSymbolRefAttr(SymbolRefAttr ref) {
  // If the symbol is already flat, there is nothing to do.
  if (auto flatSym = dyn_cast<FlatSymbolRefAttr>(ref))
    return flatSym;

  // Flatten the symbol name into a single string.
  return FlatSymbolRefAttr::get(ref.getContext(), getFlattenedSymbolName(ref));
}

bool SingletonTypeHelper::isSingletonType(Type type) {
  if (sugarIsa<OriginType, OriginSetType>(type))
    return true;

  if (auto structType = dyn_cast<LIT::StructType>(type)) {
    return lookupStructSingletonFields(structType.getSymbol()) !=
           alwaysSingletonStructs.end();
  }
  return false;
}

TypedAttr SingletonTypeHelper::getSingletonValue(Type type) {
  TypedAttr result;
  if (auto origin = sugarDynCast<OriginType>(type))
    result = AnyOriginAttr::get(origin);
  else if (auto set = dyn_cast<OriginSetType>(type))
    result = OriginSetAttr::get(/*operands=*/{}, set);

  else if (auto structType = dyn_cast<LIT::StructType>(type)) {
    auto it = lookupStructSingletonFields(structType.getSymbol());
    if (it == alwaysSingletonStructs.end())
      return {};
    result = LITStructAttr::get(it->second, structType);
  }

  if (result)
    return ParamOperatorAttr::getRebind(result, type);

  return {};
}

SingletonTypeHelper::StructCacheTy::iterator
SingletonTypeHelper::lookupStructSingletonFields(SymbolRefAttr ref) {
  StringAttr refName = flattenSymbolRefAttr(ref).getAttr();
  // If this struct is a known singleton, return its singleton value.
  if (auto it = alwaysSingletonStructs.find(refName);
      it != alwaysSingletonStructs.end())
    return it;

  // If this struct is a known non-singleton, return null attr.
  if (notAlwaysSingletonStructs.contains(refName))
    return alwaysSingletonStructs.end();

  // If we repeat an in-progress struct decl, it indicates an illegal cycle.
  // End the check and consider it _not_ a singleton type. LowerLITTypes will
  // report the error.
  if (!inProgressStructs.insert(refName).second)
    return {};

  // This is a struct decl we haven't seen before. Lookup its fields and
  // populate the cache with it.
  StructCacheTy::iterator outputIter;
  // First check the processed struct decls map.
  if (auto declIter = processedStructs.structDecls.find(refName);
      declIter != processedStructs.structDecls.end()) {
    outputIter =
        populateStructSingletonFields(refName, declIter->second.fields);
  } else {
    // If not already processed, the StructDeclOp must already exist in the
    // symbol table.
    Operation *op = symtab.lookupSymbolIn(module, ref);
    if (op == nullptr) {
      // This might be the struct currently being lowered, in which case, the
      // symbol name have been updated, yet the cache entry has not yet being
      // populated.
      op = symtab.lookupSymbolIn(module, refName);
    }
    StructDeclOp decl = cast<StructDeclOp>(op);
    SmallVector<std::pair<StringAttr, Type>> fields;
    for (StructFieldOp field : decl.getFieldDecls())
      fields.emplace_back(field.getNameAttr(), field.getType());
    outputIter = populateStructSingletonFields(refName, fields);
  }

  inProgressStructs.erase(refName);
  return outputIter;
}

SingletonTypeHelper::StructCacheTy::iterator
SingletonTypeHelper::populateStructSingletonFields(
    StringAttr key, ArrayRef<std::pair<StringAttr, Type>> fields) {
  SmallVector<std::tuple<StringAttr, TypedAttr>> values;
  for (auto [name, type] : fields) {
    TypedAttr value = getSingletonValue(type);
    if (!value) {
      notAlwaysSingletonStructs.insert(key);
      return alwaysSingletonStructs.end();
    }
    values.emplace_back(name, value);
  }
  return alwaysSingletonStructs.try_emplace(key, std::move(values)).first;
}
