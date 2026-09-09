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

#ifndef DEBUGINFOENUMS_TD
#define DEBUGINFOENUMS_TD

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// DIEmissionKind
//===----------------------------------------------------------------------===//

def DIEmissionNone                : I64EnumAttrCase<"None", 0>;
def DIEmissionFull                : I64EnumAttrCase<"Full", 1>;
def DIEmissionLineTablesOnly      : I64EnumAttrCase<"LineTablesOnly", 2>;
def DIEmissionDebugDirectivesOnly : I64EnumAttrCase<"DebugDirectivesOnly", 3>;

def DIEmissionKind : I64EnumAttr<
    "EmissionKind",
    "Debug emission kind", [
      DIEmissionNone,
      DIEmissionFull,
      DIEmissionLineTablesOnly,
      DIEmissionDebugDirectivesOnly,
    ]> {
  let cppNamespace = "::M::DebugInfo";
}

//===----------------------------------------------------------------------===//
// DINameTableKind
//===----------------------------------------------------------------------===//

def DINameTableDefault : I64EnumAttrCase<"Default", 0>;
def DINameTableGNU     : I64EnumAttrCase<"GNU", 1>;
def DINameTableNone    : I64EnumAttrCase<"None", 2>;
def DINameTableApple   : I64EnumAttrCase<"Apple", 3>;

def DINameTableKind : I64EnumAttr<
    "NameTableKind",
    "Debug name table kind", [
      DINameTableDefault,
      DINameTableGNU,
      DINameTableNone,
      DINameTableApple,
    ]> {
  let cppNamespace = "::M::DebugInfo";
}

//===----------------------------------------------------------------------===//
// DISubprogramFlags
//===----------------------------------------------------------------------===//

def DISPNone           : I32BitEnumAttrCaseNone<"None">;
def DISPVirtual        : I32BitEnumAttrCaseBit<"Virtual", 0>;
def DISPPureVirtual    : I32BitEnumAttrCaseBit<"PureVirtual", 1>;
def DISPLocalToUnit    : I32BitEnumAttrCaseBit<"LocalToUnit", 2>;
def DISPDefinition     : I32BitEnumAttrCaseBit<"Definition", 3>;
def DISPOptimized      : I32BitEnumAttrCaseBit<"Optimized", 4>;
def DISPPure           : I32BitEnumAttrCaseBit<"Pure", 5>;
def DISPElemental      : I32BitEnumAttrCaseBit<"Elemental", 6>;
def DISPRecursive      : I32BitEnumAttrCaseBit<"Recursive", 7>;
def DISPMainSubprogram : I32BitEnumAttrCaseBit<"MainSubprogram", 8>;
def DISPDeleted        : I32BitEnumAttrCaseBit<"Deleted", 9>;
def DISPObjCDirect     : I32BitEnumAttrCaseBit<"ObjCDirect", 11>;

def DISubprogramFlags : I32BitEnumAttr<
    "SubprogramFlags",
    "Subprogram specific flags", [
      DISPNone,
      DISPVirtual,
      DISPPureVirtual,
      DISPLocalToUnit,
      DISPDefinition,
      DISPOptimized,
      DISPPure, 
      DISPElemental,
      DISPRecursive,
      DISPMainSubprogram,
      DISPDeleted,
      DISPObjCDirect
    ]> {
  let cppNamespace = "::M::DebugInfo";
  let printBitEnumPrimaryGroups = 1;
}

#endif // DEBUGINFOENUMS_TD
