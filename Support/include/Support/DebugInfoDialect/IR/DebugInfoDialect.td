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

#ifndef DEBUGINFODIALECT_TD
#define DEBUGINFODIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// DebugInfo
//===----------------------------------------------------------------------===//

def DebugInfoDialect : Dialect {
  let name = "debuginfo";

  let summary = "Debug Information dialect";
  let description = [{
    The Debug Info dialect contains attributes, operations, and types for
    representing debug information in MLIR. The dialect is intended to be
    extensible, both in supporting different types of external debug
    information formats and different types of frontends. This dialect is
    intended to compose directly with any other, making as few assumptions
    about the abstraction level as reasonably possible.

    Given the nature of composing this dialect with others, the structure of
    various parts of the debug information, generally the types, have a notion
    of "resolvedness". We classify debug information as "resolved" when its
    components are purely comprised of other debug info components. This is an
    important concept to provide, because it allows for debug info to "lower"
    alongside the abstraction level it is describing. In many situations, we
    may not have a full understanding of the "native" debug representation of
    a type until deep within a compilation pipeline. This provides flexibility
    to "frontends", i.e. those generating the initial debug info, to describe
    as much as they have information for at the time.

    Debug information is generally attached to operations using a fused
    location whose metadata is a DIScopeAttr. "Function" operations should
    attach a DISubprogramAttr as the scope, which describes the debug
    information for the function. Nested operations within the function may
    use the parent function's DISubprogramAttr as the scope, or one of the
    other available DILocalScopeAttr attributes depending on the context. For
    example:

    ```mlir
    #subprogram = #debuginfo.subprogram<...>

    func.func @foo() {
      // The location of the return uses the parent subprogram as the scope.
      return loc(fused<#subprogram>("foo.cpp":5:10))

      // The location of the function references the corresponding subprogram
      // that describes the debug information for the function.
    } loc(fused<#subprogram>("foo.cpp":10:10))
    ```
  }];

  let cppNamespace = "::M::DebugInfo";
  let extraClassDeclaration = [{
  private:
    /// Register all dialect attributes.
    void registerAttributes();
    /// Register all dialect operations.
    void registerOperations();
    /// Register all dialect types.
    void registerTypes();

  public:
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Parameters
//===----------------------------------------------------------------------===//

// This class defines a base class for parameters that directly represent
// dwarf information. These effectively call into the LLVM dwarf library to get
// information for, e.g., pretty parsing/printing.
class DIDwarfParameter<string summary, string underlyingType, string parseName,
                       string printName = parseName>
    : AttrOrTypeParameter<underlyingType, "debug info " # summary> {
  let parser = [{ [&]() -> FailureOr<unsigned> {
    llvm::SMLoc tagLoc = $_parser.getCurrentLocation();
    StringRef name;
    if ($_parser.parseKeyword(&name))
      return failure();

    if (unsigned tag = llvm::dwarf::get}] # parseName # [{(name))
      return tag;
    return $_parser.emitError(tagLoc)
      << "invalid debug info }] # summary # [{ name: " << name;
  }() }];
  let printer = "$_printer << llvm::dwarf::" # printName # "String($_self)";
}

def DICallingConventionParameter : DIDwarfParameter<
  "calling convention", "unsigned", "CallingConvention", "Convention"
>;

def DIEncodingParameter : DIDwarfParameter<
  "encoding", "unsigned", "AttributeEncoding"
>;

def DILanguageParameter : DIDwarfParameter<
  "language", "unsigned", "Language"
>;

def DITagParameter : DIDwarfParameter<
  "tag", "unsigned", "Tag"
>;

#endif // DEBUGINFODIALECT_TD
