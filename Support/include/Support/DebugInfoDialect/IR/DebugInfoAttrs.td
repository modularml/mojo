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

#ifndef DEBUGINFOATTRS_TD
#define DEBUGINFOATTRS_TD

include "Support/DebugInfoDialect/IR/DebugInfoDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// DebugInfoAttr
//===----------------------------------------------------------------------===//

// Base class for debug info dialect attributes.
class DebugInfoAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<DebugInfoDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// SourceNameKind
//===----------------------------------------------------------------------===//

def SourceNameKindUnknown : I64EnumAttrCase<"Unknown", 0, "unknown">;
def SourceNameKindPackage : I64EnumAttrCase<"Package", 1, "pkg">;
def SourceNameKindModule  : I64EnumAttrCase<"Module", 2, "module">;
def SourceNameKindStruct  : I64EnumAttrCase<"Struct", 3, "struct">;
def SourceNameKindFn      : I64EnumAttrCase<"Fn", 4, "fn">;

def SourceNameKind : I64EnumAttr<
    "SourceNameKind",
    "SourceName kind", [
      SourceNameKindUnknown,
      SourceNameKindPackage,
      SourceNameKindModule,
      SourceNameKindStruct,
      SourceNameKindFn,
    ]> {
  let cppNamespace = "::M::DebugInfo";
}

//===----------------------------------------------------------------------===//
// SourceNameAttr
//===----------------------------------------------------------------------===//

def SourceNameAttr : DebugInfoAttr<"SourceName", "source_name"> {
  let summary = "full source name information about a symbol";
  let description = [{
    The `#debuginfo.source_name` attribute contains preserved source name
    information about a symbol and its lineage, e.g. including parent symbols.
    It contains all information that can be used to discriminate a function: its
    base name, its argument type strings, its parameter type strings, the
    elaborated values of the parameters, and a pointer to its parent.  It also
    has the names of compiler decorators that evaluate to function references.
  }];

  let parameters = (ins
    "StringAttr":$name,
    OptionalArrayRefParameter<"SourceNameAttr">:$paramTypes,
    OptionalArrayRefParameter<"SourceNameAttr">:$argTypes,
    OptionalArrayRefParameter<"StringAttr">:$paramValues,
    OptionalParameter<"SourceNameAttr">:$parent,
    OptionalParameter<"SourceNameKind">:$kind,
    OptionalArrayRefParameter<"SourceNameAttr">:$decorators
  );

  let assemblyFormat = [{
    `<` (`(` $kind^ `)`)? `` $name
    (`[` $paramTypes^ `]`)? ``
    (`(` $argTypes^ `)`)? ``
    (`<` $paramValues^ `>`)?
    (` ` `decorators` `<` $decorators^ `>`)?
    (` ` `from` $parent^)?
    `>`
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins
        "StringAttr":$name,
        "ArrayRef<SourceNameAttr>":$paramTypes,
        "ArrayRef<SourceNameAttr>":$argTypes,
        "ArrayRef<StringAttr>":$paramValues,
        "SourceNameAttr":$parent,
        "SourceNameKind":$kind,
        "ArrayRef<SourceNameAttr>":$decorators), [{
      return $_get(name.getContext(), name, paramTypes, argTypes, paramValues,
                   parent, kind, decorators);
    }]>,
    AttrBuilderWithInferredContext<(ins "StringAttr":$name,
                                        CArg<"SourceNameKind", "{}">:$kind,
                                        CArg<"SourceNameAttr", "{}">:$parent), [{
      return get(name, {}, {}, {}, parent, kind, {});
    }]>,
    AttrBuilderWithInferredContext<(ins "const Twine &":$name,
                                        "SourceNameAttr":$parent), [{
      return get(StringAttr::get(parent.getContext(), name), {}, {}, {},
                                 parent, {}, {});
    }]>
  ];

  let extraClassDeclaration = [{
    /// Encode the attribute into a string, producing a mangled name.
    void encode(llvm::raw_ostream &os) const;
    StringAttr encode() const;

    /// Decode a source name from a mangled string.
    static ErrorOr<SourceNameAttr> decode(MLIRContext *ctx, StringRef str);
  }];

  let constBuilderCall =
    "DebugInfo::SourceNameAttr::get($_builder.getStringAttr($0))";
}

//===----------------------------------------------------------------------===//
// DIFlags
//===----------------------------------------------------------------------===//

def DIFlagsZero : I32BitEnumAttrCaseBit<"Zero", 0>;
def DIFlagsArtificial: I32BitEnumAttrCaseBit<"Artificial", 6>;

def DIFlags: I32BitEnumAttr<
    "DIFlags",
    "DI flags", [
      DIFlagsZero,
      DIFlagsArtificial,
    ]> {
  let cppNamespace = "::M::DebugInfo";
}

//===----------------------------------------------------------------------===//
// DIAttr
//===----------------------------------------------------------------------===//

// Base class for debug info dialect attributes.
class DIAttr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "DIAttr">
    : AttrDef<DebugInfoDialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

// Base class for debug info scope attributes.
class DIScopeAttr<string name, string attrMnemonic, list<Trait> traits = [],
                  string baseCppClass = "DIScopeAttr">
    : DIAttr<name, attrMnemonic, traits, baseCppClass>;

// Base class for debug info local scope attributes.
class DILocalScopeAttr<string name, string attrMnemonic, list<Trait> traits = [],
                       string baseCppClass = "DILocalScopeAttr">
    : DIAttr<name, attrMnemonic, traits, baseCppClass>;

//===----------------------------------------------------------------------===//
// DICompileUnitAttr
//===----------------------------------------------------------------------===//

def DebugInfo_DICompileUnitAttr : DIScopeAttr<"DICompileUnit", "compile_unit"> {
  let summary = "A single file compilation unit description.";
  let description = [{
    The `debuginfo.compile_unit` attribute represents a compilation scope for
    a specific file within the given source language, and contains
    configuration information for the generation of debug info for that file.

    Example:

    ```mlir
    #compile_unit = #debuginfo.compile_unit<
      sourceLanguage = DW_LANG_Mojo,
      file = #file,
      producer = "MLIR",
      isOptimized = true,
      emissionKind = Full
    >
    ```
  }];
  let parameters = (ins
    DILanguageParameter:$sourceLanguage,
    "DIFileAttr":$file,
    "StringAttr":$producer,
    "bool":$isOptimized,
    "EmissionKind":$emissionKind,
    OptionalParameter<"NameTableKind">:$nameTableKind
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
      "unsigned":$sourceLanguage, "DIFileAttr":$file, "StringRef":$producer,
      "bool":$isOptimized, "EmissionKind":$emissionKind,
      "NameTableKind":$nameTableKind
    ), [{
      MLIRContext *ctx = file.getContext();
      return $_get(ctx, sourceLanguage, file, StringAttr::get(ctx, producer),
                   isOptimized, emissionKind, nameTableKind);
    }]>
  ];
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DIFileAttr
//===----------------------------------------------------------------------===//

def DebugInfo_DIFileAttr : DIScopeAttr<"DIFile", "file"> {
  let summary = "A source file description.";
  let description = [{
    The `debuginfo.file` attribute represents a debug scope for the given
    file name and directory.

    Example:

    ```mlir
    #file = #debuginfo.file<"foo.c" in "/mlir/">
    ```
  }];
  let parameters = (ins "StringAttr":$name, "StringAttr":$directory);
  let builders = [AttrBuilder<(ins "StringRef":$name, "StringRef":$directory), [{
      return $_get($_ctxt, StringAttr::get($_ctxt, name),
                   StringAttr::get($_ctxt, directory));
    }]>
  ];

  let assemblyFormat = "`<` $name `in` $directory `>`";
}

//===----------------------------------------------------------------------===//
// DILexicalBlockAttr
//===----------------------------------------------------------------------===//

def DebugInfo_DILexicalBlockAttr : DILocalScopeAttr<
    "DILexicalBlock", "lexical_block"
  > {
  let summary = "A lexical block scope.";
  let description = [{
    The `debuginfo.lexical_block` attribute represents a lexical block debug
    scope. A lexical block describes a sequence of source statements that may
    contain any number of declarations.

    Example:

    ```mlir
    #lex_block = #debuginfo.lexical_block<
      scope = #subprogram,
      file = #file,
      line = 10,
      column = 1
    >
    ```
  }];
  let parameters = (ins
    "DILocalScopeAttr":$scope,
    "DIFileAttr":$file,
    "unsigned":$line,
    "unsigned":$column
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
      "DILocalScopeAttr":$scope, "DIFileAttr":$file, "unsigned":$line,
      "unsigned":$column
    ), [{
      return $_get(file.getContext(), scope, file, line, column);
    }]>
  ];
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DILocalVariableAttr
//===----------------------------------------------------------------------===//

def DebugInfo_DILocalVariableAttr : DIAttr<"DILocalVariable", "local_variable", [
    TypedAttrInterface
  ]> {
  let summary = "A variable or subprogram argument description.";
  let description = [{
    The `debuginfo.local_variable` attribute represents a variable or
    subprogram argument description. When representing an argument, the `arg`
    field should be 1-based index of the argument.

    Example:

    ```mlir
    #local_variable = #debuginfo.local_variable<
      scope = #lex_block,
      name = "foo",
      file = #debuginfo.file<"foo.c" in "/mlir/">,
      line = 10,
      arg = 1
    > : !debuginfo.unresolved<index>
    ```
  }];
  let parameters = (ins
    "DILocalScopeAttr":$scope,
    "StringAttr":$name,
    OptionalParameter<"DIFileAttr">:$file,
    OptionalParameter<"unsigned">:$line,
    OptionalParameter<"unsigned">:$arg,
    OptionalParameter<"unsigned", "0">:$alignInBits,
    AttributeSelfTypeParameter<"", "DIType">:$type,
    OptionalParameter<"DIFlags", "DIFlags::Zero">:$flags
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
      "DILocalScopeAttr":$scope, "StringAttr":$name, "DIFileAttr":$file,
      "unsigned":$line, "unsigned":$arg, "unsigned":$alignInBits,
      "DIType":$type, "DIFlags":$flags
    ), [{
      return $_get(type.getContext(), scope, name, file, line, arg, alignInBits,
                   type, flags);
    }]>,
    AttrBuilderWithInferredContext<(ins
      "DILocalScopeAttr":$scope, "StringRef":$name, "DIFileAttr":$file,
      "unsigned":$line, "unsigned":$arg, "unsigned":$alignInBits,
      "DIType":$type, "DIFlags":$flags
    ), [{
      MLIRContext *ctx = type.getContext();
      return $_get(ctx, scope, StringAttr::get(ctx, name), file, line,
                   arg, alignInBits, type, flags);
    }]>
  ];
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DICallLocAttr
//===----------------------------------------------------------------------===//

def DebugInfo_DICallLocAttr : DIAttr<"DICallLoc", "call_loc", []> {
  let summary = "Holds callLoc for InlinedSubprogramScoped interface";
  let description = [{
    Helper for InlinedSubprogramScoped interface, to store data in a location
    rather than the attribute dictionary (to allow CSE).

    Example:

    ```mlir
    #call_loc = #debuginfo.call_loc<#loc1>
    ```
  }];

  let parameters = (ins "LocationAttr":$callLoc);
  let assemblyFormat = "`<` $callLoc `>`";
}

//===----------------------------------------------------------------------===//
// DISubprogramAttr
//===----------------------------------------------------------------------===//

def DebugInfo_DISubprogramAttr : DILocalScopeAttr<"DISubprogram", "subprogram", [
    TypedAttrInterface
  ]> {
  let summary = "A subprogram, i.e., function, description.";
  let description = [{
    The `debuginfo.subprogram` attribute represents the debug information for
    a function.

    Example:

    ```mlir
    #subprogram = #debuginfo.subprogram<
      compileUnit = #compile_unit,
      scope = #file,
      sourceName = "foo",
      linkageName = "foo",
      file = #file,
      line = 10,
      scopeLine = 10,
      subprogramFlags = Definition
    > : !debuginfo.subroutine<() -> (): DW_CC_normal>
    ```
  }];

  let parameters = (ins
    OptionalParameter<"DICompileUnitAttr">:$compileUnit,
    OptionalParameter<"DIScopeAttr">:$scope,
    "SourceNameAttr":$sourceName,
    OptionalParameter<"StringAttr">:$linkageName,
    OptionalParameter<"DIFileAttr">:$file,
    OptionalParameter<"unsigned">:$line,
    OptionalParameter<"unsigned">:$scopeLine,
    OptionalParameter<"SubprogramFlags">:$subprogramFlags,
    AttributeSelfTypeParameter<"", "DISubroutineType">:$type
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
      "DICompileUnitAttr":$compileUnit, "DIScopeAttr":$scope,
      "SourceNameAttr":$sourceName, "StringAttr":$linkageName,
      "DIFileAttr":$file, "unsigned":$line, "unsigned":$scopeLine,
      "SubprogramFlags":$subprogramFlags, "DISubroutineType":$type
    ), [{
      return $_get(type.getContext(), compileUnit, scope, sourceName,
                   linkageName, file, line, scopeLine, subprogramFlags, type);
    }]>
  ];
  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    /// Returns a clone with a new name and linkage name.
    DISubprogramAttr cloneWith(SourceNameAttr sourceName,
                               StringAttr linkageName,
                               DISubroutineType type = {}) const;
  }];
}

#endif
