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

#ifndef KGEN_MOJOPARSER_DECLSIGNATUREPRINTER_H
#define KGEN_MOJOPARSER_DECLSIGNATUREPRINTER_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

namespace M {
namespace KGEN {
namespace LIT {
class MojoInflightDiag;
class AliasDeclOp;
class TraitDeclOp;
class ASTDecl;
class FnOp;
class SharedState;
class StructDeclOp;
} // namespace LIT
} // namespace KGEN

namespace KGEN {

/// Optional output ranges to capture during signature emission. Useful for
/// downstream consumers (e.g. mojo-doc) that need to highlight or navigate to
/// individual signature components. All offsets are measured from the start of
/// the underlying string buffer being written into.
struct SignatureOffsets {
  /// Half-open `[start, end)` byte ranges for each documented parameter, in
  /// emission order.
  llvm::SmallVectorImpl<std::pair<unsigned, unsigned>> *parameters = nullptr;
  /// Half-open `[start, end)` byte ranges for each documented argument, in
  /// emission order.
  llvm::SmallVectorImpl<std::pair<unsigned, unsigned>> *arguments = nullptr;
  /// Byte offset where the return-type clause begins. Set even when there is
  /// no return type, so callers can splice in extra text (e.g. ` raises`)
  /// just before the result.
  unsigned *returnTypeStart = nullptr;
};

/// Print a Mojo-syntax signature for the given function op.
///
/// If `contextDecl` is non-null, it is installed as the current diagnostic decl
/// context for the duration of the call so dependent parameter references can
/// be rendered with their source names; otherwise dependent names may fall back
/// to index references.
void printFunctionSignature(LIT::FnOp fnOp, LIT::SharedState &shared,
                            llvm::raw_string_ostream &os,
                            const LIT::ASTDecl *contextDecl = nullptr,
                            const SignatureOffsets &offsets = {});

/// Print a Mojo-syntax signature for the given struct op.
void printStructSignature(LIT::StructDeclOp structOp, LIT::SharedState &shared,
                          llvm::raw_string_ostream &os,
                          const LIT::ASTDecl *contextDecl = nullptr,
                          const SignatureOffsets &offsets = {});

/// Print a Mojo-syntax signature for the given alias op.
///
/// No leading `comptime` keyword is emitted.
void printAliasSignature(LIT::AliasDeclOp aliasOp, LIT::SharedState &shared,
                         llvm::raw_string_ostream &os,
                         const LIT::ASTDecl *contextDecl = nullptr,
                         const SignatureOffsets &offsets = {});

/// Print a Mojo-syntax signature for the given trait op.
void printTraitSignature(LIT::TraitDeclOp traitOp, LIT::SharedState &shared,
                         llvm::raw_string_ostream &os,
                         const LIT::ASTDecl *contextDecl = nullptr,
                         const SignatureOffsets &offsets = {});

//===----------------------------------------------------------------------===//
// Diagnostic-oriented helpers
//===----------------------------------------------------------------------===//

/// Synthesize a Mojo-source-syntax signature for the given decl op. Dispatches
/// to `printFunctionSignature` / `printStructSignature` / `printAliasSignature`
/// based on the op kind. Returns an empty string if `op` is null or isn't one
/// of the supported decl ops.
///
/// Intended for diagnostics that need to identify a decl when a source-locator
/// won't reach the user (see `hasReadableSourceLocation`). Inexpensive enough
/// to call unconditionally; callers can fall back to other phrasings on an
/// empty return.
std::string synthesizeDeclSignature(Operation *op, LIT::SharedState &shared,
                                    const LIT::ASTDecl *contextDecl = nullptr);

/// True when `loc` resolves to a `FileLineColLoc` whose file is readable from
/// `sourceMgr`. False for `UnknownLoc`, locations with only `NameLoc`
/// metadata, or `FileLineColLoc`s referring to files the manager doesn't have
/// (e.g. decls loaded from a bytecode package). Walks `FusedLoc`/`NameLoc`/
/// `CallSiteLoc` chains.
///
/// Diagnostic emitters can use this to choose between attaching a normal
/// source-pointer note and synthesizing a self-describing note via
/// `synthesizeDeclSignature`.
bool hasReadableSourceLocation(Location loc, LIT::SharedState &shared);

} // namespace KGEN
} // namespace M

#endif // KGEN_MOJOPARSER_DECLSIGNATUREPRINTER_H
