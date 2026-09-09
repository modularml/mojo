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
//
// This file declares support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_MOJODIAGS_H
#define KGEN_MOJOPARSER_MOJODIAGS_H

#include "Mojo/MojoParser/ASTType.h"
#include "Support/Compiler/Diags.h"

namespace M::KGEN::LIT {

/// This is a wrapper around MojoInflightDiag that adds some Mojo-specific
/// functionality.
class MojoInflightDiag : public InflightDiag {
public:
  struct EmittedParamInfo {
    Location loc;
    TypedAttr value;
    ASTDecl *ctxDecl;
  };

  MojoInflightDiag(InflightDiag &&diag,
                   ArrayRef<EmittedParamInfo> emittedParams)
      : InflightDiag(std::move(diag)), emittedParams(emittedParams) {}
  ~MojoInflightDiag();

  // These are all wrappers for the underlying functionality that preserves the
  // Self type.
  MojoInflightDiag(MojoInflightDiag &&other) = default;
  MojoInflightDiag &operator=(MojoInflightDiag &&other) = default;

  /// If this diagnostic is active, return the SharedState it is associated
  /// with.  This returns null if the diagnostic has been abandoned.
  SharedState *getSharedIfActive() const {
    auto *diags = getDiags();
    return !diags ? nullptr : static_cast<SharedState *>(diags->extraContext);
  }

  MojoInflightDiag attachNote(Location loc) && {
    auto params = emittedParams;
    return {std::move(*this).InflightDiag::attachNote(loc), params};
  }
  MojoInflightDiag attachNote(llvm::SMLoc loc) &&;
  MojoInflightDiag &attachNote(Location loc) & {
    InflightDiag::attachNote(loc);
    return *this;
  }
  MojoInflightDiag &attachNote(llvm::SMLoc loc) &;
  template <typename Arg>
  MojoInflightDiag &operator<<(Arg &&value) & {
    addToDiagnostic(std::forward<Arg>(value), *this);
    return *this;
  }
  template <typename Arg>
  MojoInflightDiag operator<<(Arg &&value) && {
    addToDiagnostic(std::forward<Arg>(value), *this);
    return std::move(*this);
  }

  /// Attach the ASTDecl to this diagnostic as a note. This function always
  /// attaches a new note, as long as the diagnostic is active.
  ///
  /// 1. If the declaration is a synthetic (compiler-generated) function,
  ///  pretty-print it as custom line contents at its location. The line
  ///  contains a disclaimer that the function is compiler-generated.
  /// 2. Else, if the declaration has a readable source location, return a note
  ///  at that location.
  /// 3. Else, synthesize a pretty-printed signature for the (function, struct,
  ///  alias) decl as custom line contents. The line contains a disclaimer that
  ///  the signature is synthetic.
  MojoInflightDiag &attachNote(const ASTDecl &ctxDecl) &;

  /// Attach the attribute to this diagnostic as a note. This function always
  /// attaches a new note, as long as the diagnostic is active.

  /// If the declaration has a readable source location, return a note
  /// at that location. Else pretty-print the attribute as custom line contents.
  MojoInflightDiag &attachNote(Location loc, TypedAttr attr) &;

  void addEmittedParam(TypedAttr param, std::optional<Location> loc,
                       ASTDecl *ctxDecl);

  ArrayRef<EmittedParamInfo> getEmittedParams() const { return emittedParams; }

private:
  SmallVector<EmittedParamInfo, 2> emittedParams;
};

// A wrapper around Diags that adds Mojo-specific functionality.
class MojoDiags : public Diags {
public:
  using Diags::Diags;
  MojoInflightDiag emitError(Location loc, const Twine &message) {
    return MojoInflightDiag(Diags::emitError(loc, message), {});
  }
  MojoInflightDiag emitWarning(Location loc, const Twine &message) {
    return MojoInflightDiag(Diags::emitWarning(loc, message), {});
  }
  MojoInflightDiag emitError(llvm::SMLoc loc, const Twine &message);
  MojoInflightDiag emitWarning(llvm::SMLoc loc, const Twine &message);
};

} // namespace M::KGEN::LIT

namespace M {
using KGEN::LIT::MojoInflightDiag;
void addToDiagnostic(KGEN::LIT::ASTType type, InflightDiag &diag);
void addToDiagnostic(TypedAttr paramValue, InflightDiag &diag);
void addToDiagnostic(MojoInflightDiag &&otherDiag, InflightDiag &diag);
} // namespace M

#endif // KGEN_MOJOPARSER_MOJODIAGS_H
