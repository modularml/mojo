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
// Shared helpers for the `__mlir_op` f-string path. The user writes
// `%{name}` / `%{type_of(name)}`; the emitter rewrites those to the internal
// `%arg<N>` / `%type_of(arg<N>)` / `%param<N>` placeholders that this helper
// lowers via MLIR's parser. Used by both the Mojo IR emitter (concrete case)
// and the elaborator (deferred case).
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_MLIROPFSTRING_H
#define KGEN_TOOLCOMMON_MLIROPFSTRING_H

#include "Support/LogicalResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace M::KGEN {

/// True iff `tmpl` contains `%arg<N>`, `%type_of(arg<N>)`, or `%param<N>`.
bool isFStringTemplate(llvm::StringRef tmpl);

/// Copy `tmpl` to `out`, passing each `%` outside a `"`-string (escapes and
/// string bodies are copied verbatim) to `onPlaceholder(rest, out)` where
/// `rest` is the text past the `%`. The callback returns the count of `rest`
/// chars it consumed and wrote, 0 to copy the bare `%`, or <0 on error (stops
/// the scan; the function then returns false).
bool scanFStringTemplate(
    llvm::StringRef tmpl, llvm::raw_ostream &out,
    llvm::function_ref<int(llvm::StringRef rest, llvm::raw_ostream &out)>
        onPlaceholder);

/// Name of the `kgen.deferred` `ArrayAttr` holding `ToStringDeferredAttr`
/// entries for `%param<N>` substitutions inlined by the elaborator.
llvm::StringRef getFStringParamsAttrName();

/// Parse `tmpl` as op assembly: `operands` are wired through a synthetic
/// `func.func` entry block (so `%arg<N>` resolve) and `%type_of(arg<N>)` is
/// replaced with the operand's type text. On failure sets `errorMsg` and
/// returns null. `userTmpl`, if set, is the original `%{name}` source shown in
/// diagnostics instead of the rewritten `tmpl`.
mlir::Operation *lowerFStringMLIROp(mlir::OpBuilder &builder,
                                    mlir::Location loc, llvm::StringRef tmpl,
                                    llvm::ArrayRef<mlir::Value> operands,
                                    llvm::ArrayRef<mlir::Type> resultTypes,
                                    std::string &errorMsg,
                                    llvm::StringRef userTmpl = {});

} // namespace M::KGEN

#endif // KGEN_TOOLCOMMON_MLIROPFSTRING_H
