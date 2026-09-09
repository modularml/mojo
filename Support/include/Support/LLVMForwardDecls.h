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
// This file forward declares and imports various common LLVM datatypes that we
// want to use unqualified in the modular M namespace.  This just includes types
// that are used in (not JIT compiled) runtime code, the compiler logic uses a
// superset of these types in LLVMCompilerForwardDecls.h
//
// Note that most of these are forward declared and then imported into the
// M (Modular) namespace with using decls, rather than being #included.  This is
// because we want clients to explicitly #include the files they need.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_LLVM_FORWARD_DECLS_H
#define SUPPORT_LLVM_FORWARD_DECLS_H

// MLIR includes a lot of forward declarations of LLVM types, use them.
#include "mlir/Support/LLVM.h"

// Import Error and datatype support directly.
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

// Forward declarations of LLVM classes to be imported in to the M (Modular)
// namespace.
namespace llvm {
template <typename KeyT, typename ValueT, unsigned InlineBuckets,
          typename KeyInfoT, typename BucketT>
class SmallDenseMap;
} // namespace llvm

// Forward declarations of LLVM classes we do not import into the M (Modular)
// namespace.
namespace llvm {
class Error;
template <class T>
class ErrorOr;
template <class T>
class Expected;
} // namespace llvm

// Import classes from the `llvm` and `mlir` namespace into the `M` namespace.
// All of the following classes have been already forward declared and imported
// from `llvm` in to the `mlir` namespace. For classes with default template
// arguments, MLIR does not import the type directly, it creates a templated
// using statement. This is due to the limitation that only one declaration of
// a type can have default arguments. For those types, it is important to import
// the MLIR version, and not the LLVM version. To keep things simple, all
// classes here should be imported from the `mlir` namespace, not the `llvm`
// namespace.
namespace M {
using llvm::dyn_cast_if_present;
using llvm::LogicalResult;
using llvm::SmallDenseMap;
using mlir::APFloat;
using mlir::APInt;
using mlir::APSInt;
using mlir::ArrayRef;
using mlir::cast;
using mlir::cast_or_null;
using mlir::DenseMap;
using mlir::DenseMapInfo;
using mlir::DenseSet;
using mlir::dyn_cast;
using mlir::dyn_cast_or_null;
using mlir::function_ref;
using mlir::isa;
using mlir::isa_and_nonnull;
using mlir::iterator_range;
using mlir::MutableArrayRef;
using mlir::PointerUnion;
using mlir::raw_ostream;
using mlir::SmallPtrSet;
using mlir::SmallPtrSetImpl;
using mlir::SmallString;
using mlir::SmallVector;
using mlir::SmallVectorImpl;
using mlir::StringLiteral;
using mlir::StringRef;
using mlir::StringSet;
using mlir::TinyPtrVector;
using mlir::Twine;
using mlir::TypeSwitch;

} // namespace M

#endif // SUPPORT_LLVM_FORWARD_DECLS_H
