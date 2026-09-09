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
// This file defines the context type used to opt in to deferring unprovable
// body constraints during IR emission instead of raising a hard error.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_DEFERREDTYPINGCONTEXT_H
#define KGEN_MOJOPARSER_DEFERREDTYPINGCONTEXT_H

#include "Mojo/LITDialect/LITAttrs.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"

namespace M::KGEN::LIT {

using llvm::SMLoc;

/// An unprovable constraint that was deferred during emission. The caller that
/// installed the context is responsible for proving (or diagnosing) these
/// constraints at a later point.
struct DeferredConstraint {
  /// The unprovable body constraint. Its own `getLoc()` points at the `where`
  /// clause that declared the constraint.
  ConstraintAttr constraint;
  /// Source location of the emission site that produced this deferred
  /// constraint (the call expression, struct-binding expression, etc.).
  SMLoc deferralLoc;
};

/// Context for deferred typing decisions during IR emission. Only supports
/// deferring unprovable constraints at this time, but designed to be extended
/// to support generalized auto-parameterization in the future.
struct DeferredTypingContext {
  /// Unprovable constraints deferred during emission.
  SmallVector<DeferredConstraint> deferredConstraints;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_DEFERREDTYPINGCONTEXT_H
