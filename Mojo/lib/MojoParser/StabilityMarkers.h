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
// This file declares utilities for stability markers (@stable decorator).
//
//===----------------------------------------------------------------------===//

#ifndef MOJOPARSER_STABILITYMARKERS_H
#define MOJOPARSER_STABILITYMARKERS_H

#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/CallOperands.h"
#include "Support/Compiler/Diags.h"

namespace M::KGEN::LIT {

class ASTDecl;
class SharedState;

/// Check if accessing the given declaration should emit an unstable API
/// warning. This should be called from name resolution when a symbol is
/// referenced.
///
/// A warning is emitted when all of the following conditions are met:
/// - The --warn-on-unstable-apis flag is enabled
/// - The referenced declaration is from an opted-in package (e.g., "std")
/// - The referenced declaration is NOT marked with @stable
/// - The use site is NOT in the same opted-in package (intra-package usage
///   does not warn)
///
/// \param decl The declaration being accessed
/// \param useLoc Location of the use site (for warning emission)
/// \param useSiteDecl The declaration scope at the use site (to check if
///                    we're in the same package)
/// \param shared SharedState for diagnostics and options
/// \param range (Optional) Source range to highlight in the diagnostic
void checkStabilityAndWarn(ASTDecl &decl, llvm::SMLoc useLoc,
                           ASTDecl &useSiteDecl, SharedState &shared,
                           M::SourceRange range = {});

/// Check if accessing the given declaration should emit a deprecation warning.
/// This should be called from name resolution when a symbol is referenced.
///
/// A warning is emitted when the declaration has a @deprecated decorator,
/// unless its qualified name (e.g. `Variant.take`) appears in
/// `shared.options.ignoredDeprecations` (set via `--ignore-deprecated`), in
/// which case the warning is suppressed. If the declaration has a
/// replacement identifier (from @deprecated(use=X)), a fixit will be emitted
/// for direct calls (not for operator/subscript syntax).
///
/// \param decl The declaration being accessed
/// \param useLoc Location of the use site (for warning emission)
/// \param shared SharedState for diagnostics
/// \param range Source range to highlight in the diagnostic
/// \param syntax The call syntax (used to determine if fixit should be emitted)
/// \param fixitLoc Location for fixit replacement (defaults to useLoc if
///                 invalid). For method calls, this should be the method
///                 identifier location, not the full expression location.
void checkDeprecationAndWarn(ASTDecl &decl, llvm::SMLoc useLoc,
                             SharedState &shared, M::SourceRange range,
                             CallSyntax syntax = CallSyntax::kDirectCall,
                             llvm::SMLoc fixitLoc = {});

/// Check if accessing the given declaration should emit an unavailable-API
/// error. This should be called from name resolution when a symbol is
/// referenced.
///
/// An error is emitted when the declaration has an @unavailable decorator.
/// Unlike deprecation, unavailability is always an error and is never
/// suppressed. If the declaration has a replacement identifier (from
/// @unavailable(use=X)), a fixit will be emitted for direct calls (not for
/// operator/subscript syntax).
///
/// \param decl The declaration being accessed
/// \param useLoc Location of the use site (for error emission)
/// \param shared SharedState for diagnostics
/// \param range Source range to highlight in the diagnostic
/// \param syntax The call syntax (used to determine if fixit should be emitted)
/// \param fixitLoc Location for fixit replacement (defaults to useLoc if
///                 invalid). For method calls, this should be the method
///                 identifier location, not the full expression location.
void checkUnavailableAndError(ASTDecl &decl, llvm::SMLoc useLoc,
                              SharedState &shared, M::SourceRange range,
                              CallSyntax syntax = CallSyntax::kDirectCall,
                              llvm::SMLoc fixitLoc = {});

/// Unified function to check for both deprecation and stability warnings.
/// This should be the primary entry point for callers who want to check
/// both types of warnings at once.
///
/// This function combines the checks for:
/// 1. Deprecation warnings (from @deprecated decorator)
/// 2. Stability warnings (from missing @stable decorator in opted-in packages)
///
/// \param decl The declaration being accessed
/// \param useLoc Location of the use site (for warning emission)
/// \param useSiteDecl The declaration scope at the use site
/// \param shared SharedState for diagnostics and options
/// \param range Source range to highlight in the diagnostic
/// \param syntax The call syntax (used to determine if fixit should be emitted)
/// \param fixitLoc Location for fixit replacement (defaults to useLoc if
///                 invalid). For method calls, this should be the method
///                 identifier location, not the full expression location.
void checkDeclUsageWarnings(ASTDecl &decl, llvm::SMLoc useLoc,
                            ASTDecl &useSiteDecl, SharedState &shared,
                            M::SourceRange range,
                            CallSyntax syntax = CallSyntax::kDirectCall,
                            llvm::SMLoc fixitLoc = {});

/// Check for API author error: when a stable struct implements a stable trait,
/// the implementing member (method or alias) must also be stable if the trait
/// member is stable. This is an always-on warning (does not require
/// --warn-on-unstable-apis).
///
/// The member kind ("method" or "alias") is automatically detected from the
/// trait member's operation type.
///
/// \param structDecl The struct implementing the trait
/// \param traitDecl The trait being implemented
/// \param structMemberDecl The struct's member implementing the trait member
/// \param traitMemberDecl The trait's member being implemented
/// \param shared SharedState for diagnostics
void checkStableTraitMemberImplementation(ASTDecl &structDecl,
                                          ASTDecl &traitDecl,
                                          ASTDecl &structMemberDecl,
                                          ASTDecl &traitMemberDecl,
                                          SharedState &shared);

/// Check for API author error: a stable function should return stable types.
/// This is an always-on warning (does not require --warn-on-unstable-apis).
///
/// \param funcDecl The function declaration to check
/// \param returnType The return type of the function
/// \param shared SharedState for diagnostics
void checkStableFunctionReturnType(ASTDecl &funcDecl, ASTType returnType,
                                   SharedState &shared);

/// Check for API author error: a stable trait cannot inherit from an unstable
/// trait. This is an always-on warning (does not require
/// --warn-on-unstable-apis).
///
/// \param traitDecl The trait being defined
/// \param parentTraitDecl The parent trait being inherited from
/// \param declScope The scope containing the trait declaration (used for
///                  package membership check since trait's own parent chain
///                  may not be fully established during signature resolution)
/// \param shared SharedState for diagnostics
void checkStableTraitInheritance(ASTDecl &traitDecl, ASTDecl &parentTraitDecl,
                                 ASTDecl &declScope, SharedState &shared);

/// Check for API author error: @stable member cannot be declared in an
/// unstable struct/trait. This only applies in opted-in packages where types
/// without @stable are considered unstable.
///
/// Returns true if an error was emitted (the check failed), false otherwise.
///
/// \param memberDecl The member declaration being marked @stable
/// \param decoratorLoc Location of the @stable decorator (for error emission)
/// \param shared SharedState for diagnostics
bool checkStableMemberInUnstableParent(ASTDecl &memberDecl,
                                       llvm::SMLoc decoratorLoc,
                                       SharedState &shared);

/// Check if using a magic function should emit an unstable API warning.
/// Magic functions are compiler built-ins and are considered unstable unless
/// explicitly documented as stable.
///
/// A warning is emitted when:
/// - The --warn-on-unstable-apis flag is enabled
/// - The use site is NOT in an opted-in package (e.g., "std")
///
/// The caller is responsible for checking if the magic function is stable
/// before calling this function.
///
/// \param spelling The spelling of the magic function keyword as written in
///                 source (e.g., "__get_current_function_name")
/// \param useLoc Location of the magic function keyword (for warning emission)
/// \param useSiteDecl The declaration scope at the use site (to check if
///                    we're in an opted-in package)
/// \param shared SharedState for diagnostics and options
/// \param range Source range to highlight in the diagnostic
void checkMagicFunctionAndWarn(llvm::StringRef spelling, llvm::SMLoc useLoc,
                               ASTDecl &useSiteDecl, SharedState &shared,
                               M::SourceRange range);

} // namespace M::KGEN::LIT

#endif // MOJOPARSER_STABILITYMARKERS_H
