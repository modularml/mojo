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

#include "StabilityMarkers.h"
#include "Mojo/LITDialect/LITInterfaces.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/MojoParser/StabilityMarkers.h"
#include "Mojo/ToolCommon/CompilationOptions.h"

using namespace M::KGEN::LIT;
using namespace M::KGEN;
using namespace mlir;
using M::SourceRange;

bool M::KGEN::LIT::isPackageOptedIntoStabilityMarkers(StringRef packageName) {
  // Currently only the standard library is opted into stability markers.
  // In the future, this could be extended to support third-party packages
  // that declare their opt-in via package manifest configuration.
  //
  // "test_std_mock" is a test-only package name that allows writing tests
  // for stability markers without depending on the real standard library.
  return packageName == "std" || packageName == "test_std_mock";
}

/// Returns the name of the declaration. Asserts if the name is not available.
static StringRef getDeclName(Operation *op) {
  if (auto declItf = dyn_cast_if_present<ASTDeclInterface>(op))
    if (StringAttr name = declItf.getDeclName())
      return name.getValue();
  llvm_unreachable("declaration should have a name");
}

static StringRef getDeclName(ASTDecl &decl) {
  return getDeclName(decl.getIfOperation());
}

/// Forward declaration; defined below near `checkDeclUsageWarnings`, its
/// other use site.
static ASTDecl *getCanonicalOwnerTypeDecl(ASTDecl &decl, SharedState &shared);

/// Returns the `Owner.member` name used to match a declaration against
/// `--ignore-deprecated` (e.g. `Foo.bar`), or just the declaration's own
/// name when it has no canonical owner type (e.g. a top-level function).
static std::string getIgnoreDeprecatedName(ASTDecl &decl, SharedState &shared) {
  ASTDecl *owner = getCanonicalOwnerTypeDecl(decl, shared);
  if (owner == &decl)
    return getDeclName(decl).str();
  return getDeclName(*owner).str() + "." + getDeclName(decl).str();
}

/// Returns true if the declaration is marked @stable.
static bool hasStableDecorator(ASTDecl &decl) {
  if (auto stabilityIface = dyn_cast_if_present<StabilityDecoratorInterface>(
          decl.getIfOperation()))
    return stabilityIface.isStable();
  return false;
}

/// Find the nearest ancestor PackageOp that has opted into stability markers.
/// Walks the full ancestor chain so that e.g. code in `std::builtin::simd`
/// finds the `std` package.  Returns nullptr if no opted-in package is found.
static PackageOp findOptedInPackage(ASTDecl &decl) {
  ASTDecl *cur = decl.getNearestDeclOfType<PackageOp>();
  while (cur) {
    if (auto *op = cur->getIfOperation())
      if (auto pkgOp = dyn_cast<PackageOp>(op))
        if (isPackageOptedIntoStabilityMarkers(pkgOp.getSymName()))
          return pkgOp;
    cur = cur->getParentDecl();
    if (cur)
      cur = cur->getNearestDeclOfType<PackageOp>();
  }
  return {};
}

/// Returns true if the declaration is inside a package (or sub-package) that
/// has opted into stability markers.
static bool isDeclFromOptedInPackage(ASTDecl &decl) {
  return static_cast<bool>(findOptedInPackage(decl));
}

/// Returns true if the declaration is from an opted-in package AND marked
/// @stable.
static bool isOptedInStable(ASTDecl &decl) {
  return isDeclFromOptedInPackage(decl) && hasStableDecorator(decl);
}

/// Returns true if the declaration is from an opted-in package AND NOT marked
/// @stable.
static bool isUnstable(ASTDecl &decl) {
  return isDeclFromOptedInPackage(decl) && !hasStableDecorator(decl);
}

void M::KGEN::LIT::checkStabilityAndWarn(ASTDecl &decl, SMLoc useLoc,
                                         ASTDecl &useSiteDecl,
                                         SharedState &shared,
                                         SourceRange range) {
  // Check if the warning flag is enabled.
  if (!shared.options.warnOnUnstableAPIs)
    return;

  // Check if the declaration is unstable (from opted-in package and not marked
  // @stable).
  if (!isUnstable(decl))
    return;

  // Don't warn for intra-package usage.  Compare opted-in ancestor package
  // names so that sub-packages (e.g. std::builtin, std::collections) are
  // treated as the same package.
  auto declPkg = findOptedInPackage(decl);
  auto useSitePkg = findOptedInPackage(useSiteDecl);
  if (declPkg && useSitePkg && declPkg.getSymName() == useSitePkg.getSymName())
    return;

  // Declaration is unstable, emit warning.
  StringRef declName = getDeclName(decl);
  auto diag = shared.emitWarning(useLoc, "use of unstable API '")
              << declName << "'";
  if (range.isValid())
    diag << range;
  diag.attachNote(decl.getLoc()) << "'" << declName << "' declared here";
}

void M::KGEN::LIT::checkDeprecationAndWarn(ASTDecl &decl, SMLoc useLoc,
                                           SharedState &shared,
                                           SourceRange range, CallSyntax syntax,
                                           SMLoc fixitLoc) {
  // Check if the declaration has a @deprecated decorator.
  // Use StabilityDecoratorInterface which now handles both @stable and
  // @deprecated.
  auto stabilityItf =
      dyn_cast_if_present<StabilityDecoratorInterface>(decl.getIfOperation());
  if (!stabilityItf || !stabilityItf.isDeprecated())
    return;

  // Skip declarations explicitly suppressed via --ignore-deprecated. All
  // other deprecation warnings still fire.
  if (!shared.options.ignoredDeprecations.empty() &&
      llvm::is_contained(shared.options.ignoredDeprecations,
                         getIgnoreDeprecatedName(decl, shared)))
    return;

  StringAttr warning = stabilityItf.getDeprecationWarningAttr();
  auto diag = shared.emitWarning(useLoc, warning.getValue());
  if (range.isValid())
    diag << range;

  // Add fixit for direct and method calls (not operator/subscript syntax,
  // where replacing the magic method name wouldn't make syntactic sense).
  if (syntax == CallSyntax::kDirectCall || syntax == CallSyntax::kMethodCall) {
    if (StringAttr replacement = stabilityItf.getDeprecationReplacementAttr()) {
      // Use fixitLoc if provided, otherwise fall back to useLoc.
      SMLoc loc = fixitLoc.isValid() ? fixitLoc : useLoc;
      diag << FixIt::replaceToken(loc, replacement.getValue());
    }
  }

  StringRef declName = getDeclName(decl);
  diag.attachNote(decl.getLoc()) << "'" << declName << "' declared here";
}

void M::KGEN::LIT::checkUnavailableAndError(ASTDecl &decl, SMLoc useLoc,
                                            SharedState &shared,
                                            SourceRange range,
                                            CallSyntax syntax, SMLoc fixitLoc) {
  // Check if the declaration has an @unavailable decorator.
  auto stabilityItf =
      dyn_cast_if_present<StabilityDecoratorInterface>(decl.getIfOperation());
  if (!stabilityItf || !stabilityItf.isUnavailable())
    return;

  StringAttr reason = stabilityItf.getUnavailableReasonAttr();
  auto diag = shared.emitError(useLoc, reason.getValue());
  if (range.isValid())
    diag << range;

  // Add fixit for direct and method calls (not operator/subscript syntax,
  // where replacing the magic method name wouldn't make syntactic sense).
  //
  // The fixit is a purely syntactic rename of the callee token (e.g.
  // `old()` -> `new()`), so `use=` is intended for drop-in replacements that
  // share the same call signature. Replacements that take different arguments
  // should use a `reason` message instead of `use=`, to avoid suggesting a
  // fixit that produces code that does not compile. (Mirrors @deprecated.)
  if (syntax == CallSyntax::kDirectCall || syntax == CallSyntax::kMethodCall) {
    if (StringAttr replacement = stabilityItf.getUnavailableReplacementAttr()) {
      // Use fixitLoc if provided, otherwise fall back to useLoc.
      SMLoc loc = fixitLoc.isValid() ? fixitLoc : useLoc;
      diag << FixIt::replaceToken(loc, replacement.getValue());
    }
  }

  StringRef declName = getDeclName(decl);
  diag.attachNote(decl.getLoc()) << "'" << declName << "' declared here";
}

/// Returns the ASTDecl whose name should be checked against the use-site's
/// recursively-stable-name set:
/// - A direct struct/trait reference → &decl
/// - A method/alias declared in a struct → parent struct decl
/// - A method/alias declared in an extension → the struct the extension extends
/// - Any other decl (top-level function, alias, etc.) → &decl itself
static ASTDecl *getCanonicalOwnerTypeDecl(ASTDecl &decl, SharedState &shared) {
  // Direct struct/trait reference.
  if (isa_and_nonnull<StructDeclOp, TraitDeclOp>(decl.getIfOperation()))
    return &decl;

  ASTDecl *parent = decl.getParentDecl();
  if (!parent)
    return &decl;

  // Method/alias inside a struct or trait.
  if (isa_and_nonnull<StructDeclOp, TraitDeclOp>(parent->getIfOperation()))
    return parent;

  // Method/alias inside an extension: navigate to the extended struct.
  if (auto extOp = dyn_cast_or_null<ExtensionDeclOp>(parent->getIfOperation()))
    if (auto targetRef = extOp.getTargetStruct())
      return &shared.getDeclResolver().getDeclForTypeSymbol(*targetRef);

  // For top-level functions and other cases, check the decl itself.
  return &decl;
}

void M::KGEN::LIT::checkDeclUsageWarnings(ASTDecl &decl, SMLoc useLoc,
                                          ASTDecl &useSiteDecl,
                                          SharedState &shared,
                                          SourceRange range, CallSyntax syntax,
                                          SMLoc fixitLoc) {
  // Unavailability errors fire at every use site and are never suppressed.
  checkUnavailableAndError(decl, useLoc, shared, range, syntax, fixitLoc);

  // Deprecation warnings are never suppressed by @stable(recursive=True); the
  // only suppression mechanism is the --ignore-deprecated allowlist, applied
  // inside checkDeprecationAndWarn.
  checkDeprecationAndWarn(decl, useLoc, shared, range, syntax, fixitLoc);

  // Suppress stability warnings when the decl (or its owner type) was imported
  // with @stable(recursive=True) in the current scope.
  ASTDecl *ownerType = getCanonicalOwnerTypeDecl(decl, shared);
  if (!useSiteDecl.hasRecursivelyStableType(ownerType))
    checkStabilityAndWarn(decl, useLoc, useSiteDecl, shared, range);
}

void M::KGEN::LIT::checkStableTraitMemberImplementation(
    ASTDecl &structDecl, ASTDecl &traitDecl, ASTDecl &structMemberDecl,
    ASTDecl &traitMemberDecl, SharedState &shared) {
  // This is an API author check - it should always run regardless of the
  // --warn-on-unstable-apis flag.

  // Check if the struct, trait, and trait member are all opted-in
  // stable. Otherwise, do not check anything.
  //
  // If the trait is from a non-opted-in package,
  // do not do any check (no matter what is the struct's status).
  //
  // There is an interesting case where the trait is unsafe
  // but the struct is non-opted-in-safe. Is not possible to hit
  // this case until we have non-stdlib opted-in packages.
  // For now, consider this edge case as "do not check"

  if (!isOptedInStable(structDecl) || !isOptedInStable(traitDecl) ||
      !isOptedInStable(traitMemberDecl))
    return;

  // Check if the struct's implementing member is stable.
  if (hasStableDecorator(structMemberDecl))
    return; // All good - stable implementation.

  if (auto fnOp =
          dyn_cast_if_present<FnOp>(structMemberDecl.getIfOperation())) {
    if (fnOp.isSynthetic())
      return;
  }
  // Determine member kind from the trait member's operation type.
  StringRef memberKind =
      isa<AliasDeclOp>(traitMemberDecl.getIfOperation()) ? "alias" : "method";

  // Error: stable struct implements stable trait member with unstable member.
  StringRef memberName = getDeclName(traitMemberDecl);
  StringRef structName = getDeclName(structDecl);
  StringRef traitName = getDeclName(traitDecl);

  auto diag = shared.emitWarning(structMemberDecl.getLoc())
              << "stable struct '" << structName << "' implements stable trait "
              << memberKind << " '" << memberName
              << "' with unstable implementation";
  diag.attachNote(traitMemberDecl.getLoc())
      << "trait " << memberKind << " '" << memberName << "' in '" << traitName
      << "' is marked @stable";
}

void M::KGEN::LIT::checkStableFunctionReturnType(ASTDecl &funcDecl,
                                                 ASTType returnType,
                                                 SharedState &shared) {
  // This is an API author check - it should always run regardless of the
  // --warn-on-unstable-apis flag.

  // Check if the function is intentionally (opted-in) stable.
  // Otherwise, don't check.
  if (!isOptedInStable(funcDecl))
    return;

  // Get the underlying declaration for the return type.
  if (!returnType || !returnType.mlirType)
    return;

  // Try to get the struct/trait declaration from the return type.
  ASTDecl *returnTypeDecl = returnType.getDecl(shared);
  if (!returnTypeDecl)
    return;

  // Check if the return type is unstable.
  if (!isUnstable(*returnTypeDecl))
    return;

  // Warning: stable function returns unstable type.
  StringRef funcName = getDeclName(funcDecl);
  StringRef returnTypeName = getDeclName(*returnTypeDecl);

  auto diag = shared.emitWarning(funcDecl.getLoc())
              << "stable function '" << funcName << "' returns unstable type '"
              << returnTypeName << "'";
  diag.attachNote(returnTypeDecl->getLoc())
      << "type '" << returnTypeName << "' is not marked @stable";
}

void M::KGEN::LIT::checkStableTraitInheritance(ASTDecl &traitDecl,
                                               ASTDecl &parentTraitDecl,
                                               ASTDecl &declScope,
                                               SharedState &shared) {
  // This is an API author check - it should always run regardless of the
  // --warn-on-unstable-apis flag.

  // We use declScope for the package check because during signature resolution,
  // the trait's own parent chain may not be fully established yet.
  if (!isDeclFromOptedInPackage(declScope))
    return;

  // Check if the trait being defined is stable and parent trait is unstable.
  if (!hasStableDecorator(traitDecl) || !isUnstable(parentTraitDecl))
    return;

  // Warning: stable trait inherits from unstable trait.
  StringRef traitName = getDeclName(traitDecl);
  StringRef parentTraitName = getDeclName(parentTraitDecl);

  auto diag = shared.emitWarning(traitDecl.getLoc())
              << "stable trait '" << traitName
              << "' cannot inherit from unstable trait '" << parentTraitName
              << "'";
  diag.attachNote(parentTraitDecl.getLoc())
      << "trait '" << parentTraitName << "' is not marked @stable";
}

bool M::KGEN::LIT::checkStableMemberInUnstableParent(ASTDecl &memberDecl,
                                                     SMLoc decoratorLoc,
                                                     SharedState &shared) {
  // Check for @stable member in an unstable struct/trait.
  // Members of unstable types cannot be marked stable because the type
  // itself is not part of the stable API surface.
  //
  // This check only applies when the parent is from an opted-in package.
  // In non-opted-in packages, types without @stable are stable by default,
  // so @stable members are allowed.
  ASTDecl *parent = memberDecl.getParentDecl();
  if (!parent || !isUnstable(*parent))
    return false;

  // Determine if parent is a struct or trait for the error message.
  if (isa<StructDeclOp>(parent->getIfOperation())) {
    shared.emitWarning(
        decoratorLoc,
        "@stable member cannot be declared in an unstable struct");
    return true;
  }
  if (isa<TraitDeclOp>(parent->getIfOperation())) {
    shared.emitWarning(
        decoratorLoc, "@stable member cannot be declared in an unstable trait");
    return true;
  }

  return false;
}

void M::KGEN::LIT::checkMagicFunctionAndWarn(StringRef spelling, SMLoc useLoc,
                                             ASTDecl &useSiteDecl,
                                             SharedState &shared,
                                             SourceRange range) {
  // Check if the warning flag is enabled.
  if (!shared.options.warnOnUnstableAPIs)
    return;

  // Check if the use site is in an opted-in package (no warning for
  // intra-package usage). The std package is allowed to use magic functions
  // without warnings since it's implementing the stable API layer.
  if (isDeclFromOptedInPackage(useSiteDecl))
    return;

  // Emit warning for unstable magic function usage.
  auto diag = shared.emitWarning(useLoc, "use of unstable function '")
              << spelling << "'";
  if (range.isValid())
    diag << range;
}
