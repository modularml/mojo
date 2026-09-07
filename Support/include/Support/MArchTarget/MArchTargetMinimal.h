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
// This file contains MArchTarget declarations that are meant to be accessible
// by the static KGENCompilerRT library that links with user object files.
// Its dependencies are intentionally kept minimal to reduce the size of the
// user binary. Any code that adds additional dependencies should be in
// `MArchTarget.h` instead.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MARCHTARGET_MARCHTARGETMINIMAL_H
#define SUPPORT_MARCHTARGET_MARCHTARGETMINIMAL_H

#include "Support/DeviceSpecs.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include <memory>
#include <string>
#include <vector>

namespace llvm {
// Forward declare.
class TargetMachine;
} // namespace llvm

namespace clang {
class TargetOptions;
}

namespace M {

/// Returns the feature set according to clang for the given options, which
/// should include the Triple and CPU.
ErrorOr<std::vector<std::string>>
getFeaturesFromClang(std::shared_ptr<clang::TargetOptions> opts, StringRef cpu);

/// Returns a TargetInfo describing the host.
ErrorOr<TargetInfo> getHostTargetInfo();

/// Returns the features for the host as a signed string (e.g.
/// "+avx2,+bmi1,-avx512f") ready to pass to LLVM.
std::string getHostCPUFeatures();

/// Returns a TargetInfo describing the consequences of the given `-march`,
/// `-mcpu` and `-mtune` settings. These flags have target-dependent behaviour
/// as described in https://gcc.gnu.org/onlinedocs/gcc/. Note that the `-mtune`
/// flag is not captured in the result.
///
/// This method will construct a minimum target triple and feature set using the
/// provided architecture and CPU. Both are optional.
///
/// `-march=native` will use all the features of the host system.
///
/// For X86 architectures, `-march` or `-mcpu` can be used to specify a CPU
/// subtype, like `skylake-avx512`. If `-mcpu=generic`, then `-march` is assumed
/// to be an X86 architecture kind and a generic CPU for that is used.
///
/// For ARM architectures, `-march` specifies the base architecture or `-mcpu`
/// specifies the specific CPU kind. If only an architecture is specified, the
/// default CPU for it is used.
///
/// For AArch64 architectures, `-march` specifies the base architecture or
/// `-mcpu` specifies the specific CPU kind. If only an architecture is
/// specified, `-mcpu=generic` will be used.
///
/// `-mtune` will specify the CPU to specifically tune code for.
ErrorOr<TargetInfo> getMArchTargetInfo(StringRef targetTriple, StringRef march,
                                       StringRef mcpu, StringRef mtune);

/// Returns the CPU features for a given target triple and CPU.
///
/// A CPU the target does not accept is an error, which is the answer a name the
/// user picked deserves: substituting the baseline would compile something
/// other than what was asked for. Use `resolveCpu` for a name that was detected
/// rather than chosen.
ErrorOr<std::vector<std::string>> getFeatures(StringRef triple, StringRef cpu);

/// A CPU name the target accepts, and the features it implies.
struct ResolvedCpu {
  /// Empty when the requested CPU was rejected and the triple's baseline was
  /// used instead.
  std::string name;
  std::vector<std::string> features;
};

/// Returns a CPU the target accepts for `triple`, falling back to the triple's
/// baseline when it does not accept `cpu`.
///
/// For a CPU name that was detected rather than chosen: `getHostCPUName()`
/// answers "generic" for a part LLVM cannot identify, and that name is not
/// universally valid, so an unidentified host would otherwise fail target
/// creation outright rather than compile for the architecture baseline.
/// Returning the name with the features keeps the two in step -- a caller
/// cannot record a CPU that the features did not come from.
ErrorOr<ResolvedCpu> resolveCpu(StringRef triple, StringRef cpu);

} // namespace M

#endif // SUPPORT_MARCHTARGET_MARCHTARGETMINIMAL_H
