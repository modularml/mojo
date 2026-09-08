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

#ifndef KGEN_TOOLS_MOJO_COMMON_TARGETOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_TARGETOPTIONS_TD

include "Support/Driver/DriverCommand.td"

//===----------------------------------------------------------------------===//
// Target Options
//===----------------------------------------------------------------------===//

def TargetOptionGroup : OptionGroup<"Target options">, Index<6>;

defm target_triple : CommandOption<["--", "-"], "target-triple",
  TargetOptionGroup, Index<1>, MetaVarName<"TRIPLE">,
  HelpText<"Sets the compilation target triple. Defaults to the host target.">>;

defm target_cpu : CommandOption<["--", "-"], "target-cpu",
  TargetOptionGroup, Index<2>, MetaVarName<"CPU">,
  HelpText<"Sets the compilation target CPU. Defaults to the host CPU.">>;

defm target_features : CommandOption<["--", "-"], "target-features",
  TargetOptionGroup, Index<3>, MetaVarName<"FEATURES">,
  HelpText<"Sets the compilation target CPU features. Defaults to the host "
           "features.">>;

defm target_abi : CommandOption<["--", "-"], "target-abi",
  TargetOptionGroup, Index<4>, MetaVarName<"ABI">,
  HelpText<"Sets the target ABI name (e.g. `lp64d`), recorded as a "
           "`target-abi` LLVM module flag. Unset by default.">>;

defm march : CommandOption<["--", "-"], "march", TargetOptionGroup,
  Index<5>, MetaVarName<"ARCHITECTURE">,
  HelpText<"Sets the architecture for which to generate code.">>;

defm mcpu : CommandOption<["--", "-"], "mcpu", TargetOptionGroup, Index<6>,
  MetaVarName<"CPU">, HelpText<"Sets the CPU for which to generate code.">>;

defm mtune : CommandOption<["--", "-"], "mtune", TargetOptionGroup,
  Index<7>, MetaVarName<"TUNE">,
  HelpText<"Sets the CPU for which to tune code.">>;

defm target_accelerator : CommandOption<["--", "-"], "target-accelerator",
  TargetOptionGroup, Index<8>, MetaVarName<"ACCELERATOR">,
  HelpText<"Sets the GPU or accelerator architecture for heterogeneous "
           "computing (e.g., sm_90 for NVIDIA H100, gfx942 for AMD MI300).">>;

defm mcmodel: CommandOptionHidden<["--", "-"], "mcmodel",
  TargetOptionGroup, Index<9>, MetaVarName<"MCMODEL">,
  HelpText<"Sets the CodeModel for object file, i.e. small (default), medium or large.">>;

defm large_data_threshold: CommandOptionHidden<["--", "-"], "large-data-threshold",
  TargetOptionGroup, Index<10>, MetaVarName<"LDT">,
  HelpText<"Choose large data threshold for x86_64 medium code model.">>;

defm relocation_model: CommandOptionHidden<["--", "-"], "relocation-model",
  TargetOptionGroup, Index<11>, MetaVarName<"MODEL">,
  HelpText<"Sets the relocation model. Supported values: `static`, `pic` "
           "(default), `dynamic-no-pic`, `ropi`, `rwpi`, `ropi-rwpi`.">>;

//===----------------------------------------------------------------------===//
// Print Target Information Options
//===----------------------------------------------------------------------===//

def print_effective_target : Flag<["--"], "print-effective-target">,
  Group<TargetOptionGroup>, Index<12>,
  HelpText<"Print the effective target configuration after absorbing all "
           "command-line flags and exit.">;

def print_supported_targets : Flag<["--"], "print-supported-targets">,
  Group<TargetOptionGroup>, Index<13>,
  HelpText<"Print all available target names and exit.">;

def print_supported_cpus : Flag<["--"], "print-supported-cpus">,
  Group<TargetOptionGroup>, Index<14>,
  HelpText<"Print valid CPU names for the specified target and exit. "
           "Requires --target-triple.">;

def print_supported_accelerators : Flag<["--"], "print-supported-accelerators">,
  Group<TargetOptionGroup>, Index<15>,
  HelpText<"Print all supported GPU and accelerator architectures and exit.">;

#endif // KGEN_TOOLS_MOJO_COMMON_TARGETOPTIONS_TD
