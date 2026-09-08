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
// This file defines a common set of compilation options for setting up the
// compilation environment for Mojo programs. The index of the option groups is
// set to multiples of 5, to allow other option groups to be inserted before or
// after each.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_COMPILATIONOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_COMPILATIONOPTIONS_TD

include "Support/Driver/DriverCommand.td"

//===----------------------------------------------------------------------===//
// Compilation Options
//===----------------------------------------------------------------------===//

def CompilationOptionGroup : OptionGroup<"Compilation options">, Index<5>;

defm optimization_level : CommandOption<["--", "-"], "optimization-level",
  CompilationOptionGroup, Index<0>, MetaVarName<"LEVEL">,
  HelpText<
    "Sets the level of optimization to use at compilation. The value must be "
    "a number between 0 and 3. The default is 3."
  >>;
def : JoinedOrSeparate<["-"], "O">, Group<CompilationOptionGroup>,
  Index<1>, Alias<optimization_level>;
def : Flag<["--", "-"], "no-optimization">, Group<CompilationOptionGroup>,
  Index<2>, Alias<optimization_level>, AliasArgs<["0"]>;

defm fp_mode : CommandOption<["--", "-"], "fp-mode",
  CompilationOptionGroup, Index<17>, MetaVarName<"MODE">,
  HelpText<
    "Controls floating-point behavior as a comma-separated list of "
    "`feature=value` items (may be given more than once). The only feature is "
    "`contract`, one of `fast` (default) or `off`. `contract=fast` is like "
    "Clang's `-ffp-contract=fast`: it fuses `a + b*c` into an FMA across "
    "statements and breaking strict IEEE compliance. "
    "`contract=off` disables contraction."
  >>;

def I : JoinedOrSeparate<["-"], "I">, Group<CompilationOptionGroup>, Index<3>,
  MetaVarName<"PATH">,
  HelpText<"Appends the given path to the list of directories to search for "
           "imported Mojo files.">;

def D : JoinedOrSeparate<["-"], "D">, Group<CompilationOptionGroup>, Index<4>,
  MetaVarName<"KEY=VALUE">,
  HelpText<"Defines a named value that can be used from within the Mojo source "
           "file being executed. For example, `-Dfoo=42` defines a name `foo` "
           "that, when queried with the `std.defines` module from within the "
           "Mojo program, would yield the compile-time value `42`.">;

defm debug_level : CommandOption<["--", "-"], "debug-level",
  CompilationOptionGroup, Index<5>, MetaVarName<"LEVEL">,
  HelpText<
    "Sets the level of debug info to use at compilation. The value must be one "
    "of: `none`, `line-tables`, or `full`. Default is `none`, except when "
    "using `mojo debug foo.mojo`, which defaults to `full`. Please note that "
    "there are issues when generating debug info for some Mojo programs that "
    "have yet to be addressed."
  >>;

def : Flag<["-"], "g">, Group<CompilationOptionGroup>, Index<6>,
  Alias<debug_level>, AliasArgs<["full"]>;

def : Flag<["-"], "g0">, Group<CompilationOptionGroup>, Index<7>,
  Alias<debug_level>, AliasArgs<["none"]>;

def : Flag<["-"], "g1">, Group<CompilationOptionGroup>, Index<8>,
  Alias<debug_level>, AliasArgs<["line-tables"]>;

def : Flag<["-"], "g2">, Group<CompilationOptionGroup>, Index<9>,
  Alias<debug_level>, AliasArgs<["full"]>;

defm num_threads : CommandOption<["--", "-"], "num-threads",
  CompilationOptionGroup, Index<10>, MetaVarName<"NUM">,
  HelpText<
    "Sets the maximum number of threads to use for compilation. "
    "The default is 0 (use all available threads)."
  >>;

def : JoinedOrSeparate<["-"], "j">, Group<CompilationOptionGroup>,
  Index<11>, Alias<num_threads>;

defm loop_unrolling_warn_threshold: CommandOptionHidden<["--", "-"],
  "loop-unrolling-warn-threshold",
  CompilationOptionGroup, Index<12>, MetaVarName<"LUWT">,
  HelpText<"Sets loop unrolling factor warning threshold, "
           "default is 1024, set to 0 to disable the warning.">>;

defm elaboration_error_limit: CommandOptionHidden<["--", "-"],
  "elaboration-error-limit",
  CompilationOptionGroup, Index<13>, MetaVarName<"ELABERRORLIMIT">,
  HelpText<
    "Sets limit to number of produced errors during elaboration "
    "before stop emitting diagnostics. The default is 20, and "
    "the limit can be disabled with -elaboration-error-limit=0.">>;

def elaboration_error_include_prelude: Flag<["--", "-"],
  "elaboration-error-include-prelude">,
  Group<CompilationOptionGroup>, Index<14>,
  HelpText<
    "Show elaboration error with locations in mojo startup modules "
    "(prelude).">;

defm elaboration_error_verbose: CommandOptionHidden<["--", "-"],
  "elaboration-error-verbose",
  CompilationOptionGroup, Index<15>, MetaVarName<"ELABERRORVERBOSE">,
  HelpText<
    "Controls how parameter values are displayed as part of "
    "elaboration errors when function instantiation fails. "
    "The default is `simple-params`, which "
    "display concretized parameter values for "
    "simple types, including numeric types and "
    "strings, in a user-friendly format; "
    "`no-params` means don't display any concrete parameter values; "
    "`all-params` means show all concrete parameter values.">>;

defm elaboration_max_depth: CommandOptionHidden<["--", "-"],
  "elaboration-max-depth",
  CompilationOptionGroup, Index<16>, MetaVarName<"ELABMAXDEPTH">,
  HelpText<
      "The maximum elaborator instantiation depth. Default is "
      "std::numeric_limits<unsigned>::max(). Elaborator errors out if "
      "the depth of compile time recursion exceeds this value.">>;

def mlir_timing: Flag<["--", "-"], "mlir-timing">,
  Group<CompilationOptionGroup>, Index<18>, Flags<[HelpHidden]>,
  HelpText<
    "Times each MLIR pass and analysis, printing a report to stderr when "
    "compilation finishes, which for `mojo run` is before the program starts. "
    "Passes that the compilation cache serves from an earlier build never "
    "run, so a warm cache reports the parse and nothing else; point "
    "`MODULAR_CACHE_DIR` at an empty directory to time a whole pipeline.">;

def llvm_timing: Flag<["--", "-"], "llvm-timing">,
  Group<CompilationOptionGroup>, Index<20>, Flags<[HelpHidden]>,
  HelpText<
    "Measures the time of each LLVM pass and analysis. Writes the LLVM report "
    "to stderr when the compilation is complete. The timers of LLVM are "
    "global to the process and are not safe for more than one thread. "
    "Therefore this option makes the compiler use one thread, and it "
    "overrides `--num-threads`. The report then shows the time that LLVM "
    "uses, but it does not show the cost of a parallel build. LLVM does not "
    "get the object code that the compilation cache keeps from an earlier "
    "build. Therefore the report is empty if the cache has the object code. "
    "Set `MODULAR_CACHE_DIR` to an empty directory to measure a full "
    "pipeline. The report holds the passes of the host and the passes of "
    "each `compile_offload` target together.">;

defm mlir_timing_display: CommandOptionHidden<["--", "-"],
  "mlir-timing-display",
  CompilationOptionGroup, Index<19>, MetaVarName<"MODE">,
  HelpText<
    "How `--mlir-timing` groups its report. `tree` (the default) nests "
    "timings by pipeline structure; `list` aggregates per pass name, sorted "
    "by total time. The `list` mode merges the passes of all pipelines, so it "
    "cannot show the time of one offload target on its own.">>;

//===----------------------------------------------------------------------===//
// Linker Options
//===----------------------------------------------------------------------===//

def LinkOptionGroup : OptionGroup<"Linker options">, Index<15>;

defm Xlinker : CommandOption<["-"], "Xlinker",
  LinkOptionGroup, Index<1>, MetaVarName<"ARG">,
  HelpText<"Pass ARG to the linker.">>;

defm lld_path : CommandOption<["--", "-"], "lld-path",
  LinkOptionGroup, Index<2>, MetaVarName<"PATH">,
  HelpText<"Overrides the path to the `lld` linker used when linking. Takes "
           "precedence over the `MODULAR_MOJO_MAX_LLD_PATH` environment "
           "variable and the `mojo-max.lld_path` configuration value.">>;

//===----------------------------------------------------------------------===//
// Experimental Compilation Options
//===----------------------------------------------------------------------===//

def ExperimentalCompilationOptionGroup :
  OptionGroup<"Experimental compilation options">, Index<16>;

defm sanitize : CommandOption<["--", "-"], "sanitize",
  ExperimentalCompilationOptionGroup, Index<1>, MetaVarName<"CHECK">,
  HelpText<
    "Turns on runtime checks. The following values are supported: `address` "
    "(detects memory issues), and `thread` (detects multi-threading issues)."
  >>;

def shared_libasan : Flag<["--", "-"], "shared-libasan">,
  Group<ExperimentalCompilationOptionGroup>, Index<2>,
  HelpText<
    "Dynamically link the address sanitizer runtime. Requires address "
    "sanitization turned on with `--sanitize` option."
  >;

defm external_libasan : CommandOptionHidden<["--", "-"], "external-libasan",
  ExperimentalCompilationOptionGroup, Index<3>, MetaVarName<"FILE">,
  HelpText<
    "Uses the provided file as libasan instead of the default provided by "
    "the system. This file must provide a compatible interface as libasan, "
    "and must be present at both compile time and run time of the program."
  >>;

defm bitcode_libs : CommandOptionHidden<["--", "-"], "bitcode-libs",
  ExperimentalCompilationOptionGroup, Index<4>, MetaVarName<"LIBS">,
  HelpText<
    "External bitcode libraries to link into the generated LLVM IR."
  >>;

defm debug_info_language : CommandOption<["--", "-"], "debug-info-language",
  ExperimentalCompilationOptionGroup, Index<5>, MetaVarName<"LANGUAGE">,
  HelpText<
    "Sets the language to emit as part of the debug info. The supported "
    "languages are: `Mojo`, and `C`. `Mojo` is the default. `C` is useful to "
    "enable rudimentary debugging and binary introspection in tools "
    "that don't understand Mojo, but is not required for `mojo debug`."
  >>;

#endif // KGEN_TOOLS_MOJO_COMMON_COMPILATIONOPTIONS_TD
