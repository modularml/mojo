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

include "Doc/DocDescription.td"

include "Common/DiagnosticFormatOptions.td"
include "Support/Driver/CommonOptions.td"

//===----------------------------------------------------------------------===//
// Output Options
//===----------------------------------------------------------------------===//

def OutputOptionGroup : OptionGroup<"Output options">, Index<0>;

defm o : CommandOption<["-"], "o", OutputOptionGroup, Index<0>,
  MetaVarName<"PATH">,
  HelpText<"Sets the path and filename for the JSON output. If not provided, "
           "output is written to stdout.">>;

//===----------------------------------------------------------------------===//
// Compilation Options
//===----------------------------------------------------------------------===//

def CompilationOptionGroup : OptionGroup<"Compilation options">, Index<1>;

def I : JoinedOrSeparate<["-"], "I">, Group<CompilationOptionGroup>, Index<0>,
  MetaVarName<"PATH">,
  HelpText<"Appends the given path to the list of directories that Mojo will "
           "search for any package/module dependencies. That is, if the file "
           "you pass to `mojo doc` imports any packages that do not reside in "
           "the local path and are not part of the Mojo standard library, use "
           "this to specify the path where Mojo can find those packages.">;

//===----------------------------------------------------------------------===//
// Validation Options
//===----------------------------------------------------------------------===//

def ValidationOptionGroup :
  OptionGroup<"Validation options">, Index<2>,
  HelpText<
    "The following validation options help ensure that your docstrings use "
    "valid structure and meet other style criteria. By default, warnings are "
    "emitted only if the docstrings contain errors that prevent translation to "
    "the output format. (More options coming later.)">;

def diagnose_missing_doc_strings : Flag<["--", "-"],
  "diagnose-missing-doc-strings">, Group<ValidationOptionGroup>, Index<0>,
  HelpText<"Emits diagnostic warnings for missing or partial doc strings.">;

defm docs_base_path : CommandOption<["--", "-"], "docs-base-path",
  ValidationOptionGroup, Index<2>, MetaVarName<"PATH">,
  HelpText<"Sets the path prefix for generated documentation links.">>;

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

def DiagnosticOptionGroup : OptionGroup<"Compilation diagnostic options">,
  Index<3>,
  HelpText<"Controls how the Mojo compiler outputs diagnostics related to "
           "compiling and running Mojo source code.">;

defm max_notes : CommandOption<["--", "-"], "max-notes-per-diagnostic",
  DiagnosticOptionGroup, Index<3>, MetaVarName<"INTEGER">,
  HelpText<"When the Mojo compiler emits diagnostics, it sometimes also prints "
           "notes with additional information. This option sets an upper "
           "threshold on the number of notes that can be printed with a "
           "diagnostic. If not specified, the default maximum is 10.">>;

defm strip_file_prefix : CommandOptionHidden<["-"], "strip-file-prefix",
  DiagnosticOptionGroup, Index<4>, MetaVarName<"PATH">,
  HelpText<
    "Strip this prefix from filenames used for diagnostics & debugging.">>;

def werror : Flag<["--", "-"], "Werror">,
  Group<DiagnosticOptionGroup>, Index<5>,
  HelpText<"Treat warnings as errors.">;

def wno_error : Flag<["--", "-"], "Wno-error">,
  Group<DiagnosticOptionGroup>, Index<6>,
  HelpText<"Do not treat warnings as errors (overrides -Werror).">;

def validate_doc_strings : Flag<["--", "-"], "validate-doc-strings">,
  Group<DiagnosticOptionGroup>, Index<7>,
  Flags<[HelpHidden]>,
  HelpText<"Deprecated: use -Werror instead. Treat doc string warnings as errors.">;

defm ignore_deprecated : CommandOption<["--", "-"], "ignore-deprecated",
  DiagnosticOptionGroup, Index<8>, MetaVarName<"NAME">,
  HelpText<"Suppress the deprecation warning for the given declaration (e.g. "
           "`Foo.bar`, or `some_fn` for a top-level declaration).">>;
