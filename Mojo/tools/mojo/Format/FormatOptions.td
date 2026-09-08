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

include "Format/FormatDescription.td"
include "Support/Driver/CommonOptions.td"

//===----------------------------------------------------------------------===//
// Format Options
//===----------------------------------------------------------------------===//

def FormatOptionGroup : OptionGroup<"Format options">, Index<0>;

defm line_length : CommandOption<["--", "-"], "line-length",
  FormatOptionGroup, Index<0>, MetaVarName<"INTEGER">,
  HelpText<"Sets the max character line length. Default is 80.">>;
def : Separate<["-"], "l">, Group<FormatOptionGroup>, Index<1>,
  MetaVarName<"INTEGER">, Alias<line_length>;

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

def DiagnosticOptionGroup : OptionGroup<"Diagnostic options">, Index<1>;

def quiet : Flag<["--"], "quiet">, Group<DiagnosticOptionGroup>,
  Index<0>, HelpText<"Disables non-error messages.">;
def : Flag<["-"], "q">, Group<DiagnosticOptionGroup>, Index<0>, Alias<quiet>;

def print_cache_dir : Flag<["--"], "print-cache-dir">,
  Group<DiagnosticOptionGroup>, Index<1>,
  Flags<[HelpHidden]>,
  HelpText<"Print the cache directory path and exit.">;
