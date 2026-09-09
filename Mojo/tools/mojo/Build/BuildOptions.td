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

include "BuildDescription.td"

include "Common/CompilationOptions.td"
include "Common/DiagnosticFormatOptions.td"
include "Common/DiagnosticOptions.td"
include "Common/TargetOptions.td"
include "Support/Driver/CommonOptions.td"

//===----------------------------------------------------------------------===//
// Output Options
//===----------------------------------------------------------------------===//

def OutputOptionGroup : OptionGroup<"Output options">, Index<0>;

defm o : CommandOption<["-"], "o", OutputOptionGroup, Index<0>,
  MetaVarName<"PATH">, HelpText<
    "Sets the path and filename for the executable output. By default, it "
    "outputs the executable to the current directory, with the same name "
    "and no extension."
  >>;

defm emitted_file_type: CommandOptionWithValues<
  ["--"], "emit", OutputOptionGroup, Index<1>,
  MetaVarName<"FILE_TYPE">,
  HelpText<"The type of output file to generate."
            "\n\n"
            "* `exe` (default): emit an executable binary file.\n"
            "* `shared-lib`: emit a shared (dynamic) library.\n"
            "* `object`: (EXPERIMENTAL) emit a single object file.\n"
            "* `llvm`: emit unoptimized LLVM IR.\n"
            "* `llvm-bitcode`: emit bitcode of unoptimized LLVM IR.\n"
            "* `asm`: emit target assembly. For GPU targets, also emits a\n"
            "  sidecar file per kernel alongside the host assembly: `.ptx`\n"
            "  for NVIDIA, `.amdgcn` for AMD, `.ll` for Metal.">,
  Values<"exe,shared-lib,object,llvm,llvm-bitcode,asm">
>;
