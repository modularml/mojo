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

// No Values list: the accepted kinds come from the registered TargetTraits,
// which also render the per-kind table below this option in --help.
defm emitted_file_type: CommandOption<
  ["--"], "emit", OutputOptionGroup, Index<1>,
  MetaVarName<"FILE_TYPE">,
  HelpText<"The kind of output to produce instead of an executable\n"
           "(default: `exe`). Each compilation target declares the kinds\n"
           "it accepts; they are listed below with the targets that\n"
           "support them.">
>;
