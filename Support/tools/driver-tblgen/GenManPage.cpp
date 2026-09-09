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
// This file defines a TableGen backend that outputs a manual page, formatted
// using the roff formatting language (see `man roff` for details).
//
// Basically, the document has a title denoted by `TH`, followed by sections
// denoted by `SH` ("section header"). Within sections there may be
// sub-sections, denoted by `SS`. Special formatting can be applied to text
// within those sections by using control characters such as `\fBfoo\fR` (this
// puts "foo" in bold).
//
//===----------------------------------------------------------------------===//

#include "GenManPage.h"
#include "BackendRegistry.h"
#include "DriverCommand.h"

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

using namespace M;

/// Instantiates a new string based on the given text, but with special roff
/// characters, such as "-", escaped ("\-").
static std::string escape(const Twine &text) {
  std::string result = text.str();
  for (size_t i = 0; i < result.size(); ++i)
    if (result[i] == '-')
      result.insert(i++, 1, '\\');
  return result;
}

/// Output the title, as well as some text controls that apply to the entire
/// file.
static void genTitle(raw_ostream &os, const CommandDescription &cmd) {
  // Each `man` document starts with the `TH` macro specifying the document's
  // name and section.
  os << escape(llvm::formatv(".TH \"{0}\" \"1\"\n",
                             StringRef(cmd.getName()).upper()))
     // `nh` disables automatic hyphenation for the following text.
     << ".nh\n"
     // `ad` controls line adjustment for what follows; `l` specifies
     // left-adjusted.
     << ".ad l\n";
}

static void genNameSection(raw_ostream &os, const CommandDescription &cmd) {
  os << ".SH \"NAME\"\n"
     << escape(
            llvm::formatv("{0} \\[em] {1}\n", cmd.getName(), cmd.getSummary()));
}

static void genSynopsisSection(raw_ostream &os, const CommandDescription &cmd) {
  os << ".SH \"SYNOPSIS\"\n";
  for (const llvm::Record *usage : cmd.getUsages()) {
    os << "\\fB" << escape(cmd.getName(/*join=*/" ")) << "\\fR";
    StringRef options = usage->getValueAsString("optionsName");
    if (!options.empty())
      os << " [\\fI" << escape(options) << "\\fR]";

    StringRef input = usage->getValueAsString("inputName");
    if (!input.empty()) {
      os << " \\fI" << input
         << (usage->getValueAsBit("variadicInput") ? "..." : "") << "\\fR";
      if (usage->getValueAsBit("inputHasArguments"))
        os << " [\\fI" << input << "-arguments...\\fR]";
    }
    os << "\n.br\n";
  }
}

static void genDescriptionSection(raw_ostream &os,
                                  const CommandDescription &cmd) {
  os << ".SH \"DESCRIPTION\"\n" << escape(cmd.getDescription()) << '\n';
}

/// If the given command has subcommands, outputs a section named "COMMANDS"
/// that lists each of them.
static void genSubcommandsSection(raw_ostream &os,
                                  const CommandDescription &cmd) {
  ArrayRef<const llvm::Record *> subcommands = cmd.getSubcommands();
  if (subcommands.empty())
    return;

  os << ".SH \"COMMANDS\"\n";
  for (const llvm::Record *sub : subcommands)
    os << "\\fB" << escape(sub->getValueAsString("subcommand"))
       << "\\fR \\[em] " << escape(sub->getValueAsString("summary"))
       << "\n.br\n";
}

/// Output the given LLVM `Option` record's prefix and name, followed by its
/// `MetaVarName` if present.
static void genOptionName(raw_ostream &os, const llvm::Record *option,
                          std::optional<StringRef> metaVarName) {
  os << escape(llvm::formatv("\\fB{0}{1}\\fR",
                             CommandOption::getPreferredPrefix(option),
                             option->getValueAsString("Name")));

  if (metaVarName) {
    if (option->getValueAsDef("Kind")->getValueAsString("Name") != "Joined")
      os << ' ';
    os << "\\fI" << escape(*metaVarName) << "\\fR";
  }
}

/// If there are 1 or more option groups present, outputs an "OPTIONS" section,
/// with a separate sub-section for each option group.
static void genOptionsSection(raw_ostream &os,
                              ArrayRef<CommandOptionGroup> groups) {
  if (groups.empty())
    return;

  os << ".SH \"OPTIONS\"\n";

  for (const CommandOptionGroup &group : groups) {
    // Skip any hidden option groups.
    if (group.isHidden())
      continue;

    // Print each option group as a subsection.
    os << escape(llvm::formatv(".SS \"{0}\"\n",
                               group.getGroup()->getValueAsString("Name")));
    if (auto helpText = group.getGroup()->getValueAsOptionalString("HelpText"))
      os << escape(*helpText) << '\n';
    os << ".sp\n";

    // Populate the option group subsection with each of the options that belong
    // to that group.
    for (const CommandOption &option : group.getOptions()) {
      // Skip any hidden options.
      if (CommandOption::isHidden(option.getOption()))
        continue;

      // Print the option's name, and then the names of its aliases.
      std::optional<StringRef> metaVarName = option.getMetaVarName();
      genOptionName(os, option.getOption(), metaVarName);
      for (const CommandAlias &option : option.getAliases()) {
        // Skip any hidden aliases.
        if (CommandOption::isHidden(option.getRecord()))
          continue;

        os << ", ";
        genOptionName(os, option.getRecord(), option.getMetaVarName());

        std::vector<StringRef> aliasArgs = option.getAliasArguments();
        if (!aliasArgs.empty()) {
          os << " (";
          if (metaVarName)
            os << "\\fI" << escape(*metaVarName) << "\\fR=";
          llvm::interleave(
              aliasArgs, os,
              [&](StringRef arg) { os << "\\fI" << escape(arg) << "\\fR"; },
              ",");
          os << ")";
        }
      }
      os << '\n';

      // Print the main option's help text, indented using `RS` and `RE`.
      // (The aliases' help text is ignored.) The help text may be an empty
      // string, if the documentation writer ignored mojo-tblgen warnings.
      os << ".RS 4\n"
         << escape(option.getHelpText()) << "\n"
         << ".RE\n"
         << ".sp\n";
    }
  }
}

/// If the given description describes a subcommand, outputs a "SEE ALSO"
/// section that points to the parent executable.
static void genSeeAlsoSection(raw_ostream &os, const CommandDescription &cmd) {
  if (cmd.getSubcommand().empty())
    return;

  os << ".SH \"SEE ALSO\"\n"
     << escape(llvm::formatv("\\fB{0}\\fR(1)\n", cmd.getExecutable()));
}

static bool genManPage(raw_ostream &os, const llvm::RecordKeeper &records) {
  ErrorOr<CommandDescription> cmdOrErr = CommandDescription::get(records);
  if (failed(cmdOrErr)) {
    llvm::PrintError(cmdOrErr.getError());
    return true;
  }
  CommandDescription cmd = *cmdOrErr;

  ErrorOr<std::vector<CommandOptionGroup>> groupsOrErr =
      CommandOptionGroup::getAll(records);
  if (failed(groupsOrErr)) {
    llvm::PrintError(groupsOrErr.getError());
    return true;
  }
  std::vector<CommandOptionGroup> groups = *groupsOrErr;

  genTitle(os, cmd);
  genNameSection(os, cmd);
  genSynopsisSection(os, cmd);
  genDescriptionSection(os, cmd);
  genSubcommandsSection(os, cmd);
  genOptionsSection(os, groups);
  genSeeAlsoSection(os, cmd);
  return false;
}

void M::registerGenManPageBackend(BackendRegistry &registry) {
  registry.addBackend("gen-man-page", "Generate a man page formatted with roff",
                      [](raw_ostream &os, const llvm::RecordKeeper &records) {
                        return genManPage(os, records);
                      });
}
