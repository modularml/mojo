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
// This file defines a TableGen backend that, given a command description and
// option groups, outputs Markdown that can be embedded into a static website
// generator such as Quarto.
//
//===----------------------------------------------------------------------===//

#include "GenMarkdown.h"
#include "BackendRegistry.h"
#include "DriverCommand.h"

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <optional>
#include <vector>

using namespace M;

static void genFrontMatter(raw_ostream &os, const CommandDescription &cmd) {
  os << "---\n"
     << "title: " << cmd.getName(/*join=*/" ") << "\n"
     << "description: " << cmd.getSummary() << "\n"
     << "css: /static/styles/hide-github-actions.css\n"
     << "---\n\n";
}

static void genSummarySection(raw_ostream &os, const CommandDescription &cmd) {
  os << cmd.getSummary() << "\n\n";
}

static void genSynopsisSection(raw_ostream &os, const CommandDescription &cmd) {
  os << "## Synopsis\n\n"
     << "```\n";
  for (const llvm::Record *usage : cmd.getUsages()) {
    os << cmd.getName(/*join=*/" ");
    StringRef options = usage->getValueAsString("optionsName");
    if (!options.empty())
      os << " [" << options << ']';

    StringRef input = usage->getValueAsString("inputName");
    if (!input.empty()) {
      os << " <" << input
         << (usage->getValueAsBit("variadicInput") ? "..." : "") << '>';
      if (usage->getValueAsBit("inputHasArguments"))
        os << " [" << input << "-arguments...]";
    }
    os << '\n';
  }
  os << "```\n\n";
}

static void genDescriptionSection(raw_ostream &os,
                                  const CommandDescription &cmd) {
  os << "## Description\n\n" << cmd.getDescription() << "\n\n";
}

/// If the given command has subcommands, outputs a section named "COMMANDS"
/// that lists each of them.
static void genSubcommandsSection(raw_ostream &os,
                                  const CommandDescription &cmd) {
  ArrayRef<const llvm::Record *> subcommands = cmd.getSubcommands();
  if (subcommands.empty())
    return;

  os << "## Commands\n\n";
  for (const llvm::Record *sub : subcommands) {
    StringRef name = sub->getValueAsString("subcommand");
    // TODO: Remove this "package" replacement once MOCO-3819 is fixed.
    StringRef linkStem = name == "package" ? "precompile" : name;
    os << "[`" << name << "`](" << linkStem << ".md) — "
       << sub->getValueAsString("summary") << "\n\n";
  }
}

/// Output the given LLVM `Option` record's prefix and name, followed by its
/// `MetaVarName` if present.
static void genOptionName(raw_ostream &os, const llvm::Record *option,
                          std::optional<StringRef> metaVarName) {
  os << CommandOption::getPreferredPrefix(option)
     << option->getValueAsString("Name");

  if (metaVarName) {
    if (option->getValueAsDef("Kind")->getValueAsString("Name") != "Joined")
      os << ' ';
    os << '<' << *metaVarName << '>';
  }
}

/// If there are 1 or more option groups present, outputs an "OPTIONS" section,
/// with a separate sub-section for each option group.
static void genOptionsSection(raw_ostream &os,
                              ArrayRef<CommandOptionGroup> groups) {
  if (groups.empty())
    return;

  os << "## Options\n\n";

  for (const CommandOptionGroup &group : groups) {
    // Skip any hidden option groups.
    if (group.isHidden())
      continue;

    // Print each option group, and its help text if available.
    os << "### " << group.getGroup()->getValueAsString("Name") << "\n\n";
    if (std::optional<StringRef> helpText =
            group.getGroup()->getValueAsOptionalString("HelpText"))
      os << *helpText << "\n\n";

    // Print all the options that belong to this group.
    for (const CommandOption &option : group.getOptions()) {
      // Skip any hidden options.
      if (CommandOption::isHidden(option.getOption()))
        continue;
      std::optional<StringRef> metaVarName = option.getMetaVarName();

      // Print the option's name, and then the names of its aliases.
      os << "#### `";
      genOptionName(os, option.getOption(), metaVarName);
      os << '`';
      for (const CommandAlias &option : option.getAliases()) {
        // Skip any hidden aliases.
        if (CommandOption::isHidden(option.getRecord()))
          continue;

        os << ", `";
        genOptionName(os, option.getRecord(), option.getMetaVarName());

        std::vector<StringRef> aliasArgs = option.getAliasArguments();
        if (!aliasArgs.empty()) {
          os << " (";
          if (metaVarName)
            os << *metaVarName << "=";
          llvm::interleave(aliasArgs, os, ",");
          os << ")";
        }
        os << '`';
      }
      os << "\n\n";

      // Print the main option's help text (the aliases' help text is ignored).
      // The help text may be an empty string, if the documentation writer
      // ignored mojo-tblgen warnings.
      os << option.getHelpText() << "\n\n";
    }
  }
}

static bool genHelpText(raw_ostream &os, const llvm::RecordKeeper &records) {
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

  genFrontMatter(os, cmd);
  genSummarySection(os, cmd);
  genSynopsisSection(os, cmd);
  genDescriptionSection(os, cmd);
  genSubcommandsSection(os, cmd);
  genOptionsSection(os, groups);
  return false;
}

void M::registerGenMarkdownBackend(BackendRegistry &registry) {
  registry.addBackend("gen-markdown", "Generate help text as Markdown",
                      [](raw_ostream &os, const llvm::RecordKeeper &records) {
                        return genHelpText(os, records);
                      });
}
