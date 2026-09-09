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
// option groups, outputs a raw C++ string literal that can be used as help
// text.
//
//===----------------------------------------------------------------------===//

#include "GenHelpText.h"
#include "BackendRegistry.h"
#include "DriverCommand.h"

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <optional>
#include <string>
#include <sys/types.h>
#include <tuple>
#include <vector>

using namespace M;

/// Given a string, splits it into two substrings around the first occurrence of
/// a space or newline character, along with a boolean indicating whether the
/// character was a newline.
static std::tuple<StringRef, StringRef, bool> splitString(StringRef str) {
  size_t idx = str.find_if([](char c) { return c == ' ' || c == '\n'; });
  if (idx == StringRef::npos)
    return std::make_tuple(str, StringRef(), false);
  return std::make_tuple(str.slice(0, idx), str.slice(idx + 1, StringRef::npos),
                         str[idx] == '\n');
}

/// Write the given `text` to the output stream `os`, inserting a line break if
/// a "word" would exceed the column `limit` (this is a simple function that,
/// for now, treats spaces and newlines as word delimiters. It will need to be
/// updated if we wish to better support splitting `inline code` or
/// hyphenated-text). Each new line is indented by `indent`.
static raw_ostream &writeWordWrapped(raw_ostream &os, const Twine &text,
                                     size_t indent = 0, size_t limit = 78) {
  ssize_t maxLineLength = limit - indent;
  assert(maxLineLength > 0 && "indent must not exceed line length limit");

  SmallVector<char> buffer;
  StringRef str = text.toStringRef(buffer);

  // The number of characters that can still fit on this line.
  ssize_t remainingLength = maxLineLength;
  // Whether any text we are about to print appears on a new line.
  bool startsNewline = true;

  while (!str.empty()) {
    auto [word, rest, isSplitOnNewline] = splitString(str);
    // If we're on a new line, indent before printing any characters.
    if (startsNewline)
      os.indent(indent);

    // If the word can't fit on this line, then start a new line and try again.
    // A word that follows another word is preceded by a space, so that space
    // counts against the line here, where the decision is made.
    ssize_t size = word.size() + !startsNewline;
    if (size > remainingLength) {
      os << '\n';
      // If the word can't fit on *any* line, just print it on its own line.
      // Alone on a line, no space precedes it, so measure the word itself.
      if (ssize_t(word.size()) > maxLineLength) {
        os.indent(indent) << word << '\n';
        str = rest;
      }
      remainingLength = maxLineLength;
      startsNewline = true;
      continue;
    }

    // The word can fit on this line.
    if (!startsNewline) {
      // If the word is following another word, print a space.
      os << ' ';
      --remainingLength;
    }

    os << word;

    // If a newline character separates this word and the next, print it and
    // reset the line length. Otherwise, subtract the word we printed from the
    // line length.
    if (isSplitOnNewline) {
      os << '\n';
      remainingLength = maxLineLength;
      startsNewline = true;
    } else {
      remainingLength -= word.size();
      startsNewline = false;
    }
    // Move on to the rest of the string.
    str = rest;
  }

  return os;
}

static void genNameSection(raw_ostream &os, const CommandDescription &cmd) {
  os << "NAME\n";
  std::string formatted =
      llvm::formatv("{0} — {1}", cmd.getName(), cmd.getSummary());
  writeWordWrapped(os, formatted, /*indent=*/8) << "\n\n";
}

static void genSynopsisSection(raw_ostream &os, const CommandDescription &cmd) {
  os << "SYNOPSIS\n";
  for (const llvm::Record *usage : cmd.getUsages()) {
    os.indent(8) << cmd.getName(/*join=*/" ");
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
  os << '\n';
}

static void genDescriptionSection(raw_ostream &os,
                                  const CommandDescription &cmd) {
  os << "DESCRIPTION\n";
  writeWordWrapped(os, cmd.getDescription(), /*indent=*/8);
  os << "\n\n";
}

/// If the given command has subcommands, outputs a section named "COMMANDS"
/// that lists each of them, in 2 columns: the name of the subcommand, and its
/// summary. The summaries are vertically aligned.
static void genSubcommandsSection(raw_ostream &os,
                                  const CommandDescription &cmd) {
  ArrayRef<const llvm::Record *> subcommands = cmd.getSubcommands();
  if (subcommands.empty())
    return;

  size_t maxSize = 0;
  for (const llvm::Record *sub : subcommands)
    maxSize = std::max(maxSize, sub->getValueAsString("subcommand").size());

  os << "COMMANDS\n";
  for (const llvm::Record *sub : subcommands) {
    StringRef name = sub->getValueAsString("subcommand");
    os.indent(8) << name;
    os.indent(maxSize - name.size());
    os << " — " << sub->getValueAsString("summary") << '\n';
  }
  os << '\n';
}

/// Output the given LLVM `Option` record's prefix and name, followed by its
/// `MetaVarName` if present.
static void genOptionName(raw_ostream &os, const llvm::Record *option,
                          std::optional<StringRef> metaVarName,
                          size_t indent = 0) {
  os.indent(indent) << CommandOption::getPreferredPrefix(option)
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
                              ArrayRef<CommandOptionGroup> groups,
                              bool includeHiddenOptions) {
  if (groups.empty())
    return;

  os << "OPTIONS\n";

  for (const CommandOptionGroup &group : groups) {
    // Skip any hidden option groups.
    if (group.isHidden() && !includeHiddenOptions)
      continue;

    // Print each option group, and its help text if available.
    os.indent(4) << group.getGroup()->getValueAsString("Name") << '\n';
    if (std::optional<StringRef> helpText =
            group.getGroup()->getValueAsOptionalString("HelpText"))
      writeWordWrapped(os, *helpText, /*indent=*/8) << "\n\n";

    // Print all the options that belong to this group.
    for (const CommandOption &option : group.getOptions()) {
      // Skip any hidden options.
      if (CommandOption::isHidden(option.getOption()) && !includeHiddenOptions)
        continue;

      // Print the option's name, and then the names of its aliases.
      std::optional<StringRef> metaVarName = option.getMetaVarName();
      genOptionName(os, option.getOption(), metaVarName, /*indent=*/8);
      for (const CommandAlias &option : option.getAliases()) {
        // Skip any hidden aliases.
        if (CommandOption::isHidden(option.getRecord()) &&
            !includeHiddenOptions)
          continue;

        os << ", ";
        genOptionName(os, option.getRecord(), option.getMetaVarName());

        std::vector<StringRef> aliasArgs = option.getAliasArguments();
        if (!aliasArgs.empty()) {
          os << " (";
          if (metaVarName)
            os << *metaVarName << "=";
          llvm::interleave(aliasArgs, os, ",");
          os << ")";
        }
      }
      os << '\n';

      // Print the main option's help text (the aliases' help text is ignored).
      // The help text may be an empty string, if the documentation writer
      // ignored mojo-tblgen warnings.
      writeWordWrapped(os, option.getHelpText(), /*indent=*/12) << "\n\n";
    }
  }
}

static bool genHelpText(raw_ostream &os, const llvm::RecordKeeper &records,
                        bool includeHiddenOptions) {
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

  os << "u8R\"(";
  genNameSection(os, cmd);
  genSynopsisSection(os, cmd);
  genDescriptionSection(os, cmd);
  genSubcommandsSection(os, cmd);
  genOptionsSection(os, groups, includeHiddenOptions);
  os << ")\"";
  return false;
}

void M::registerGenHelpTextBackend(BackendRegistry &registry) {
  registry.addBackend("gen-help-text",
                      "Generate help text as a C++ constant string",
                      [](raw_ostream &os, const llvm::RecordKeeper &records) {
                        return genHelpText(os, records, false);
                      });
  registry.addBackend(
      "gen-help-hidden-text",
      "Generate help text with hidden options as a C++ constant string",
      [](raw_ostream &os, const llvm::RecordKeeper &records) {
        return genHelpText(os, records, true);
      });
}
