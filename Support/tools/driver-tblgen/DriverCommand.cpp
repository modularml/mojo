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

#include "DriverCommand.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SMLoc.h"
#include <cassert>
#include <cctype>
#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

using namespace M;

/// Prints the given `message` as an error at the given `locations` and returns
/// `true`.
static bool printError(ArrayRef<llvm::SMLoc> locations, const Twine &message) {
  llvm::PrintError(locations, message);
  return true;
}

/// Prints an error and returns a failure if the given text is empty, or if it
/// does not begin with a lowercase character. Otherwise, prints nothing and
/// returns success.
static LogicalResult validateCapitalized(StringRef text,
                                         ArrayRef<llvm::SMLoc> locs,
                                         const Twine &description) {
  if (text.empty())
    return success(
        /*isSuccess=*/!printError(locs, description + " should not be empty"));

  if (!isupper(text.front()))
    return success(/*isSuccess=*/!printError(
        locs, description + " should begin with a capital letter"));

  return success();
}

/// Given a TableGen record, return its 'index' integer value, if one is
/// defined.
static std::optional<int64_t>
getValueAsOptionalIndex(const llvm::Record *record) {
  const llvm::RecordVal *val = record->getValue("index");
  if (!val)
    return {};

  const llvm::IntInit *i =
      dyn_cast_if_present<const llvm::IntInit>(val->getValue());
  if (!i)
    return {};

  return i->getValue();
}

/// Given an LLVMOption TableGen record, return whether "HelpHidden" appears in
/// its "Flags" list value. "Flags" must be defined on this record, as it is for
/// `OptionGroup` and `Option` records.
static bool isRecordHidden(const llvm::Record *record) {
  return llvm::any_of(
      record->getValueAsListOfDefs("Flags"),
      [](const llvm::Record *flag) { return flag->getName() == "HelpHidden"; });
}

//===----------------------------------------------------------------------===//
// CommandAlias
//===----------------------------------------------------------------------===//

std::vector<StringRef> CommandAlias::getAliasArguments() const {
  const llvm::RecordVal *aliasArgsRecord = record->getValue("AliasArgs");
  if (!aliasArgsRecord || !aliasArgsRecord->getValue())
    return {};
  return record->getValueAsListOfStrings("AliasArgs");
}

//===----------------------------------------------------------------------===//
// CommandDescription
//===----------------------------------------------------------------------===//

ErrorOr<CommandDescription>
M::CommandDescription::get(const llvm::RecordKeeper &records) {
  ArrayRef<const llvm::Record *> descriptions =
      records.getAllDerivedDefinitions("CommandDescription");
  if (descriptions.empty())
    return Error("you must define a 'CommandDescription' record");

  const llvm::Record *topLevel = nullptr;
  std::vector<const llvm::Record *> subcommands;
  for (const llvm::Record *record : descriptions) {
    if (record->getValueAsString("subcommand").empty()) {
      if (topLevel) {
        printError(
            record->getLoc(),
            "a second top-level 'CommandDescription' record is defined here");
        llvm::PrintNote(topLevel->getLoc(),
                        "a top-level 'CommandDescription' record has "
                        "already been defined here");
        return Error(
            "cannot define a second top-level 'CommandDescription' record");
      }
      topLevel = record;
    } else {
      subcommands.push_back(record);
    }
  }
  if (!topLevel && subcommands.size() > 1) {
    printError(subcommands.back()->getLoc(),
               "a second subcommand 'CommandDescription' is defined here");
    llvm::PrintNote(
        subcommands.front()->getLoc(),
        "a subcommand 'CommandDescription' has already been defined here");
    return Error("cannot define a second subcommand 'CommandDescription'");
  }

  const llvm::Record *record = nullptr;
  if (topLevel) {
    record = topLevel;
  } else {
    record = subcommands.back();
    subcommands.clear();
  }

  // Now that we have our description and its subcommands, perform some
  // validation.
  auto validateDescription = [](const llvm::Record *record) -> LogicalResult {
    bool invalid = false;
    if (record->getValueAsString("executable").empty())
      invalid |= printError(record->getLoc(),
                            "command executable name should not be empty");

    StringRef summary = record->getValueAsString("summary");
    invalid |= failed(
        validateCapitalized(summary, record->getLoc(), "command summary"));
    if (!summary.ends_with("."))
      invalid |= printError(record->getLoc(),
                            "command summary should end with a period");

    StringRef description = record->getValueAsString("description");
    invalid |= failed(validateCapitalized(description, record->getLoc(),
                                          "command description"));
    if (!description.ends_with("."))
      invalid |= printError(record->getLoc(),
                            "command description should end with a period");

    std::vector<const llvm::Record *> usages =
        record->getValueAsListOfDefs("usages");
    if (usages.empty())
      invalid |= printError(record->getLoc(),
                            "command should include at least one usage");

    for (const llvm::Record *usage : usages) {
      StringRef optionsName = usage->getValueAsString("optionsName");
      if (optionsName.lower() != optionsName)
        invalid |= printError(usage->getLoc(),
                              "usage options name should be lowercase");

      StringRef metaVarName = usage->getValueAsString("inputName");
      if (metaVarName.lower() != metaVarName)
        invalid |= printError(usage->getLoc(),
                              "usage input metavar name should be lowercase");
    }

    return success(/*isSuccess=*/!invalid);
  };

  // First validate the main description.
  bool invalid = failed(validateDescription(record));

  // Then, validate any subcommands it may have.
  DenseMap<int64_t, const llvm::Record *> indices;
  for (const llvm::Record *sub : subcommands) {
    invalid |= failed(validateDescription(sub));

    // In addition to the standard validations applied to all descriptions,
    // subcommands must also be ordered by index.
    std::optional<int64_t> index = getValueAsOptionalIndex(sub);
    if (!index) {
      invalid |= printError(
          sub->getLoc(),
          llvm::formatv(
              "subcommand '{0}' has no index with which to order it by; "
              "it will appear in a non-deterministic order",
              sub->getValueAsString("subcommand")));
      continue;
    }

    if (!indices.insert({*index, sub}).second) {
      invalid |= printError(
          sub->getLoc(),
          llvm::formatv(
              "subcommand '{0}' has index {1}, which has already been "
              "used; it will appear in a non-deterministic order",
              sub->getValueAsString("subcommand"), *index));
      const llvm::Record *previous = indices[*index];
      llvm::PrintNote(
          previous->getLoc(),
          llvm::formatv(
              "subcommand '{0}' has already been defined with index {1} here",
              previous->getValueAsString("subcommand"), *index));
    }
  }

  if (invalid)
    return Error("command description failed to validate");

  llvm::sort(subcommands, LessIndex());
  return CommandDescription(record, subcommands);
}

//===----------------------------------------------------------------------===//
// CommandOptionGroup
//===----------------------------------------------------------------------===//

CommandOptionGroup::CommandOptionGroup(const llvm::Record *group)
    : group(group) {
  assert(group->isSubClassOf("OptionGroup") && "unexpected record class");
}

ErrorOr<std::vector<CommandOptionGroup>>
M::CommandOptionGroup::getAll(const llvm::RecordKeeper &records) {
  // Create a sorted list of groups.
  std::vector<CommandOptionGroup> groups;
  ArrayRef<const llvm::Record *> groupRecords =
      records.getAllDerivedDefinitions("OptionGroup");
  groups.reserve(groupRecords.size());
  llvm::transform(
      groupRecords, std::back_inserter(groups),
      [](const llvm::Record *record) { return CommandOptionGroup(record); });
  llvm::sort(groups, LessIndex());

  // For each group, add the options that belong to that group.
  // First, bucket all option records based on their group record.
  DenseMap<const llvm::Record *, std::vector<const llvm::Record *>>
      groupOptions;
  llvm::SmallSet<int64_t, 4> groupIndices;
  for (const llvm::Record *option : records.getAllDerivedDefinitions("Option"))
    if (const llvm::Record *group = option->getValueAsOptionalDef("Group"))
      groupOptions[group].push_back(option);

  // Then, add each group record's options to their (sorted) lists.
  bool invalid = false;
  for (CommandOptionGroup &group : groups) {
    llvm::SmallSet<int64_t, 4> optionIndices;
    for (const llvm::Record *option : groupOptions[group.getGroup()]) {
      // If the option is an alias, don't add it to the group, add it to its
      // aliased option.
      if (const llvm::Record *aliased =
              option->getValueAsOptionalDef("Alias")) {
        ErrorOr<CommandOption &> aliasedOption =
            group.findOrCreateOption(aliased);
        if (failed(aliasedOption)) {
          invalid = true;
          continue;
        }

        invalid |= failed(aliasedOption.get().addAlias(option));
        continue;
      }

      invalid |= failed(group.findOrCreateOption(option));
    }

    // Now that we've constructed a group and all of its options, perform some
    // additional validation.
    if (std::optional<int64_t> index = group.getIndex()) {
      if (!groupIndices.insert(*index).second) {
        invalid |= printError(
            group.getGroup()->getLoc(),
            llvm::formatv("group '{0}' has index {1}, which has already been "
                          "used; it will appear in a non-deterministic order",
                          group.getGroupName(), *index));
      }
    } else {
      invalid |= printError(
          group.getGroup()->getLoc(),
          llvm::formatv("group '{0}' has no index with which to order it by; "
                        "it will appear in a non-deterministic order",
                        group.getGroupName()));
    }

    if (group.isHidden() && group.getOptions().empty())
      invalid |= printError(
          group.getGroup()->getLoc(),
          llvm::formatv("group '{0}' has no options", group.getGroupName()));

    if (!group.isHidden() &&
        llvm::none_of(group.getOptions(), [](const CommandOption &option) {
          return !CommandOption::isHidden(option.getOption());
        }))
      invalid |= printError(
          group.getGroup()->getLoc(),
          llvm::formatv("publicly documented group '{0}' has no publicly "
                        "documented options",
                        group.getGroupName()));
  }

  if (invalid)
    return Error("option groups failed to validate");

  return groups;
}

std::optional<int64_t> CommandOptionGroup::getIndex() const {
  return getValueAsOptionalIndex(group);
}

bool CommandOptionGroup::isHidden() const { return isRecordHidden(group); }

ErrorOr<CommandOption &>
M::CommandOptionGroup::findOrCreateOption(const llvm::Record *option) {
  assert(group == option->getValueAsDef("Group") &&
         "option does not belong to this group");

  auto it = llvm::lower_bound(options, CommandOption(option), LessIndex());
  if (it != options.end() && it->getOption() == option)
    return *it;

  CommandOption &result = *options.insert(it, CommandOption(option));

  // Now that we're processing this option for the first time, perform some
  // validation.
  StringRef name = option->getValueAsString("Name");
  StringRef helpText = result.getHelpText();
  bool invalid = failed(
      validateCapitalized(helpText, option->getLoc(),
                          llvm::formatv("help text for option '{0}'", name)));
  if (!helpText.ends_with(".") && !helpText.ends_with("\""))
    invalid |= printError(option->getLoc(),
                          llvm::formatv("help text for option '{0}' should end "
                                        "with a period or quotation mark",
                                        name));

  if (std::optional<StringRef> metaVarName = result.getMetaVarName()) {
    if (metaVarName->empty())
      invalid |= printError(
          option->getLoc(),
          llvm::formatv("option '{0}' metavar name should not be empty", name));

    if (metaVarName->upper() != *metaVarName)
      invalid |= printError(
          option->getLoc(),
          llvm::formatv("option '{0}' metavar name should be uppercase", name));
  } else if (!result.isFlag()) {
    invalid |=
        printError(option->getLoc(),
                   llvm::formatv("option '{0}' takes a value, but does not "
                                 "define a metavar name for that value",
                                 name));
  }

  if (!result.getIndex())
    invalid |= printError(
        option->getLoc(),
        llvm::formatv("option '{0}' has no index with which to order it by; "
                      "it will appear in a non-deterministic order",
                      name));

  if (isHidden() && CommandOption::isHidden(option))
    invalid |= printError(
        option->getLoc(),
        llvm::formatv("option group '{0}' is already hidden, marking its "
                      "option '{1}' as hidden is redundant",
                      getGroupName(), name));

  if (invalid)
    return Error("option failed to validate");

  return result;
}

//===----------------------------------------------------------------------===//
// CommandOption
//===----------------------------------------------------------------------===//

std::optional<int64_t> CommandOption::getIndex() const {
  return getValueAsOptionalIndex(option);
}

bool CommandOption::isHidden(const llvm::Record *option) {
  return isRecordHidden(option);
}

LogicalResult CommandOption::addAlias(const llvm::Record *aliasRecord) {
  CommandAlias alias(aliasRecord);

  bool invalid = false;
  std::optional<int64_t> aliasIndex =
      getValueAsOptionalIndex(alias.getRecord());
  auto it = llvm::lower_bound(aliases, alias, LessIndex());
  if (it != aliases.end()) {
    // If the alias already exists in the collection, no need to insert it.
    if (*it == alias)
      return success();

    // If we're inserting an alias behind another, they may have the same
    // index value. If so, emit a warning.
    if (std::optional<int64_t> index = getValueAsOptionalIndex(it->getRecord()))
      if (aliasIndex && aliasIndex == *index)
        invalid |= printError(
            aliasRecord->getLoc(),
            llvm::formatv("alias '{0}' has index {1}, which has already been "
                          "used; it will appear in a non-deterministic order",
                          aliasRecord->getValueAsString("Name"), *aliasIndex));
  }

  // Now that we're adding this alias for the first time, perform some
  // validation.
  if (!aliasIndex)
    invalid |= printError(
        aliasRecord->getLoc(),
        llvm::formatv("alias '{0}' has no index with which to order it by; "
                      "it will appear in a non-deterministic order",
                      aliasRecord->getValueAsString("Name")));

  aliases.insert(it, alias);
  return success(/*isSuccess=*/!invalid);
}

//===----------------------------------------------------------------------===//
// LessIndex
//===----------------------------------------------------------------------===//

bool LessIndex::operator()(const llvm::Record *lhs,
                           const llvm::Record *rhs) const {
  if (std::optional<int64_t> lhsIndex = getValueAsOptionalIndex(lhs))
    if (std::optional<int64_t> rhsIndex = getValueAsOptionalIndex(rhs))
      return lhsIndex < rhsIndex;
  return llvm::LessRecordByID()(lhs, rhs);
}
