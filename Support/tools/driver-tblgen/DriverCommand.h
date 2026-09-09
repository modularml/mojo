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

#ifndef SUPPORT_TOOLS_DRIVERTBLGEN_DRIVERCOMMAND_H
#define SUPPORT_TOOLS_DRIVERTBLGEN_DRIVERCOMMAND_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Record.h"
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace M {

/// A convenience wrapper for an `Alias` TableGen record. This provides
/// getters for the record's values, as well as other helper functions.
class CommandAlias {
public:
  /// Compare this alias with another alias for equality.
  bool operator==(const CommandAlias &other) const {
    return record == other.record;
  }

  /// Return the alias's optional metavar name, if one is defined.
  std::optional<StringRef> getMetaVarName() const {
    return record->getValueAsOptionalString("MetaVarName");
  }

  /// Return the underlying LLVM `alias` record.
  const llvm::Record *getRecord() const { return record; }

  /// Return the alias arguments for the alias.
  std::vector<StringRef> getAliasArguments() const;

private:
  /// Initializes the wrapper with the given `Alias` record.
  CommandAlias(const llvm::Record *record) : record(record) {
    assert(record->isSubClassOf("Alias") && "unexpected record class");
  }

  friend class CommandOption;

  /// The underlying alias record.
  const llvm::Record *record;
};

/// A convenience wrapper for a `CommandDescription` TableGen record. This
/// provides getters for the record's values, as well as other helper functions.
///
/// Instead of constructing instances of this class directly, use the static
/// `get` member function to construct one based on parsed TableGen records.
class CommandDescription {
public:
  /// Given a set of parsed TableGen records, either returns a concrete command
  /// description, or an error if none could be found. This also returns an
  /// error if more than one viable command description is found.
  ///
  /// The logic for finding the one viable command description is this:
  /// * Find all description records. If ANY describe a top-level command (their
  ///   `subcommand` field is an empty string), then the viable record will be a
  ///   top-level command (all subcommand records are ignored).
  ///   * If more than one record describes a top-level command, a warning is
  ///     emitted. In all cases, there should only be one top-level command
  ///     description.
  /// * If NO records describe a top-level command, then the viable record will
  ///   be a subcommand description.
  ///   * If more than one record describes a subcommand, a warning is emitted.
  ///     In the subcommand case, there should only be one subcommand
  ///     description.
  static ErrorOr<CommandDescription> get(const llvm::RecordKeeper &records);

  StringRef getExecutable() const {
    return record->getValueAsString("executable");
  }

  StringRef getSubcommand() const {
    return record->getValueAsString("subcommand");
  }

  StringRef getSummary() const { return record->getValueAsString("summary"); }

  StringRef getDescription() const {
    return record->getValueAsString("description");
  }

  std::vector<const llvm::Record *> getUsages() const {
    return record->getValueAsListOfDefs("usages");
  }

  /// Returns all of the command description records that are subcommands of
  /// this record.
  ArrayRef<const llvm::Record *> getSubcommands() const { return subcommands; }

  /// Given an joining string, joins the command description record's executable
  /// and subcommand values by that string.
  std::string getName(const Twine &join) const {
    return llvm::formatv("{0}{1}{2}", getExecutable(),
                         getSubcommand().empty() ? "" : join, getSubcommand());
  }
  std::string getName() const { return getName("-"); }

private:
  /// Initializes the wrapper with the given `CommandDescription` record, as
  /// well as any subcommand records.
  CommandDescription(const llvm::Record *record,
                     ArrayRef<const llvm::Record *> subcommands)
      : record(record), subcommands(subcommands) {
    assert(record->isSubClassOf("CommandDescription") &&
           "unexpected record class");
  }

  /// The underlying record representing this command description.
  const llvm::Record *record;
  /// This command's subcommands, if any.
  SmallVector<const llvm::Record *> subcommands;
};

/// A wrapper around an LLVM `Option` record, plus all of its aliases, which are
/// stored in a continuously sorted list. This helps backends print options and
/// their aliases side-by-side.
///
/// Instead of constructing instances of this class directly, use the
/// `CommandOptionGroup::getAll` member function to construct a collection of
/// groups and their options, based on parsed TableGen records.
class CommandOption {
public:
  /// Return the underlying LLVM `Option` record.
  const llvm::Record *getOption() const { return option; }

  /// Whether this option is a flag, meaning an option that takes no values.
  bool isFlag() const {
    return option->getValueAsDef("Kind")->getValueAsString("Name") == "Flag";
  }

  /// Return the option's help text, or an empty string if none exists.
  StringRef getHelpText() const {
    auto helpText = option->getValueAsOptionalString("HelpText");
    return helpText ? *helpText : "";
  }

  /// Return the option's optional metavar name, if one is defined.
  std::optional<StringRef> getMetaVarName() const {
    return option->getValueAsOptionalString("MetaVarName");
  }

  /// Return the option's index value, if one is defined.
  std::optional<int64_t> getIndex() const;

  /// Add an LLVM `Option` to the sorted list of aliases for this option.
  /// If an alias is newly added and is not valid, returns a failure. Otherwise,
  /// returns success.
  LogicalResult addAlias(const llvm::Record *aliasRecord);

  /// Return all the aliases of this option.
  ArrayRef<CommandAlias> getAliases() const { return aliases; }

  /// Whether the given option is hidden from help text.
  static bool isHidden(const llvm::Record *option);

  /// Return the first prefix defined for the given `option`, which we treat as
  /// the "preferred" prefix for help text.
  static StringRef getPreferredPrefix(const llvm::Record *option) {
    std::vector<StringRef> prefixes =
        option->getValueAsListOfStrings("Prefixes");
    // Only options such as `INPUT` and `UNKNOWN` can be defined without a
    // prefix, and we don't process those.
    assert(!prefixes.empty() && "all options must have a prefix");
    return prefixes.front();
  }

private:
  /// Initializes the wrapper with the given `Option` record.
  CommandOption(const llvm::Record *option) : option(option) {
    assert(option->isSubClassOf("Option") && "unexpected record class");
  }
  /// Allow `CommandOptionGroup` to construct instances of this class.
  friend class CommandOptionGroup;

  const llvm::Record *option;
  SmallVector<CommandAlias> aliases;
};

/// A wrapper around an LLVM `OptionGroup` record, as well as all of the
/// (continuously sorted) options that belong to that group. This helps backends
/// print options group-wise.
///
/// Instead of constructing instances of this class directly, use the static
/// `getAll` member function to construct a collection of them based on parsed
/// TableGen records.
class CommandOptionGroup {
public:
  /// Given a set of parsed TableGen records, returns a sorted list of all the
  /// option groups defined therein, along with their options. If any of the
  /// option group records are invalid, returns an error.
  static ErrorOr<std::vector<CommandOptionGroup>>
  getAll(const llvm::RecordKeeper &records);

  /// Return the underlying LLVM `OptionGroup` record.
  const llvm::Record *getGroup() const { return group; }
  /// Return all the options that belong to this group.
  ArrayRef<CommandOption> getOptions() const { return options; }

  StringRef getGroupName() const {
    return getGroup()->getValueAsString("Name");
  }

  /// Return the option group's index value, if one is defined.
  std::optional<int64_t> getIndex() const;

  /// Return whether the option group is hidden from help text.
  bool isHidden() const;

  /// Given an LLVM `Option` record, either add it to the sorted list of group
  /// options, or return the option that was already added. If the option record
  /// is to be newly added but is invalid, this returns an error.
  ErrorOr<CommandOption &> findOrCreateOption(const llvm::Record *option);

private:
  /// Initializes the wrapper with the given `OptionGroup` record.
  CommandOptionGroup(const llvm::Record *group);

  const llvm::Record *group;
  std::vector<CommandOption> options;
};

/// A comparator that can be used to sort option groups and options, based on
/// their index, in ascending order.
struct LessIndex {
  bool operator()(const llvm::Record *lhs, const llvm::Record *rhs) const;

  bool operator()(const CommandAlias &lhs, const CommandAlias &rhs) const {
    return operator()(lhs.getRecord(), rhs.getRecord());
  }
  bool operator()(const CommandOptionGroup &lhs,
                  const CommandOptionGroup &rhs) const {
    return operator()(lhs.getGroup(), rhs.getGroup());
  }

  bool operator()(const CommandOption &lhs, const CommandOption &rhs) const {
    return operator()(lhs.getOption(), rhs.getOption());
  }
};

} // namespace M

#endif // SUPPORT_TOOLS_DRIVERTBLGEN_DRIVERCOMMAND_H
