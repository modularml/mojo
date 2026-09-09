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
// This file contains utilities for processing and formatting Mojo doc strings
// into various formats.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_DOCSTRING_H
#define KGEN_MOJOPARSER_DOCSTRING_H

#include "Mojo/MojoParser/SharedState.h"
#include "Support/ADT/SmartVariant.h"

namespace M::KGEN::LIT {
class DocStringAttr;

//===----------------------------------------------------------------------===//
// DocString
//===----------------------------------------------------------------------===//

/// This class represents a processed Mojo doc string.
class DocString {
public:
  /// Construct a new DocString from a given raw doc-string.
  DocString(DocStringAttr rawDocStringAttr);

  /// Return the summary of the doc string.
  StringRef getSummary() const { return summary; }

  /// Return the full body description of the doc string.
  ArrayRef<StringRef> getDescription() const { return descriptionLines; }

  /// Format the given set of description lines into a single string for
  /// display as documentation, with each line separated by a newline character.
  static std::string formatDescription(ArrayRef<StringRef> descriptionLines);

  /// Return the beginning location of the doc string, or nullptr if the doc
  /// string is not attached to a location.
  FileLineColLoc getLoc() const { return loc; }

  //===----------------------------------------------------------------------===//
  // Code Blocks

  /// This class represents a code block within a doc string. Code blocks are
  /// defined by ```mojo style markdown blocks.
  class CodeBlock {
  public:
    /// Return the raw code block contained in the original doc string, fully
    /// indented as defined within the source file.
    StringRef getRawCode() const;

    /// Return the indentation level of the code block within the raw code
    /// string.
    unsigned getRawIndentLevel() const {
      return indentLevel + docString->indent;
    }

  private:
    CodeBlock(const DocString &docString, unsigned indentLevel,
              unsigned beginLine)
        : docString(&docString), indentLevel(indentLevel),
          lineRange(beginLine, beginLine) {}

    /// Allow access to the constructor.
    friend class DocString;

    /// The owning doc string.
    const DocString *docString;

    /// The indent level of the code block within the doc string.
    unsigned indentLevel;

    /// The range of lines, [start, end], within the description of the parent
    /// doc string.
    std::pair<unsigned, unsigned> lineRange;
  };

  /// Return the code blocks defined within the doc string.
  SmallVector<CodeBlock> getCodeBlocks() const;

  //===----------------------------------------------------------------------===//
  // Section names

  /// Within a doc string, the "Constraints" section describes invariants that
  /// must be true for the struct or function.
  static constexpr StringLiteral kSectionConstraints = "Constraints";

  /// Within a doc string, the "Parameters" section lists descriptions of each
  /// parameter.
  static constexpr StringLiteral kSectionParameters = "Parameters";

  /// Within a doc string, the "Args" section lists descriptions of each
  /// function argument.
  static constexpr StringLiteral kSectionArgs = "Args";

  /// Within a doc string, the "Returns" section describes the results of a
  /// function.
  static constexpr StringLiteral kSectionReturns = "Returns";

  /// Within a doc string, the "Raises" section describes the invariants
  /// surrounding raises within the function.
  static constexpr StringLiteral kSectionRaises = "Raises";

  //===----------------------------------------------------------------------===//
  // Known ad-hoc section names

  /// These "fake" section names appear in description portion of
  /// the docstring; if they're followed by indented content like
  /// a true section, it will format incorrectly, so we check for that.
  static constexpr StringLiteral kAdHocSectionExample = "Example";
  static constexpr StringLiteral kAdHocSectionExamples = "Examples";
  static constexpr StringLiteral kAdHocSectionNote = "Note";
  static constexpr StringLiteral kAdHocSectionNotes = "Notes";
  static constexpr StringLiteral kAdHocSectionWarning = "Warning";
  static constexpr StringLiteral kAdHocSectionPerformance = "Performance";
  static constexpr StringLiteral kAdHocSectionSafety = "Safety";

private:
  /// The short summary of the doc string.
  std::string summary;

  /// The lines comprising the description.
  SmallVector<StringRef> descriptionLines;

  /// The beginning location of the doc string.
  FileLineColLoc loc;

  /// The indentation of the doc string within the source file.
  size_t indent = 0;
};

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

/// Returns if the given decl should be hidden during documentation generation.
bool shouldHideDeclInDocGen(ASTDecl &decl, StringRef name);

/// Validate the doc string for the given decl, emitting warnings for any
/// invalid format issues.
void validateDocString(ASTDecl &decl);

/// Generate a template doc string for the given decl. Returns nullopt if no
/// template is available. The provided `indent` is the desired indentation
/// level of the template after the first line, and should be a multiple of 2.
std::optional<std::string> generateDocStringTemplate(ASTDecl &decl,
                                                     size_t indent = 0);

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_DOCSTRING_H
