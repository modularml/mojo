##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Copy over the notebooks and strip out the check lines from the notebooks.
get_filename_component(notebookDir "${OUTPUT_NOTEBOOK}" DIRECTORY)
file(COPY "${INPUT_NOTEBOOK}" DESTINATION "${notebookDir}")

file(READ "${OUTPUT_NOTEBOOK}" fileContents)
string(REGEX REPLACE "\n +\" *#\\|[^\"]*\"," "" fileContents "${fileContents}")

# Get the first 'raw' cell. That will be the front matter.
string(JSON numCells LENGTH "${fileContents}" "cells")
math(EXPR numCellsMinus1 "${numCells} - 1")
foreach(i RANGE 0 ${numCellsMinus1})
  string(JSON cellType GET "${fileContents}" "cells" ${i} "cell_type")
  if ("${cellType}" STREQUAL "raw")
    string(JSON frontMatter GET "${fileContents}" "cells" ${i} "source")
    break()
  endif()
endforeach()

# If we have front matter, check if it's valid YAML.
if (NOT ("${frontMatter}" STREQUAL ""))
  string(JSON frontMatterArrayLen LENGTH "${frontMatter}")

  # Match the first source line to the YAML start token.
  string(JSON srcLine GET "${frontMatter}" 0)
  string(REGEX MATCH "---" yamlStart "${srcLine}")

  if ("${yamlStart}" STREQUAL "---")
    set(frontMatterIsYaml TRUE)
  endif()

  # Match the inner source lines to YAML key/value lines.
  math(EXPR arrayLenMinus2 "${frontMatterArrayLen} - 2")
  foreach(i RANGE 1 ${arrayLenMinus2})
    string(JSON srcLine GET "${frontMatter}" ${i})
    string(REGEX MATCH "[a-z]+:.*\n" yamlLine "${srcLine}")
    if ("${yamlLine}" STREQUAL "")
      set(frontMatterIsYaml FALSE)
    endif()
  endforeach()

  # Finally, match the last line to the YAML end token.
  math(EXPR arrayLenMinus1 "${frontMatterArrayLen} - 1")
  string(JSON srcLine GET "${frontMatter}" ${arrayLenMinus1})
  string(REGEX MATCH "---" yamlEnd "${srcLine}")
  if ("${yamlEnd}" STREQUAL "")
    set(frontMatterIsYaml FALSE)
  endif()

  # OK great - if it is in fact YAML, we're good to remove it.
  if (frontMatterIsYaml)
    string(JSON fileContents REMOVE "${fileContents}" "cells" 0)
  endif()
endif()

file(WRITE "${OUTPUT_NOTEBOOK}" "${fileContents}")
