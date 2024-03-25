##===----------------------------------------------------------------------===##
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##===----------------------------------------------------------------------===##

# Read the notebook and strip out the CHECK lines first thing.
file(READ "${INPUT_NOTEBOOK}" fileContents)
string(REGEX REPLACE
  "\n +\" *#\\| CHECK[^\"]*\","
  ""
  fileContents
  "${fileContents}")


# Write the file without the CHECK lines, but with YAML.
get_filename_component(notebookDir "${OUTPUT_NOTEBOOK}" DIRECTORY)
get_filename_component(notebookName "${OUTPUT_NOTEBOOK}" NAME)
set(WEBSITE_NOTEBOOK ${notebookDir}/nocheckyesyaml/${notebookName})
file(WRITE "${WEBSITE_NOTEBOOK}" "${fileContents}")
# For the version of notebooks going to the docs website,
# find cells with "REMOVE_FOR_WEBSITE" and strip the entire cell
file(MAKE_DIRECTORY ${notebookDir}/nocheckyesyaml/stripped/)
set(STRIPPED_NOTEBOOK ${notebookDir}/nocheckyesyaml/stripped/${notebookName})
execute_process(
    COMMAND sh -c "cat ${WEBSITE_NOTEBOOK} | jq '.cells = [.cells[] | select(.source[0] | test(\"REMOVE_FOR_WEBSITE\")? | not)]' > ${STRIPPED_NOTEBOOK}"
)
file(REMOVE ${WEBSITE_NOTEBOOK}) # Don't need the "nocheckyesyaml" file anymore

# For the version of notebooks going to GitHub and the Playground,
# just remove the comment for "REMOVE_FOR_WEBSITE" (leaving the cell intact)
string(REGEX REPLACE
  "\n +\" *\\[\\/\\/\\]: # REMOVE_FOR_WEBSITE[^\"]*\","
  ""
  fileContents
  "${fileContents}")

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
    string(REGEX MATCH "[a-z]+:.*\n" yamlKVLine "${srcLine}")
    string(REGEX MATCH "- .*\n" yamlArrayLine "${srcLine}")
    if (("${yamlKVLine}" STREQUAL "") AND ("${yamlArrayLine}" STREQUAL ""))
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

# Write the final output.
file(WRITE "${OUTPUT_NOTEBOOK}" "${fileContents}")
