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
file(WRITE "${OUTPUT_NOTEBOOK}" "${fileContents}")
