#!/bin/bash

for arg in "$@"
do
  output=$(jupyter nbconvert --log-level ERROR --to markdown "$arg" --stdout)
  if [ $? -ne 0 ]; then
    echo "^^^ Failed to convert $arg to markdown, see error above ^^^"
    echo "------------------------------------------------------------"
    exit 1
  fi
  echo "$output" | codespell -L inout -
  if [ $? -ne 0 ]; then
    echo "^^^ Spelling errors found in $arg, see error above ^^^"
    echo "------------------------------------------------------------"
    exit 1
  fi
done
