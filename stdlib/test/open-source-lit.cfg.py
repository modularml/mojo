# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from pathlib import Path
import os

import lit.formats
import lit.llvm

config.test_format = lit.formats.ShTest(True)

# name: The name of this test suite.
config.name = "Mojo Standard Library"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mojo"]

# test_source_root: The root path where tests are located.
config.test_source_root = Path.cwd()

# TODO: Perhaps we'll have a build-and-test script
# that creates a `build` directory and runs the tests there.
config.test_exec_root = Path.cwd()

# Substitute %mojo for just `mojo` itself
# since we're not supporting `--sanitize` initially
# to allow running the tests with LLVM sanitizers.
config.substitutions.insert(0, ("%mojo", "mojo"))

test_root = Path.cwd().parent
stdlib_root = Path.cwd().parent.parent
os.environ["MODULAR_MOJO_NIGHTLY_IMPORT_PATH"] = (
    str(stdlib_root) + "," + str(test_root)
)

# Pass through several environment variables
# to the underlying subprocesses that run the tests.
lit.llvm.initialize(lit_config, config)
lit.llvm.llvm_config.with_system_environment(
    [
        "HOME",
        "INCLUDE",
        "LIB",
        "PATH",
        "MODULAR_HOME",
        "MODULAR_MOJO_NIGHTLY_IMPORT_PATH",
        "TEMP_DIR",
        "TMP",
        "TEMP",
    ]
)
