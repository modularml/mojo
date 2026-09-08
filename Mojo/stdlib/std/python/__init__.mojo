# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Python interoperability: import packages and modules, call functions, type conversion.

The `python` package enables interoperability between Mojo and Python
code. It provides mechanisms for importing Python packages and modules, calling Python
functions, and converting values between Mojo and Python types. This package
allows Mojo programs to leverage the extensive Python ecosystem while
maintaining Mojo's performance characteristics.

Use this package when you need to call Python libraries, integrate existing
Python code into Mojo applications, or leverage Python's ecosystem for
functionality not yet available in native Mojo.
"""

from .python import Python
from .conversions import ConvertibleFromPython, ConvertibleToPython
from .python_object import PythonObject
