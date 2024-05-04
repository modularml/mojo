# ===----------------------------------------------------------------------=== #
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
# ===----------------------------------------------------------------------=== #
"""This module includes the exit functions."""


from .ffi import external_call


fn exit():
    """Exits from Mojo. Unlike the Python implementation this does not raise an
    exception to exit.
    """
    exit(0)


fn exit[intable: Intable](code: intable):
    """Exits from Mojo. Unlike the Python implementation this does not raise an
    exception to exit.

    Parameters:
        intable: The type of the exit code.

    Args:
        code: The exit code.
    """
    external_call["exit", NoneType](Int32(int(code)))
