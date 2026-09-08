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

# RUN: kgen-doc --diagnose-missing-doc-strings -Wno-error %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WNO-ERROR
# RUN: not kgen-doc --diagnose-missing-doc-strings -Werror %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WERROR
# RUN: kgen-doc --diagnose-missing-doc-strings -Werror -Wno-error %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WERROR-THEN-WNO
# RUN: not kgen-doc --diagnose-missing-doc-strings -Wno-error -Werror %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WNO-THEN-WERROR

# WERROR: error: unknown argument 'y' in doc string
# WERROR-NOT: warning: unknown argument 'y' in doc string

# WNO-ERROR: warning: unknown argument 'y' in doc string
# WNO-ERROR-NOT: error: unknown argument 'y' in doc string

# -Werror followed by -Wno-error: warnings remain warnings (last wins)
# WERROR-THEN-WNO: warning: unknown argument 'y' in doc string
# WERROR-THEN-WNO-NOT: error: unknown argument 'y' in doc string

# -Wno-error followed by -Werror: warnings become errors (last wins)
# WNO-THEN-WERROR: error: unknown argument 'y' in doc string
# WNO-THEN-WERROR-NOT: warning: unknown argument 'y' in doc string

# Test that --validate-doc-strings is a deprecated alias for -Werror.
# RUN: not kgen-doc --diagnose-missing-doc-strings --validate-doc-strings %s -o /dev/null 2>&1 | FileCheck %s --check-prefix DEPRECATED
# DEPRECATED: warning: --validate-doc-strings is deprecated, use -Werror instead
# DEPRECATED: error: unknown argument 'y' in doc string

# Test that -Wno-error takes precedence over --validate-doc-strings.
# RUN: kgen-doc --diagnose-missing-doc-strings --validate-doc-strings -Wno-error %s -o /dev/null 2>&1 | FileCheck %s --check-prefix DEPRECATED-WNO
# RUN: kgen-doc --diagnose-missing-doc-strings -Wno-error --validate-doc-strings %s -o /dev/null 2>&1 | FileCheck %s --check-prefix DEPRECATED-WNO
# DEPRECATED-WNO-NOT: --validate-doc-strings is deprecated
# DEPRECATED-WNO: warning: unknown argument 'y' in doc string
# DEPRECATED-WNO-NOT: error: unknown argument 'y' in doc string

# Test that -Werror takes precedence over --validate-doc-strings.
# RUN: not kgen-doc --diagnose-missing-doc-strings --validate-doc-strings -Werror %s -o /dev/null 2>&1 | FileCheck %s --check-prefix DEPRECATED-WERROR
# RUN: not kgen-doc --diagnose-missing-doc-strings -Werror --validate-doc-strings %s -o /dev/null 2>&1 | FileCheck %s --check-prefix DEPRECATED-WERROR
# DEPRECATED-WERROR-NOT: --validate-doc-strings is deprecated
# DEPRECATED-WERROR: error: unknown argument 'y' in doc string


def f(x: Int):
    """This is a function with an invalid doc string.

    Args:
        y: This argument doesn't appear in the argument list.
    """
    pass
