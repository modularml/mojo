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
#
# This file contains python code to be run inside of the REPL process for
# enabling matplotlib inline plotting. It registers a custom backend with
# matplotlib, see the `backend_str` variable below and the `matplotlib_backend`
# module, to route data to the kernel process for display.
#
# ===----------------------------------------------------------------------=== #

try:
    # Do all imports with leading underscores. This ensures that they don't
    # clash with user imports.
    import json as _json
    import sys as _sys
    import types as _types
    from binascii import b2a_base64 as _b2a_base64
    from io import BytesIO as _BytesIO

    import matplotlib as _matplotlib
    import matplotlib.pyplot as _plt
    from matplotlib.backend_bases import FigureCanvasBase as _FigureCanvasBase

    # ===------------------------------------------------------------------=== #
    # Backend loading
    # To keep the repl process contained, and reduce the amount of expression
    # traffic, we manually load the backend module and register it with
    # matplotlib. `backend_str` below is provided by the module that imports
    # this file, and contains the source code for the backend module.
    raw_backend = "matplotlib_mojo_inline_backend"
    backend_module = _types.ModuleType(raw_backend)
    exec(backend_str, backend_module.__dict__)
    _sys.modules[raw_backend] = backend_module

    # ===------------------------------------------------------------------=== #
    # matplotlib registration

    _matplotlib.interactive(True)
    backend = f"module://{raw_backend}"
    _matplotlib.rcParams["backend"] = backend
    _plt.switch_backend(backend)
    _plt.show._needmain = False

    def display(fig, metadata) -> None:  # noqa: ANN001
        if fig.canvas is None:
            _FigureCanvasBase(fig)

        # Encode the figure as a PNG so that we can send it to the kernel
        # process.
        byte_stream = _BytesIO()
        fig.canvas.print_figure(
            byte_stream,
            format="png",
            facecolor=fig.get_facecolor(),
            edgecolor=fig.get_edgecolor(),
            dpi=fig.dpi,
            bbox_inches="tight",
        )
        encoded_png = str(_b2a_base64(byte_stream.getvalue()).decode("ascii"))[
            :-1
        ]

        # Build an encoded message containing the image data and metadata.
        msg = [
            {"image/png": encoded_png},
            metadata if metadata is not None else {},
        ]

        # The kernel process needs to be able to extract our display data from
        # the output stream. We use a special delimiter to mark the start and
        # end of our display data. We also make sure to flush the output stream
        # so that the kernel process can read the data immediately.
        # `display_start` and `display_end` are provided by the script that
        # imports this one so we are never out of sync.
        print(display_start, _json.dumps(msg), display_end, flush=True)

    # Manually set the property we need on the backend we created with `exec`
    # above.
    _sys.modules[raw_backend].show._display = display
except:
    # If anything goes wrong, don't blow up the jupyter kernel, just pass. This
    # allows for processing to continue as before, e.g. if matplotlib or other
    # dependencies are not installed.
    pass
