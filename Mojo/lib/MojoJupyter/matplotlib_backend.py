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
# Copyright (c) IPython Development Team.
# Distributed under the terms of the BSD 3-Clause License.
#
# ===----------------------------------------------------------------------=== #
#
# This file contains a modified version of the matplotlib inline backend from
# IPython. It has been tweaked to work with the Mojo Kernel, and be able to
# send data back to the kernel process for display.
#
# The genesis of this version of the code originated from the swift-jupyter
# project: https://github.com/liuliu/swift-jupyter/blob/main/kernels/swift/matplotlib_swift/backend_inline.py
#
# ===----------------------------------------------------------------------=== #

import matplotlib as _matplotlib
import matplotlib.pyplot as _pyplot
from matplotlib import colors
from matplotlib._pylab_helpers import Gcf as _Gcf
from matplotlib.backends.backend_agg import FigureCanvasAgg as _FigureCanvasAgg


def show(close=None, block=None) -> None:  # noqa: ANN001
    if close is None:
        close = show._close_figures
    try:
        for figure_manager in _Gcf.get_all_fig_managers():
            show._display(
                figure_manager.canvas.figure,
                _fetch_figure_metadata(figure_manager.canvas.figure),
            )
    finally:
        show._to_draw = []
        # only call close('all') if any to close
        # close triggers gc.collect, which can be slow
        if close and _Gcf.get_all_fig_managers():
            _pyplot.close("all")


# This flag will be reset by draw_if_interactive when called
show._draw_called = False
# list of figures to draw when flush_figures is called
show._to_draw = []
# Close all figures at the end of each cell.
show._close_figures = True
# Placeholder for display function.
show._display = None


def draw_if_interactive() -> None:
    manager = _Gcf.get_active()
    if manager is None:
        return
    fig = manager.canvas.figure

    # Hack: matplotlib FigureManager objects in interactive backends (at least
    # in some of them) monkeypatch the figure object and add a .show() method
    # to it.  This applies the same monkeypatch in order to support user code
    # that might expect `.show()` to be part of the official API of figure
    # objects.
    # For further reference:
    # https://github.com/ipython/ipython/issues/1612
    # https://github.com/matplotlib/matplotlib/issues/835

    if not hasattr(fig, "show"):
        # Queue up `fig` for display
        fig.show = lambda *a: show._display(fig, _fetch_figure_metadata(fig))

    # If matplotlib was manually set to non-interactive mode, this function
    # should be a no-op (otherwise we'll generate duplicate plots, since a user
    # who set ioff() manually expects to make separate draw/show calls).
    if not _matplotlib.is_interactive():
        return

    # ensure current figure will be drawn, and each subsequent call
    # of draw_if_interactive() moves the active figure to ensure it is
    # drawn last
    try:
        show._to_draw.remove(fig)
    except ValueError:
        # ensure it only appears in the draw list once
        pass
    # Queue up the figure for drawing in next show() call
    show._to_draw.append(fig)
    show._draw_called = True


def flush_figures():  # noqa: ANN201
    if not show._draw_called:
        return

    if show._close_figures:
        # ignore the tracking, just draw and close all figures
        return show(True)
    try:
        # exclude any figures that were closed:
        active = {fm.canvas.figure for fm in _Gcf.get_all_fig_managers()}
        for fig in [fig for fig in show._to_draw if fig in active]:
            show._display(fig, _fetch_figure_metadata(fig))
    finally:
        # clear flags for next round
        show._to_draw = []
        show._draw_called = False


# Changes to matplotlib in version 1.2 requires a mpl backend to supply a default
# figurecanvas. This is set here to a Agg canvas
# See https://github.com/matplotlib/matplotlib/pull/1125
FigureCanvas = _FigureCanvasAgg


def _fetch_figure_metadata(fig):  # noqa: ANN001, ANN202
    # determine if a background is needed for legibility
    if _is_transparent(fig.get_facecolor()):
        # the background is transparent
        ticksLight = _is_light(
            [
                label.get_color()
                for axes in fig.axes
                for axis in (axes.xaxis, axes.yaxis)
                for label in axis.get_ticklabels()
            ]
        )
        if ticksLight.size and (ticksLight == ticksLight[0]).all():
            # there are one or more tick labels, all with the same lightness
            return {"needs_background": "dark" if ticksLight[0] else "light"}

    return None


def _is_light(color):  # noqa: ANN001, ANN202
    rgbaArr = colors.to_rgba_array(color)
    return rgbaArr[:, :3].dot((0.299, 0.587, 0.114)) > 0.5


def _is_transparent(color):  # noqa: ANN001, ANN202
    rgba = colors.to_rgba(color)
    return rgba[3] < 0.5


def set_matplotlib_close(close=True) -> None:  # noqa: ANN001
    show._close_figures = close
