#!/usr/bin/env python3
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

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from kprofile import KbenchPKL

MARKER_COLORS = [
    "#4285f4",  # blue
    "#ea4335",  # red
    "#fbbc04",  # yellow
    "#34a853",  # green
    "#ff6d01",  # orange
    # alternative repeat colors
    "#0077b6",  # blue
    "#c1121f",  # red
    "#a7c957",  # green
    "#ffc300",  # yellow
    "coral",
    "lightskyblue",
    "silver",
    "magneta",
]

# Conversion factors: multiply by this value to convert from the unit to seconds.
# Order matters: descending magnitude for auto-selection algorithm.
TIME_UNIT_SECONDS = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}
TIME_UNITS = tuple(TIME_UNIT_SECONDS.keys())
_VALID_UNITS = frozenset(TIME_UNIT_SECONDS.keys()) | {"auto"}


def _get_max_groups_per_chart(x_labels, width_px, font_size_pt):  # noqa: ANN001, ANN202
    x_labels_len = [
        ([len(t) for t in entry.split("<br>")]) for entry in x_labels
    ]
    max_width = max(max(x_labels_len))
    avg_bar_width = (font_size_pt * max_width) if max_width else 1
    max_groups_per_chart = (width_px + avg_bar_width - 1) // avg_bar_width
    return max_groups_per_chart


@dataclass
class PlotConfig:
    """
    extension: File extension for output images
    prefix: Prefix for output filenames
    scale: Scaling factor for output image quality
    font_size_pt: Font-size in points
    width_px: Plot width in pixels
    height_px: Plot height in pixels
    prec: FP rounding precision of values
    bgcolor: Background color
    ytext: Print value on top of the bar
    ytext_unit: Unit for ytext (auto/s/ms/us/ns)
    font_family: Font family
    barmode: Bar mode ["group", "stacked"]
    groups_per_chart: ["all"=-1 (default), "auto"=0, int]
    """

    extension: str = "png"  # ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'eps', 'json', 'html']
    prefix: str = "img"
    scale: float = 1.0
    font_size_pt: int = 16
    width_px: int = 1920
    height_px: int = 1080
    prec: int = 3
    bgcolor: str = "#ffffff"
    y_text: bool = True  # Print value on top of the bar
    ytext_unit: str = "auto"
    barmode: str = "group"  # ["group", "stacked"]
    font_family: str = "Courier New"
    title_font_family: str = "Times New Roman"
    groups_per_chart: int = 0


def _replace_time_unit(label: str, base_unit: str, target_unit: str) -> str:
    """Replace time unit in a label: 'met (s)' with target='us' -> 'met (us)'."""
    if base_unit == target_unit:
        return label
    pattern = rf"\({re.escape(base_unit)}\)"
    return re.sub(
        pattern, f"({target_unit})", label, count=1, flags=re.IGNORECASE
    )


def _resolve_ytext_unit(
    y_title: str, y_values: np.ndarray, requested_unit: str
) -> tuple[str, float, str, str | None]:
    """Resolve the target time unit and compute the scaling factor.

    Returns:
        (updated_y_title, scale_factor, unit_suffix, base_unit)
        - scale_factor: multiply original values by this to get target unit
        - unit_suffix: string to append to bar labels (e.g., 'us')
        - base_unit: original unit parsed from y_title, or None
    """
    unit = requested_unit.lower()
    if unit not in _VALID_UNITS:
        raise ValueError(f"Unsupported ytext unit '{requested_unit}'.")

    # Parse base unit from y_title (e.g., 'met (s)' -> 's')
    if match := re.search(r"\(([^)]+)\)", y_title):
        parsed = match.group(1).strip().lower()
        base_unit = parsed if parsed in TIME_UNIT_SECONDS else None
    else:
        base_unit = None

    if base_unit is None:
        if unit == "auto":
            return y_title, 1.0, "", None
        raise ValueError(
            "ytext-unit requires a time axis label like 'met (s)'."
        )

    # Auto-select unit based on data magnitude
    if unit == "auto":
        # Compute max absolute value, ignoring NaN/Inf
        arr = np.asarray(y_values, dtype=float).ravel()
        finite = arr[np.isfinite(arr)] if arr.size > 0 else arr
        max_val = float(np.max(np.abs(finite))) if finite.size > 0 else 0.0
        max_seconds = max_val * TIME_UNIT_SECONDS[base_unit]

        # Select largest unit where value >= 1 in that unit
        unit = base_unit
        if max_seconds > 0:
            for u, threshold in list(TIME_UNIT_SECONDS.items())[:-1]:  # skip ns
                if max_seconds >= threshold:
                    unit = u
                    break
            else:
                unit = "ns"

    # Compute scale: e.g., s->us means scale = 1.0 / 1e-6 = 1e6
    scale = TIME_UNIT_SECONDS[base_unit] / TIME_UNIT_SECONDS[unit]
    y_title = _replace_time_unit(y_title, base_unit, unit)
    return y_title, scale, unit, base_unit


def _apply_time_unit(
    y_title: str,
    y_names: Sequence[str],
    y_values: np.ndarray,
    requested_unit: str,
) -> tuple[str, list[str], np.ndarray, str]:
    """Apply time unit conversion to title, legend names, and values.

    Ensures consistent units across the entire chart (axis, legend, bar labels).
    """
    y_title, scale, unit, base_unit = _resolve_ytext_unit(
        y_title=y_title,
        y_values=y_values,
        requested_unit=requested_unit,
    )
    # Update legend names to reflect the new unit
    if base_unit is not None:
        y_names = [
            _replace_time_unit(name, base_unit, unit) for name in y_names
        ]
    if scale != 1.0:
        y_values = y_values * scale
    return y_title, list(y_names), y_values, unit


def draw_plot(
    x: list[str],
    y_list: np.ndarray,
    y_names: Sequence[str],
    x_title: str,
    y_title: str,
    cfg: PlotConfig,
) -> None:
    """Draw bar plots for benchmark results.

    Args:
        x: X-axis labels
        y_list: 2D array of Y values
        y_names: Names for each Y series
        x_title: X-axis title
        y_title: Y-axis title
        cfg: Plot configuration data
    """

    extension = cfg.extension
    prefix = cfg.prefix
    scale = cfg.scale
    font_size_pt = cfg.font_size_pt
    width_px = cfg.width_px
    height_px = cfg.height_px
    prec = cfg.prec
    bgcolor = cfg.bgcolor

    # Convert to a human-readable time unit (e.g., seconds -> microseconds)
    y_title, y_names, y_list, unit_suffix = _apply_time_unit(
        y_title=y_title,
        y_names=y_names,
        y_values=y_list,
        requested_unit=cfg.ytext_unit,
    )

    layout = go.Layout(autosize=False, width=width_px, height=height_px)

    def plot_draw(xs: Sequence[str], ys_list: np.ndarray, name: str) -> None:
        fig = go.Figure(layout=layout)
        fig.update_layout(
            font_family=cfg.font_family,
            title_font_family=cfg.title_font_family,
            font_size=font_size_pt,
        )

        fig.update_layout(
            legend=dict(
                yanchor="top",
                y=1.2,
                xanchor="left",
                orientation="h",
            )
        )

        for i, ys in enumerate(ys_list):
            # Format bar labels: round values and append unit suffix (e.g., '10.6us')
            if cfg.y_text:
                rounded = np.round(ys, prec)
                text = (
                    [f"{v}{unit_suffix}" for v in rounded]
                    if unit_suffix
                    else rounded
                )
            else:
                text = None

            fig.add_trace(
                go.Bar(
                    x=xs,
                    y=ys,
                    name=y_names[i],
                    text=text,
                    textposition="outside",
                    textangle=0,
                    marker_color=MARKER_COLORS[i % len(MARKER_COLORS)],
                    showlegend=True,
                )
            )

        fig.update_layout(
            barmode=cfg.barmode,
            xaxis_tickangle=0,
            # bargap=0.1,
            bargroupgap=0.075,
            xaxis_title=x_title,
            yaxis_title=y_title,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor=bgcolor,
            paper_bgcolor=bgcolor,
        )
        fig.update_xaxes(showline=True, linewidth=2, linecolor="black")
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black")

        # fig.show(config=config)
        if to_html:
            plotly.offline.plot(fig, filename=f"{name}.html")
            # config = dict({"scrollZoom": True})
            # fig.write_html(file=name, config=config)
        else:
            fig.write_image(name, scale=scale)
        print(f"- added [{name}]")

    to_html = extension == "html"

    if cfg.groups_per_chart == -1:
        delta = len(x)
    elif cfg.groups_per_chart == 0:
        delta = _get_max_groups_per_chart(
            x_labels=x,
            width_px=width_px,
            font_size_pt=font_size_pt,
        )
    else:
        assert cfg.groups_per_chart > 0
        delta = cfg.groups_per_chart

    for i in range(0, len(x), delta):
        plot_draw(
            x[i : i + delta],
            y_list[:, i : i + delta],
            name=f"{prefix}_{i}.{extension}",
        )


def get_labels_by_pivots(x_labels, pivots):  # noqa: ANN001, ANN201
    df = label_to_df(x_labels)
    assert len(df) == len(x_labels)
    pivot_labels = []
    for i in range(len(df)):
        result = []
        for pivot in pivots:
            result += [f"{pivot}={df.loc[i, pivot]}"]
        pivot_labels.append("<br>".join(result))
    return pivot_labels


def wrap_labels(x_labels):  # noqa: ANN001, ANN201
    return [entry.replace("$", "") for entry in x_labels]


def label_to_df(x_labels):  # noqa: ANN001, ANN201
    ds = []
    for label in x_labels:
        vals = label.replace("$", "").split("/")
        # ignore the name in vals[0]
        vals = [val.split("=") for val in vals[1:]]
        d = {}
        for k, v in vals:
            d[k] = v
        ds.append(d)
    df = pd.DataFrame(ds)
    return df


def extract_pivots(x_labels):  # noqa: ANN001, ANN201
    df = label_to_df(x_labels)
    pivot_columns = []
    for c in df.columns:
        if len(set(df[c])) > 1:
            pivot_columns.append(c)

    # set(df.columns)-set(pivot_columns)
    non_pivot_columns = [c for c in df.columns if c not in pivot_columns]
    return pivot_columns, non_pivot_columns


def append_wrap_fixed_width(lst: list[str], sep: str, num_lines: int = 2):  # noqa: ANN201
    s: list[str] = []
    result = []
    current_len = 0
    width = sum([len(x) for x in lst]) // num_lines
    for x in lst:
        if current_len > width:
            result += [sep.join(s)]
            s = []
            current_len = 0
        s += [x]
        current_len += len(x)
    result += [sep.join(s)]
    return "<br>".join(result)


def parse_and_plot(
    path_list: list[Path],
    label_list: list[str],
    key_col: str,
    target_col: str = "1",
    compare: bool = False,
    pivots: list[str] = [],  # noqa: B006
    cfg: PlotConfig = PlotConfig(),  # noqa: B008
    force: bool = False,
) -> None:
    """Parse CSV files and generate plots.

    Args:
        path_list: List of paths to CSV files
        label_list: List of names for each CSV
        target_col: Index of column to plot
        compare: Whether to compare against baseline
        cfg: PlotConfig
        force: Skip input validation if True
    """

    tables = []
    for path in path_list:
        if path.suffix == ".csv":
            table = pd.read_csv(path)
        else:
            # add support for build_df
            table = KbenchPKL.load(path)["merged_df"]
            if "mesh_idx" in table.columns:
                table = table.drop(columns=["mesh_idx"])
        tables.append(table)

    assert key_col in ["name", "spec"]

    base_table = tables[0]
    base_table_columns = base_table.columns

    #######################################################
    # Check target_col
    #######################################################
    # if target_col is an index then ensure it is smaller than number of base table columns.
    if str(target_col).isnumeric():
        target_col_idx = int(target_col)
        msg = f"target_col ({target_col_idx}) cannot exceed the number of columns ({len(base_table_columns)})"
        assert target_col_idx < len(base_table_columns), msg
    else:
        # if target_col is str then ensure it is among base table column titles.
        msg = f"target-col={target_col} is not in column names: {list(base_table_columns)}"
        assert target_col in base_table_columns, msg
        target_col_idx = base_table_columns.get_loc(target_col)

    # ensure the target_col is a measure
    col_dtype = base_table[base_table_columns[target_col_idx]].dtype
    msg = f"target-col dtype ({col_dtype}) is not np.floating or np.integer"
    assert np.issubdtype(col_dtype, np.floating) or np.issubdtype(
        col_dtype, np.integer
    ), msg
    target_col_name = base_table_columns[target_col_idx]
    print(f"- Plot column:[{target_col_idx}]['{target_col_name}']")

    #######################################################
    # Sanity check to ensure the tables are comparable
    #######################################################
    key_col_idx = base_table_columns.get_loc(key_col)
    print(f"- Key column: [{key_col_idx}]['{key_col}']")

    if not force:
        base_key_col_values = base_table.iloc[:, key_col_idx]
        # ensure the each row of key-col has a unique value
        assert len(list(base_key_col_values)) == len(set(base_key_col_values))

        for t in tables[1:]:
            np.testing.assert_equal(list(base_table_columns), list(t.columns))
            np.testing.assert_equal(base_table.shape, t.shape)
            np.testing.assert_equal(
                list(base_table.iloc[:, key_col_idx]),
                list(t.iloc[:, key_col_idx]),
            )
    # TODO: add better log message
    print(f"- Entries have matching reference columns on key=['{key_col}']")

    #######################################################
    # Fetch values for y-axis
    #######################################################
    y_list, y_names = [], []

    if compare:
        ratio_df = pd.DataFrame({base_table_columns[0]: base_table.iloc[:, 0]})
        for i, t in enumerate(tables[1:]):
            ratio = (
                t.iloc[:, target_col_idx] / base_table.iloc[:, target_col_idx]
            )
            name = f"{target_col_name} [{label_list[i + 1]}/{label_list[0]}]"
            ratio_df[name] = ratio
            y_list.append(ratio)
            y_names.append(name)
    else:
        for i, t in enumerate(tables):
            name = f"{target_col_name} [{label_list[i]}]"
            y_list.append(t.iloc[:, target_col_idx])
            y_names.append(name)

    #######################################################
    # ??
    #######################################################
    x_labels = np.array(base_table.iloc[:, key_col_idx])
    x_title_append = ""
    if key_col == "spec":
        # search for pivots in the data (exclude columns that contain the same value in all rows.)
        ext_pivots, non_pivots = extract_pivots(x_labels)
        ext_pivots.extend(pivots)

        df = label_to_df(x_labels)
        s: list[str] = []
        for npv in non_pivots:
            if npv in pivots:
                continue
            s += [f"{npv}={df.iloc[0][npv]}"]

        x_title_append = append_wrap_fixed_width(s, " / ")
        x_labels = get_labels_by_pivots(x_labels, pivots=ext_pivots)

    x_labels = wrap_labels(x_labels)
    # TODO: check for sanity of x_labels

    draw_plot(
        x=list(x_labels),
        y_list=np.array(y_list),
        y_names=y_names,
        x_title=f"<b>{key_col}</b>"
        + (f"<br>({x_title_append})" if x_title_append else ""),
        y_title=target_col_name,
        cfg=cfg,
    )


@click.command(
    help="Drawing bar plots for kbench results",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "-help", "--help"]},
)
@click.option(
    "--label",
    "label_list",
    multiple=True,
    help="List of corresponding labels for CSV files",
)
@click.option(
    "--output",
    "-o",
    "output_prefix",
    default=None,
    help="Prefix for output file",
)
@click.option(
    "--plot-col", default=1, type=click.STRING, help="Plot column index/name"
)
@click.option(
    "--key",
    "-k",
    default="spec",
    type=click.STRING,
    help="Name of ref-key column (should be identical for corresponding entries across files)",
)
@click.option(
    "--compare",
    "-c",
    is_flag=True,
    default=False,
    help="Compare CSVs using first one as baseline",
)
@click.option(
    "--extension",
    "-x",
    default="png",
    type=click.STRING,
    help="Output file extension ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'eps', 'json', 'html'] (default=png)",
)
@click.option(
    "--scale",
    "-s",
    default=2.0,
    type=click.FLOAT,
    help="Output image scaling factor (default=2.0)",
)
@click.option(
    "--prec",
    default=2,
    type=click.INT,
    help="Floating point rounding precision for bar values (default=2)",
)
@click.option(
    "--ytext",
    default=True,
    type=click.BOOL,
    help="Print bar values (default=True)",
)
@click.option(
    "--ytext-unit",
    default="auto",
    type=click.Choice(["auto", *TIME_UNITS], case_sensitive=False),
    help="Ytext/axis unit for time metrics (auto/s/ms/us/ns)",
)
@click.option(
    "--groups-per-chart",
    "-g",
    default=0,
    type=click.INT,
    help="Number of groups per chart [-1=all, 0=auto, integer] (default=auto)",
)
@click.option(
    "--force", "-f", is_flag=True, default=False, help="Skip input validation"
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable verbose output"
)
@click.option(
    "--pivot",
    "-p",
    "pivot",
    multiple=True,
    help="List of corresponding pivots for plot",
)
@click.argument("input_files", nargs=-1, type=click.UNPROCESSED)
def cli(
    input_files,  # noqa: ANN001
    label_list: list[str],
    output_prefix: str | None,
    plot_col: str,
    key: str,
    compare: bool,
    extension: str,
    scale: float,
    prec: int,
    ytext: bool,
    ytext_unit: str,
    groups_per_chart: int,
    force: bool,
    verbose: bool,
    pivot,  # noqa: ANN001
) -> None:
    """CLI entry point for plotting benchmark results."""
    input_path_list = [Path(file).resolve() for file in input_files]
    num_input = len(input_path_list)

    for path in input_path_list:
        assert path.suffix in [".csv", ".pkl"], (
            "Path should have .csv/.pkl suffix."
        )
        assert Path(path).exists(), f"Path {path} doesn't exist!"

    # ignore rest of labels
    if len(label_list) < num_input:
        label_list = list(label_list)
        label_list.extend([p.name for p in input_path_list[len(label_list) :]])

    # add documentation for spec: name/separated param-value pairs
    assert key in ["spec", "name"]
    prefix = output_prefix or "img"

    cfg = PlotConfig(
        extension=extension,
        prefix=prefix,
        scale=scale,
        prec=prec,
        y_text=ytext,
        ytext_unit=ytext_unit.lower(),
        groups_per_chart=groups_per_chart,
    )

    parse_and_plot(
        input_path_list,
        label_list=label_list,
        key_col=key,
        target_col=plot_col,
        compare=compare,
        pivots=pivot,
        cfg=cfg,
        force=force,
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    if directory := os.environ.get("BUILD_WORKING_DIRECTORY"):
        os.chdir(directory)
    main()
