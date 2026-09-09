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


import ast
import os
import pickle
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from kbench import Spec
from pandas.api.types import is_numeric_dtype
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")

LINE = 80 * "-"


HEADER = """##===----------------------------------------------------------------------===##
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
##===----------------------------------------------------------------------===##
"""


def spec_to_dict(spec: str) -> dict:  # type: ignore[type-arg]
    """Convert kbench spec to dictionary"""
    # TODO: move this method to `kbench`
    spec_split = spec.split("/")
    d = {}
    d["name"] = spec_split[0]
    for x in spec_split[1:]:
        k, v = x.split("=")
        k = k.strip("$")
        try:
            d[k] = eval(v)
        except:
            d[k] = v
    return d


def specs_to_df(specs: list[str]) -> pd.DataFrame:
    ds = [spec_to_dict(x) for x in specs]
    df = pd.DataFrame(ds)
    return df


def extract_pivots_df(df: pd.DataFrame, exclude: list[str] | None = None):  # noqa: ANN201
    if exclude is None:
        exclude = []
    # df = specs_to_df(x_labels)
    valid_columns = []
    for c in list(df.columns):
        if c not in exclude:
            valid_columns.append(c)

    pivot_columns = []
    for c in valid_columns:
        if len(set(df[c])) > 1:
            pivot_columns.append(c)

    # set(df.columns)-set(pivot_columns)
    non_pivot_columns = [c for c in valid_columns if c not in pivot_columns]
    return pivot_columns, non_pivot_columns


def extract_pivots(x_labels: list[str], exclude: list[str] | None = None):  # noqa: ANN201
    df = specs_to_df(x_labels)
    return extract_pivots_df(df=df, exclude=exclude)


def load_pickle(path):  # noqa: ANN001, ANN201
    with open(path, "rb") as handle:
        return pickle.load(handle)


def dump_yaml(obj, out_path) -> None:  # noqa: ANN001
    with open(out_path, "w") as f:
        yaml.dump(obj, f, sort_keys=False)

    # TODO: add this as a separate option, probably dict->yaml-str
    yaml.dump(obj, sys.stdout, sort_keys=False)


def top_idx(x, top_percentage=0.05):  # noqa: ANN001, ANN201
    # calculate the threshold to pick top_percentage of the results
    threshold = (top_percentage + (np.min(x) / np.max(x))) * np.max(x)
    return x.where(x < threshold).dropna().index


def find_common_params(subset):  # noqa: ANN001, ANN201
    spec_list = []
    for index, row in subset.iterrows():  # noqa: B007
        p = spec_to_dict(row["spec"])
        spec_list.append(pd.DataFrame([p]))
    merged_specs = pd.concat(spec_list, axis=0, ignore_index=True)

    spec = {}
    for c in merged_specs.columns:
        frequent_val = merged_specs[c].value_counts().idxmax()
        spec[c] = frequent_val

    return spec


def df_to_console_table(
    df,  # noqa: ANN001
    col_style={},  # noqa: ANN001, B006
    header_style="bold blue",  # noqa: ANN001
    index=False,  # noqa: ANN001
) -> None:
    console = Console()
    table = Table(show_header=True, header_style=header_style)

    if index:
        style = col_style.get("index", "bold blue")
        table.add_column("index", justify="left", style=style)

    for c in df.columns:
        style = col_style.get(c, None)
        table.add_column(c, justify="left", style=style)

    def wrap(x):  # noqa: ANN001, ANN202
        return "\n".join(x.split("/"))

    for row in df.itertuples(index=index):
        l = [wrap(str(x)) for x in list(row)]
        table.add_row(*l)
        table.add_section()
    console.print(table)


# TODO: could this be refactored into a kbench class?
# TODO: refactor spec-to-df for entire kbench suite
@dataclass(frozen=True)
class TuningSpec:
    name: str = ""
    file: Path = Path("")
    params: list[dict] = field(default_factory=list)  # type: ignore[type-arg]
    src_path: Path = Path()
    git_sha: str = ""
    datetime: str = ""


@dataclass(repr=True)
class KbenchPKL:
    merged_df: pd.DataFrame
    tune_df: pd.DataFrame
    pkl_data: dict  # type: ignore[type-arg]
    metric: str

    def __init__(self, pickle_path, metric: str) -> None:  # noqa: ANN001
        self.pkl_data = KbenchPKL.load(pickle_path)
        self.merged_df = self.pkl_data["merged_df"].drop(
            ["name", "iters"], axis=1
        )
        # Finding the appropriate metric
        cols = list(self.merged_df.columns)
        valid_metric = metric in cols
        if not valid_metric:
            for c in cols:
                if c.lower().startswith(metric.lower()):
                    metric = c
                    valid_metric = True
                    break
        assert valid_metric, f"ERROR: metric [{metric}] is not valid!"
        self.metric = metric
        assert is_numeric_dtype(self.merged_df[metric]), (
            f"ERROR: metric [{metric}] is not numeric!"
        )
        # Setting the sort order based on the metric
        # TODO: move this dict to a global variable or singleton.
        ascending_sort = {
            "met (ms)": True,  # lower is better
            "throughput (GElems/s)": False,  # higher is better
            "DateMovement (GB/s)": False,  # higher is better
            "Arithmetic (GFLOPS/s)": False,  # higher is better
            "TheoreticalArithmetic (GFLOPS/s)": False,  # higher is better
        }[metric]

        # Set tune_df based on sort-order and merged_df
        tune_df = self.merged_df.sort_values([metric], ascending=ascending_sort)
        best_spec = tune_df.iloc[0]
        # Add ratio: current metric/best metric
        tune_df["ratio"] = tune_df[metric].div(best_spec[metric])

        self.tune_df = tune_df

    @staticmethod
    def load(path) -> dict:  # type: ignore[type-arg]  # noqa: ANN001
        f = load_pickle(path)
        for k in ["merged_df", "build_df"]:
            assert k in f
        return f


def df_round_floats(df, prec=3):  # noqa: ANN001, ANN201
    "Round values in dataframe to specified precision"
    for c in df.columns:
        if df.dtypes[c] in (np.float64, np.float32):
            df = df.round({c: prec})
    return df


def profile_results(
    pickle_path,  # noqa: ANN001
    top_percentage=0.0,  # noqa: ANN001
    ratio=False,  # noqa: ANN001
    head=-1,  # noqa: ANN001
    tail=-1,  # noqa: ANN001
    metric: str = "met (ms)",
    pivots: list[str] = [],  # noqa: B006
    verbose=False,  # noqa: ANN001
) -> TuningSpec | None:
    try:
        pkl = KbenchPKL(pickle_path=pickle_path, metric=metric)
    except:
        print(f"Invalid pkl [{pickle_path}]")
        print(LINE)
        return None
    merged_df, tune_df, pkl_data = (
        pkl.merged_df,
        pkl.tune_df,
        pkl.pkl_data,
    )
    print(f"- num entries: {len(merged_df)}")

    if top_percentage:
        idx = top_idx(tune_df[pkl.metric], top_percentage=top_percentage)
        subset = merged_df.iloc[idx]
        if verbose:
            print(f"common subset in [{top_percentage}]%")
            print(subset.to_string(index=False))
        print(LINE)

        # form a spec with most frequent values among top percentage of picks
        spec = find_common_params(subset)
    else:
        prec = 3
        # get the spec from the first pick
        print(f"- best idx: {tune_df.iloc[0]['mesh_idx']}")
        print(f"- [{metric}] worst: {tune_df.iloc[-1][metric]:.{prec}f}")
        print(f"- [{metric}] best : {tune_df.iloc[0][metric]:.{prec}f}")

        # TODO: revise this part to automatically extract pivots if not provided!
        spec = spec_to_dict(tune_df.iloc[0]["spec"])

        if not pivots:
            _, pivots = extract_pivots(
                list(tune_df["spec"]), exclude=["name", "AUTOTUNING_MODE"]
            )

        shape = "/".join([f"{p}={spec[p]}" for p in pivots])
        print(
            f"- [{metric}] worst/best ratio: {tune_df.iloc[-1]['ratio']:.2f}x [shape: {shape}]"
        )
        print(LINE)

    tune_df = df_round_floats(tune_df, prec=3)
    if ratio:
        df_to_console_table(
            tune_df,
            col_style={"ratio": "bold green"},
        )
        print(LINE)

    if head > 0:
        df_to_console_table(tune_df[:head])

    if tail > 0:
        df_to_console_table(tune_df[-1 * tail :])

    if verbose:
        # TODO: select based on pivots and cleanup the view
        print(LINE)
        merged_df = df_round_floats(merged_df, prec=3)
        df_to_console_table(merged_df)

    spec = TuningSpec(
        name=str(pkl_data.get("name", None)),
        file=Path(pkl_data.get("file", Path())),
        params=[spec],
        src_path=pickle_path,
        git_sha=str(pkl_data.get("git-revision", None)),
        datetime=str(pkl_data.get("datetime", None)),
    )
    if verbose:
        print("[Best Spec]\n")
        print(spec)
        print(LINE)
    return spec


def identical_pivot_values(x, y, pivots) -> bool:  # noqa: ANN001
    for p in pivots:
        if (p not in x) or (p not in y) or (x.get(p, None) != y.get(p, None)):
            print(f"ERROR: FAILED assert on pivot {p}: [{x[p]}] vs. [{y[p]}]")
            return False
    return True


def yaml_to_tuning_spec(yaml_path: Path) -> list[TuningSpec]:
    spec = Spec.load_yaml(yaml_path)
    tuning_specs = []
    for s in spec:
        ts = TuningSpec(
            name=s.name,
            file=s.file,
            params=[spec_to_dict(str(s))],
            src_path=yaml_path,
            git_sha="",
        )
        tuning_specs.append(ts)
    return tuning_specs


def spec_to_tuning_spec(spec: Spec) -> list[TuningSpec]:
    tuning_specs = []
    for s in spec:
        ts = TuningSpec(
            name=s.name,
            file=s.file,
            params=[spec_to_dict(str(s))],
            src_path=Path(),
            git_sha="",
        )
        tuning_specs.append(ts)
    return tuning_specs


def diff_baseline(
    files,  # noqa: ANN001
    metric: str,
    pivots: list = [],  # type: ignore[type-arg]  # noqa: B006
    head: int = -1,
    verbose: bool = False,
) -> None:
    base_pkl = KbenchPKL(files[0], metric=metric)
    metric = base_pkl.metric
    tune_df_base = base_pkl.tune_df

    base_dict = spec_to_dict(str(tune_df_base.iloc[0]["spec"]))
    if verbose:
        print("base-config", base_dict)
        print(LINE)

    # Find the common pivots between all pkl's if none specified
    if not pivots:
        _, pivots = extract_pivots(
            list(tune_df_base["spec"]), exclude=["name", "AUTOTUNING_MODE"]
        )
        for f in files[1:]:
            pkl = KbenchPKL(f, metric=metric)
            _, non_pivot_columns = extract_pivots(
                list(pkl.tune_df["spec"]), exclude=["name", "AUTOTUNING_MODE"]
            )
            pivots = list_intersection(pivots, non_pivot_columns)

    shape = "/".join([f"{p}={base_dict[p]}" for p in pivots])
    for i, f in enumerate(files[1:]):
        tune_df = KbenchPKL(f, metric=metric).tune_df
        spec_dict = spec_to_dict(str(tune_df.iloc[0]["spec"]))
        assert base_dict["name"] == spec_dict["name"]
        assert identical_pivot_values(base_dict, spec_dict, pivots=pivots)

        if verbose:
            print(f"config [{i}]", spec_dict)
            print(LINE)

        base_metric = tune_df_base.iloc[0][metric]
        prec = 3

        num_rows = min(len(tune_df[:head]), len(tune_df))
        for j in range(num_rows):
            current_metric = tune_df.iloc[j][metric]
            # TODO: fix the following and check the accuracy issues.
            metric_ratio = round(current_metric / base_metric, prec)
            print(
                f"[{i}][shape:{shape}][metric:{metric}]: {metric_ratio:.{prec}f} (ratio current/baseline = {current_metric:.{prec}f} / {base_metric:.{prec}f})"
            )
            d = {
                "shape": [shape],
                "metric": metric,
                "best_tuning_metric": [round(current_metric, prec)],
                "baseline_metric": [round(base_metric, prec)],
                "ratio": [metric_ratio],
            }
            shape_path = shape.replace("/", "_")
            pd.DataFrame.from_dict(d).to_csv(f"{shape_path}.csv", index=False)
            print(LINE)


def yaml_reference_handling(s: str):  # noqa: ANN201
    # all of them should be uniq
    ref_pattern = re.compile(r"('<<'[\s]*:[\s]*'*)([^']*)'")
    refs = re.findall(ref_pattern, s)
    assert len(set(refs)) == 1
    ref_name = refs[0][1].lstrip("*")

    s = re.sub(ref_pattern, r"<<: \2", s)
    refline_pattern = re.compile(rf"'([^:]*):[\s]*(&[\s]*{ref_name})':")
    refline = re.findall(refline_pattern, s)
    assert len(refline) == 1
    new_refline = f"{refline[0][0]}: &{ref_name}"

    s = re.sub(refline_pattern, new_refline, s)
    return s


def codegen_yaml(specs: list[TuningSpec], output_path: Path) -> None:
    names = [s.name for s in specs]
    files = [s.file for s in specs]
    # "name" and "file" in all specs should be identical.
    assert len(set(names)) == 1
    assert len(set(files)) == 1

    # collect first set of parameters (i.e., "params[0]") from all specs.
    params = [s.params[0] for s in specs]
    df = pd.DataFrame(params)
    pivots, non_pivots = extract_pivots_df(df, exclude=[])

    # remove common columns (non-pivots) to get unique rows of data
    uniq_rows = df.drop(columns=non_pivots)
    # drop duplicates in common columns (non-pivots)
    common = df[non_pivots].drop_duplicates()

    # convert to list of dictionaries
    uniq_params = list(uniq_rows.T.to_dict().values())
    common_params = list(common.T.to_dict().values())

    common_key = "common"
    b = {"<<": f"*{common_key}"}
    for i, p in enumerate(uniq_params):
        uniq_params[i] = dict(b, **p)

    obj = {
        "name": specs[0].name,
        "file": str(specs[0].file),
        f"common: &{common_key}": common_params,
        "params": uniq_params,
    }

    # TODO: revise this dump function to directly write yaml merge-key values.
    # TODO: remove or revise 'yaml_reference_handling'.
    yaml_str = yaml.dump(obj, sort_keys=False)
    yaml_str = HEADER + yaml_reference_handling(yaml_str)

    output_yaml = Path(output_path).with_suffix(".yaml")
    with open(output_yaml, "w") as f:
        f.write(yaml_str)

    print(f"common columns: {non_pivots}")
    print(f"pivot columns: {pivots}")
    print(f"num entries: [{len(uniq_rows)}]")
    print(f"wrote results to [{output_yaml}]")
    print(LINE)


def list_intersection(a: list, b: list) -> list:  # type: ignore[type-arg]
    return [x for x in a if x in b]


def merge_specs(
    spec_list: list[Spec],
    pivots: list[str] | None = None,
    sort_pivots: list[str] | None = None,
) -> pd.DataFrame:
    """
    Merges a list of Spec objects into a single pandas DataFrame, join on specified pivot columns.

    Args:
        spec_list: List of Spec objects to merge.
        pivots: List of column names to use as pivots for merging. If None, uses the intersection of all columns in the DataFrames.
        sort_pivots: List of column names to sort the merged DataFrame by. If None, uses the pivots.

    Returns:
        pd.DataFrame: Merged DataFrame containing the specified pivots and a 'spec' column with the corresponding Spec objects.
    """
    df_list = []
    for spec in spec_list:
        df_list += [specs_to_df([str(s) for s in spec])]

    # limit to a subset of pivots, if not select them all
    if not pivots:
        pivots = list(df_list[0].columns)
        for df in df_list[1:]:
            pivots = list_intersection(pivots, list(df.columns))

    # assert all pivots are present in columns of all df's
    for i, df in enumerate(df_list):
        assert np.all([p in df.columns for p in pivots])
        # ignore non-pivot columns
        df = df[pivots]
        df["spec"] = spec_list[i]
        df_list[i] = df

    # merge all df's
    df_merged = df_list[0]
    for df in df_list[1:]:
        df_merged = pd.merge(df_merged, df, on=pivots, how="outer")

    # Fill 'spec' with 'spec_x' if 'spec_x' is not NaN, otherwise with 'spec_y'.
    # Then, drop 'spec_x' and 'spec_y'.
    df_merged["spec"] = np.where(
        df_merged["spec_x"].notna(), df_merged["spec_x"], df_merged["spec_y"]
    )
    df_merged = df_merged.drop(["spec_x", "spec_y"], axis=1)

    if not sort_pivots:
        sort_pivots = pivots
    assert set(sort_pivots).issubset(set(pivots))
    df_merged = (
        df_merged.drop_duplicates(sort_pivots)
        .sort_values(by=sort_pivots)
        .reset_index(drop=True)
    )
    return df_merged


class ComplexParamList(click.Option):
    """Complex parameter list
    Example:
        --pivot=[M] --pivot=[N] --pivot=[K] is equivalent to --pivot=[M,N,K] and vice versa.
    """

    def type_cast_value(self, ctx, value_in):  # noqa: ANN001, ANN201
        """DO NOT REMOVE this function, it is called from ctx in click."""
        p = []
        assert isinstance(value_in, list)
        for v in value_in:
            p.extend(self.parse(v))
        return p

    @staticmethod
    def parse(value) -> list:  # type: ignore[type-arg]  # noqa: ANN001
        try:
            return ast.literal_eval(value)
        except:
            # case of [x, y, z, ...]
            if value.startswith("[") and value.endswith("]"):
                return value[1:-1].split(",")
            # case of single value x
            elif (
                not value.startswith("[")
                and not value.endswith("]")
                and "," not in value
            ):
                return [value]
            else:
                raise click.BadParameter(value)  # noqa: B904


def draw_heatmap(df: pd.DataFrame, img: str = "correlation.png") -> None:
    column_names = df.columns
    width_px = 1600
    height_px = 1200
    layout = go.Layout(autosize=True, width=width_px, height=height_px)
    fig = go.Figure(
        data=go.Heatmap(
            z=df,
            x=column_names,
            y=column_names,
            text=df.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 20},
            hoverongaps=False,
            colorscale="Blues",
        ),
        layout=layout,
    )
    font_size_pt = 20
    fig.update_layout(
        #         font_family=cfg.font_family,
        #         title_font_family=cfg.title_font_family,
        font_size=font_size_pt,
    )
    fig.write_image(img)


def correlation_analysis(
    files,  # noqa: ANN001
    output_path,  # noqa: ANN001
    metric: str = "met (ms)",
    verbose: bool = False,
) -> None:
    pkl_list = []
    for pkl in files:
        try:
            pkl_list.append(KbenchPKL(pickle_path=pkl, metric=metric))
        except:
            if verbose:
                print(f"invalid pkl: [{pkl}]")
    print(LINE)

    # Note: it is crucial to ignore index for later merging the two sets.
    merged_df = pd.concat([p.merged_df for p in pkl_list], ignore_index=True)
    specs_df = specs_to_df(list(merged_df["spec"]))
    agg_df = pd.merge(merged_df, specs_df, left_index=True, right_index=True)

    rm_list = ["mesh_idx", "spec", "name"]
    metric_col = "met (ms)"
    cols = []
    agg_df[metric_col] = agg_df[metric_col].astype(float)
    for c in list(agg_df.columns):
        if c not in rm_list:
            try:
                agg_df[c] = agg_df[c].astype(float)
                cols.append(c)
            except:
                pass

    corr_method = "kendall"
    c = pd.DataFrame(agg_df[cols].corr(corr_method))

    c_abs = c[metric_col].fillna(0).abs().sort_values(ascending=False).round(2)
    c_abs = c_abs.to_frame(name=f"Sorted Absolute Correlation: {metric_col}")

    df_to_console_table(c_abs, index=True)
    df_to_console_table(round(c, 2), index=True)

    draw_heatmap(c, "correlation.png")


help_str = "Profile kbench output pickle"


@click.command(
    help=help_str,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "-help", "--help"]},
)
@click.option(
    "--output",
    "-o",
    "output_path",
    default="output.mojo",
    help="Path to output file.",
)
@click.option(
    "--top",
    "-t",
    default=0.0,
    help="Form a new spec from frequent values of each param from top percent.",
    multiple=False,
)
@click.option(
    "--ratio",
    "-r",
    is_flag=True,
    default=False,
    help="Print the running time ratio of each entry to the best entry.",
)
@click.option(
    "--head",
    default=-1,
    help="The number of elements at head to print (sorted by running time).",
    multiple=False,
)
@click.option(
    "--tail",
    default=-1,
    help="The number of elements at tail to print (sorted by running time).",
    multiple=False,
)
@click.option(
    "--diff",
    "-d",
    is_flag=True,
    default=False,
    help="Show the difference between best running time of multiple pkl's, "
    "first one as baseline. For example, pass current results "
    "(first pkl as baseline) followed by auto-tuning results.",
)
@click.option(
    "--metric",
    "-m",
    default="met (ms)",
    help="Specify the profiling metric (default='met (ms)').",
    multiple=False,
)
@click.option(
    "--pivots",
    "-p",
    cls=ComplexParamList,
    default=[],
    help="Specify the pivots to select the values.",
    multiple=True,
)
@click.option(
    "--sort-pivots",
    cls=ComplexParamList,
    default=[],
    help="Specify the pivots to sort values.",
    multiple=True,
)
@click.option(
    "--merge",
    is_flag=True,
    default=False,
    help="Merge the first incoming pkl/yamls with the first one.",
)
@click.option(
    "--query",
    default=None,
    help="Query the yaml. Example: kprofile --merge *yaml --query 'M>64 and K<1024' ",
)
@click.option(
    "--correlation",
    "-c",
    is_flag=True,
    default=False,
    help="Correlation analysis.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Print all the (unsorted) entries from pkl.",
)
@click.argument("files", nargs=-1, type=click.UNPROCESSED)
def cli(
    files,  # noqa: ANN001
    output_path,  # noqa: ANN001
    top,  # noqa: ANN001
    ratio,  # noqa: ANN001
    head,  # noqa: ANN001
    tail,  # noqa: ANN001
    diff,  # noqa: ANN001
    metric,  # noqa: ANN001
    pivots,  # noqa: ANN001
    sort_pivots,  # noqa: ANN001
    merge: bool,
    query: str,
    correlation: bool,
    verbose,  # noqa: ANN001
) -> bool:
    if not verbose:
        sys.tracebacklimit = 1

    assert files, "No files provided"

    if diff:
        assert len(files) > 1, "Provide at least two pkl's for --diff option."

    if verbose:
        print(f"top_percentage: [{top}]")
        print(LINE)

    top_percentage = float(top) if top else 0

    # TODO: refactor profile, diff, merge, correlation into separate commands

    if merge:
        # All files should be .yaml
        assert np.all([Path(path).suffix == ".yaml" for path in files])
        spec_list = [Spec.load_yaml(Path(path)) for path in files]
        merged_df = merge_specs(
            spec_list, pivots=pivots, sort_pivots=sort_pivots
        )

        if query:
            print(f"Applying query: [{query}]")
            merged_df = merged_df.query(query).reset_index(drop=True)
        if verbose:
            print(merged_df.loc[:, merged_df.columns != "spec"].to_string())
            print(LINE)
        if len(merged_df) == 0:
            raise ValueError(
                "Query resulted in empty dataset. No data matches the specified query or filters."
            )

        specs = spec_to_tuning_spec(merged_df["spec"])

        codegen_yaml(specs, output_path=output_path)

    elif diff:
        if head == -1:
            head = 1
        diff_baseline(
            files, metric=metric, pivots=pivots, head=head, verbose=verbose
        )
    else:
        specs = []
        invalid_pkls = []
        for path in files:
            suffix = Path(path).suffix
            if suffix == ".yaml":
                tuning_specs = yaml_to_tuning_spec(Path(path))
                specs.extend(tuning_specs)
                df = pd.DataFrame([ts.params[0] for ts in tuning_specs])
                print(f"[{path}]")
                print(df.to_string())
                print(LINE)
            elif suffix == ".pkl":
                top_spec = profile_results(
                    pickle_path=path,
                    top_percentage=top_percentage,
                    ratio=ratio,
                    head=head,
                    tail=tail,
                    metric=metric,
                    pivots=pivots,
                    verbose=verbose,
                )
                if top_spec:
                    specs += [top_spec]
                else:
                    invalid_pkls += [path]
            else:
                print(f"unsupported input: [{path}]")

        codegen_yaml(specs=specs, output_path=output_path)

        print(
            f"num invalid pkls: {len(invalid_pkls)} out of num-files: {len(files)}"
        )
        if verbose:
            for i, pkl in enumerate(invalid_pkls):
                print(f"invalid pkl [{i}]: [{pkl}]")

        print(LINE)

    if correlation:
        correlation_analysis(files, output_path, verbose=verbose)

    return True


def main() -> None:
    cli()


if __name__ == "__main__":
    if directory := os.environ.get("BUILD_WORKING_DIRECTORY"):
        os.chdir(directory)
    main()
