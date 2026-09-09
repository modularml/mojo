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

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd


# Function to get role name based on role value
def get_role_name(role_value: int) -> str:
    base_roles = ["load", "scheduler", "compute", "epilogue"]
    if role_value < len(base_roles):
        return base_roles[role_value]
    else:
        # For values > 3, return epilogue1, epilogue2, etc.
        epilogue_num = role_value - 3
        return f"epilogue{epilogue_num}"


def plot_sm(
    df: pd.DataFrame,
    grouped_dataframes: dict[int, pd.DataFrame],
    sm_id: int,
    save_path: str | None = None,
) -> None:
    # Get data for the specific SM
    if sm_id not in grouped_dataframes:
        print(f"SM ID {sm_id} not found in data")
        return

    df_block = grouped_dataframes[sm_id]

    # Normalize time values - make the minimum start time = 0
    all_start_times = df["time_start"].values
    all_end_times = df["time_end"].values
    min_time = min(all_start_times.min(), all_end_times.min())

    # Create visualization for the SM
    _, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = ["red", "blue", "green", "orange"]  # Colors for each role

    # Keep track of which roles we've already labeled for the legend
    labeled_roles = set()

    # Plot each role as a horizontal bar
    for i, row in df_block.iterrows():
        role_idx = int(row["role"])
        role_name = get_role_name(role_idx)

        # Use color cycling for roles beyond the base 4
        color_idx = role_idx % len(colors)
        color = colors[color_idx]

        start_norm = row["time_start"] - min_time
        duration = row["time_end"] - row["time_start"]

        # Only add label if this role hasn't been labeled yet
        label = role_name if role_name not in labeled_roles else ""
        if role_name not in labeled_roles:
            labeled_roles.add(role_name)

        ax.barh(
            i, duration, left=start_norm, color=color, alpha=0.7, label=label
        )

    ax.set_title(f"SM {sm_id} - Role Timeline")
    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("Execution Order")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()


def plot_all_sms(
    df: pd.DataFrame,
    grouped_dataframes: dict[int, pd.DataFrame],
    save_path: str | None = None,
) -> None:
    # Create mapping from unique (block_x, block_y) pairs to numeric IDs
    unique_sms = list(grouped_dataframes.keys())
    sm_id_mapping = {sm: idx for idx, sm in enumerate(unique_sms)}

    # Add SM ID column to the original dataframe
    df["linear_sm_id"] = df.apply(
        lambda row: sm_id_mapping[row["sm_id"]],
        axis=1,
    )

    # Drop the original block index columns since we now have sm_id
    df = df.drop(["sm_id"], axis=1)

    # Normalize time values - make the minimum time = 0
    min_time = min(df["time_start"].min(), df["time_end"].min())
    df["time_start"] = df["time_start"] - min_time
    df["time_end"] = df["time_end"] - min_time

    roles = ["load", "scheduler", "compute", "epilogue"]

    # Create comprehensive visualization showing all SMs
    _, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Define colors and markers for each role
    colors = ["blue", "red", "green", "orange"]  # Colors for each role
    markers = ["o", "s", "^", "D"]  # Different markers for each role
    role_names = ["load", "scheduler", "compute", "epilogue"]

    # Sort dataframe by sm_id and time_start for better visualization
    df_sorted = df.sort_values(["linear_sm_id", "time_start"])

    # Plot each role execution as points
    for role_idx in range(len(roles)):
        role_data = df_sorted[df_sorted["role"] == role_idx]

        if len(role_data) > 0:
            # Plot start times
            ax.scatter(
                role_data["time_start"],
                role_data["linear_sm_id"],
                color=colors[role_idx],
                marker=markers[role_idx],
                s=30,
                alpha=0.7,
                label=f"{role_names[role_idx]} start",
            )

            # Plot end times with 'x' marker
            ax.scatter(
                role_data["time_end"],
                role_data["linear_sm_id"],
                color=colors[role_idx],
                marker="x",
                s=30,
                alpha=0.7,
                label=f"{role_names[role_idx]} end",
            )

            # Draw lines connecting start and end times
            for _, row in role_data.iterrows():
                ax.plot(
                    [row["time_start"], row["time_end"]],
                    [row["linear_sm_id"], row["linear_sm_id"]],
                    color=colors[role_idx],
                    alpha=0.5,
                    linewidth=2,
                )

    # Customize the plot
    ax.set_xlabel("Time (normalized)")
    ax.set_ylabel("Linear SM ID")
    ax.set_title("SM Execution Timeline - All Roles")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Set y-axis to show all SM IDs
    unique_sm_ids = sorted(df["linear_sm_id"].unique())
    ax.set_yticks(
        unique_sm_ids[:: max(1, len(unique_sm_ids) // 20)]
    )  # Show every nth tick if too many SMs

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize SM execution timeline"
    )
    parser.add_argument("filename", help="CSV file to process")
    parser.add_argument(
        "--sm-id",
        type=int,
        help="SM ID to display (if not provided, displays all SMs)",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save figure to specified file path (e.g., output.png)",
    )

    parser.add_argument(
        "--list-sms",
        action="store_true",
        help="List all SM IDs",
    )

    args = parser.parse_args()

    list_sms = args.list_sms

    filename = args.filename
    sm_id = args.sm_id  # None if not provided, otherwise the specific SM ID
    save_path = args.save

    show_all = sm_id is None

    df = pd.read_csv(filename + ".csv")

    # Check column names and strip whitespace
    df.columns = df.columns.str.strip()

    # Remove rows where all values are 0
    df_filtered = df[(df != 0).any(axis=1)]
    df_filtered = df_filtered.reset_index(drop=True)

    if list_sms:
        print(df_filtered["sm_id"].unique())
        sys.exit(0)

    # Group by block_idx_x and block_idx_y, then sort by role and split into separate dataframes
    grouped = df_filtered.groupby(["sm_id"])

    # Create a dictionary of dataframes, one for each group
    dataframes = {}
    for group_sm_id, group in grouped:
        # Sort each group by role and reset index
        sorted_group = group.sort_values("role").reset_index(drop=True)
        dataframes[group_sm_id[0]] = sorted_group

    if show_all:
        plot_all_sms(df_filtered, dataframes, save_path)
    else:
        plot_sm(df_filtered, dataframes, sm_id, save_path)
