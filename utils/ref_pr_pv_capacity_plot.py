from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.ticker import FuncFormatter

DEFAULT_EXHIBIT_2_URL = (
    "https://energia.pr.gov/wp-content/uploads/sites/7/2026/03/"
    "2026.03.06_Exhibit_2_Distributed_Generation_Data_October_to_December_2025.xlsx"
)
DEFAULT_SHEET_NAME = "Monthly "
DEFAULT_QUARTER_START = "2017-01-01"
DEFAULT_QUARTER_END = "2025-12-31"

HEADER_PERIOD_CANDIDATES = (("mes", "ano"), ("periodo",))
CAPACITY_COLUMN_CANDIDATES = (
    (("capacidad", "clientes", "gd", "registrados"), ("promedio", "cliente")),
    (("capacidad", "registrados"), ("promedio", "cliente")),
)
CLIENT_COLUMN_CANDIDATES = (
    (("clientes", "registrados", "fotovoltaico"), ("capacidad", "promedio")),
    (("promedio", "clientes", "registrados"), ("capacidad", "cliente")),
)


def resolve_project_root(start: Path | None = None) -> Path:
    """Find repository root from an arbitrary file or working directory."""

    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    markers = ("project_rules.md", ".git")
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return current


PROJECT_ROOT = resolve_project_root(Path(__file__).resolve())
EXCEL_PATH = PROJECT_ROOT / "data" / "tabular" / Path(urlparse(DEFAULT_EXHIBIT_2_URL).path).name
OUTPUT_STEM = PROJECT_ROOT / "outputs" / "figures" / "pr_pv_capacity_quarterly_2017_2025"
CAPACITY_VARIANT_CLIENTS_STEM = PROJECT_ROOT / "outputs" / "figures" / "pr_pv_capacity_vs_clients"
CAPACITY_VARIANT_GROWTH_STEM = PROJECT_ROOT / "outputs" / "figures" / "pr_pv_capacity_vs_growth"
CAPACITY_SOURCE_NOTE = "Sources: IEEFA (2024); NEPR Exhibit 2 (Mar. 2026)."
CAPACITY_MAIN_COLOR = "#0f766e"
CAPACITY_FILL_COLOR = "#99f6e4"
CAPACITY_ACCENT_COLOR = "#ea580c"
CLIENT_COLOR = "#2563eb"
POSTER_TEXT_COLOR = "#0f172a"
POSTER_MUTED_TEXT_COLOR = "#475569"


def _normalize_label(value: object) -> str:
    """Normalize workbook labels for resilient column matching."""

    if value is None:
        return ""
    text = str(value).strip().casefold()
    if not text or text == "nan":
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text


def _tokens_match(label: str, include_tokens: tuple[str, ...], exclude_tokens: tuple[str, ...] = ()) -> bool:
    """Return True when all include tokens and no exclude tokens are present."""

    if not label:
        return False
    tokens = set(label.split())
    return all(token in tokens for token in include_tokens) and not any(token in tokens for token in exclude_tokens)


def is_remote_excel_source(path_or_url: str | Path) -> bool:
    """Return True when the workbook source points to HTTP(S)."""

    text = str(path_or_url)
    return text.startswith(("http://", "https://"))


def download_workbook(
    url: str = DEFAULT_EXHIBIT_2_URL,
    destination: Path = EXCEL_PATH,
    force: bool = False,
    timeout_seconds: int = 180,
) -> Path:
    """Cache a remote NEPR workbook locally for repeatable notebook runs."""

    target_path = destination if destination.is_absolute() else PROJECT_ROOT / destination
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not force:
        return target_path

    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    return target_path


def _resolve_excel_source(
    path_or_url: str | Path = DEFAULT_EXHIBIT_2_URL,
    cache_remote: bool = True,
    force_download: bool = False,
) -> Path:
    """Resolve workbook source to a local path when possible."""

    if is_remote_excel_source(path_or_url):
        if not cache_remote:
            raise ValueError("Remote workbook sources must be cached locally before reading.")
        return download_workbook(str(path_or_url), destination=EXCEL_PATH, force=force_download)

    path = Path(path_or_url)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _find_header_row(frame: pd.DataFrame) -> int:
    """Locate the row that contains the real workbook field labels."""

    for row_index, row in frame.iterrows():
        normalized_labels = [_normalize_label(value) for value in row.tolist()]
        has_period = any(
            _tokens_match(label, period_tokens)
            for label in normalized_labels
            for period_tokens in HEADER_PERIOD_CANDIDATES
        )
        has_capacity = any("capacidad" in label for label in normalized_labels)
        if has_period and has_capacity:
            return int(row_index)

    raise RuntimeError("Could not locate the NEPR workbook header row.")


def _find_column_index(
    header_labels: dict[int, str],
    candidate_specs: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
    label: str,
) -> int:
    """Match one target workbook column by normalized token pattern."""

    for include_tokens, exclude_tokens in candidate_specs:
        for column_index, normalized_header in header_labels.items():
            if _tokens_match(normalized_header, include_tokens, exclude_tokens):
                return column_index

    raise KeyError(f"Could not locate the '{label}' column in the NEPR workbook.")


def inspect_workbook(
    path_or_url: str | Path = DEFAULT_EXHIBIT_2_URL,
    sheet_name: str = DEFAULT_SHEET_NAME,
    cache_remote: bool = True,
    force_download: bool = False,
) -> dict[str, object]:
    """Return light workbook metadata for quick debugging and notebook reporting."""

    source_path = _resolve_excel_source(path_or_url, cache_remote=cache_remote, force_download=force_download)
    raw = pd.read_excel(source_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    header_row_index = _find_header_row(raw)
    header_labels = [_normalize_label(value) for value in raw.iloc[header_row_index].tolist()]
    period_candidates = pd.to_datetime(raw.iloc[:, 1], errors="coerce").dropna()
    return {
        "source_path": source_path,
        "sheet_name": sheet_name,
        "header_row_index": header_row_index,
        "shape": raw.shape,
        "normalized_header_preview": header_labels[:12],
        "period_min": period_candidates.min() if not period_candidates.empty else None,
        "period_max": period_candidates.max() if not period_candidates.empty else None,
    }


def load_monthly_capacity_records(
    path_or_url: str | Path = DEFAULT_EXHIBIT_2_URL,
    sheet_name: str = DEFAULT_SHEET_NAME,
    cache_remote: bool = True,
    force_download: bool = False,
) -> pd.DataFrame:
    """Return month-level registered PV capacity and client counts from Exhibit 2."""

    source_path = _resolve_excel_source(path_or_url, cache_remote=cache_remote, force_download=force_download)
    raw = pd.read_excel(source_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    header_row_index = _find_header_row(raw)
    header_labels = {
        int(column_index): _normalize_label(value)
        for column_index, value in raw.iloc[header_row_index].items()
    }

    period_column_index = _find_column_index(
        header_labels,
        tuple((tokens, ()) for tokens in HEADER_PERIOD_CANDIDATES),
        label="period",
    )
    capacity_column_index = _find_column_index(
        header_labels,
        CAPACITY_COLUMN_CANDIDATES,
        label="capacity_kw",
    )
    client_column_index = _find_column_index(
        header_labels,
        CLIENT_COLUMN_CANDIDATES,
        label="client_count",
    )

    records = raw.iloc[header_row_index + 1 :].copy()
    data = pd.DataFrame(
        {
            "period": pd.to_datetime(records.iloc[:, period_column_index], errors="coerce"),
            "capacity_kw": pd.to_numeric(records.iloc[:, capacity_column_index], errors="coerce"),
            "client_count": pd.to_numeric(records.iloc[:, client_column_index], errors="coerce"),
        }
    )
    data = data.dropna(subset=["period", "capacity_kw"]).copy()
    data = data[data["period"].dt.day.eq(1)].sort_values("period").reset_index(drop=True)
    data = data[data["capacity_kw"] >= 0].copy()
    data.loc[:, "client_count"] = data["client_count"].round().astype("Int64")
    data.loc[:, "capacity_mw"] = data["capacity_kw"] / 1000.0
    data.loc[:, "source_path"] = source_path.as_posix()
    data.loc[:, "sheet_name"] = sheet_name.strip() or sheet_name
    return data


def aggregate_quarterly_capacity_records(
    monthly_data: pd.DataFrame,
    start: str | pd.Timestamp = DEFAULT_QUARTER_START,
    end: str | pd.Timestamp = DEFAULT_QUARTER_END,
) -> pd.DataFrame:
    """Convert month-level installed capacity totals to quarter-end snapshots."""

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    filtered = monthly_data.copy()
    filtered = filtered[(filtered["period"] >= start_ts) & (filtered["period"] <= end_ts)].copy()
    if filtered.empty:
        raise RuntimeError("No monthly capacity rows were available inside the requested quarter range.")

    filtered.loc[:, "quarter"] = filtered["period"].dt.to_period("Q")
    quarterly = filtered.sort_values("period").groupby("quarter", as_index=False).tail(1).copy()
    quarterly = quarterly.sort_values("period").reset_index(drop=True)

    quarterly.loc[:, "quarter_start"] = quarterly["quarter"].dt.to_timestamp(how="start")
    quarterly.loc[:, "quarter_end"] = quarterly["quarter"].dt.to_timestamp(how="end").dt.normalize()
    quarterly.loc[:, "quarter_key"] = quarterly["quarter"].astype(str).str.replace("Q", "-Q", regex=False)
    quarterly.loc[:, "quarter_label"] = quarterly["quarter"].map(lambda period: f"Q{period.quarter} {period.year}")
    quarterly.loc[:, "capacity_delta_mw"] = quarterly["capacity_mw"].diff()
    quarterly.loc[:, "capacity_growth_pct"] = quarterly["capacity_mw"].pct_change() * 100.0
    quarterly.loc[:, "client_delta"] = quarterly["client_count"].astype(float).diff().round().astype("Int64")
    quarterly.loc[:, "is_year_end"] = quarterly["quarter"].map(lambda period: period.quarter == 4)

    return quarterly[
        [
            "quarter",
            "quarter_key",
            "quarter_label",
            "quarter_start",
            "quarter_end",
            "period",
            "capacity_kw",
            "capacity_mw",
            "capacity_delta_mw",
            "capacity_growth_pct",
            "client_count",
            "client_delta",
            "is_year_end",
            "source_path",
            "sheet_name",
        ]
    ].reset_index(drop=True)


def _style_capacity_axis(axis: plt.Axes, ylabel: str) -> None:
    """Apply a consistent poster-forward axis style."""

    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#334155")
    axis.grid(axis="y", color="#cbd5e1", linewidth=0.85, alpha=0.9)
    axis.set_axisbelow(True)
    axis.set_ylabel(ylabel, fontsize=12, color=POSTER_TEXT_COLOR)
    axis.tick_params(axis="both", colors="#334155", labelsize=10.5)


def _set_year_ticks(axis: plt.Axes, plot_data: pd.DataFrame, x_column: str) -> None:
    """Show one x tick per year at the fourth quarter when available."""

    q4_ticks = plot_data.loc[plot_data["is_year_end"], x_column]
    if q4_ticks.empty:
        q4_ticks = plot_data[x_column]
    axis.set_xticks(q4_ticks)
    axis.set_xticklabels([str(pd.Timestamp(value).year) for value in q4_ticks])


def _year_end_rows(plot_data: pd.DataFrame) -> pd.DataFrame:
    """Return only year-end quarter rows for light labeling."""

    year_end_rows = plot_data[plot_data["is_year_end"]].copy()
    return year_end_rows if not year_end_rows.empty else plot_data.iloc[[-1]].copy()


def _apply_chart_title(figure: plt.Figure, title: str, subtitle: str) -> None:
    """Apply a two-line poster title block."""

    figure.suptitle(title, fontsize=19, fontweight="bold", color=POSTER_TEXT_COLOR, x=0.125, ha="left")
    figure.text(0.125, 0.93, subtitle, fontsize=10.8, color=POSTER_MUTED_TEXT_COLOR, ha="left")


def _annotate_growth_note(axis: plt.Axes, first_row: pd.Series, absolute_growth_mw: float, growth_ratio: float | None) -> None:
    """Place a compact summary note in the top-left plot whitespace."""

    note_lines = [f"+{absolute_growth_mw:,.0f} MW since {first_row['quarter_key']}"]
    if growth_ratio is not None:
        note_lines.append(f"{growth_ratio:.0%} capacity growth")

    axis.text(
        0.02,
        0.95,
        "\n".join(note_lines),
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=10.6,
        color="#134e4a",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ecfeff", edgecolor="#99f6e4", linewidth=1.0),
    )


def _annotate_year_end_capacity_labels(axis: plt.Axes, year_end_rows: pd.DataFrame) -> None:
    """Label only Q4 markers with installed capacity values."""

    for _, row in year_end_rows.iterrows():
        axis.annotate(
            f"{row['capacity_mw']:,.0f}",
            xy=(row["quarter_end"], row["capacity_mw"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.7,
            color=CAPACITY_MAIN_COLOR,
            fontweight="semibold",
        )


def load_quarterly_capacity_records(
    path_or_url: str | Path = DEFAULT_EXHIBIT_2_URL,
    start: str | pd.Timestamp = DEFAULT_QUARTER_START,
    end: str | pd.Timestamp = DEFAULT_QUARTER_END,
    sheet_name: str = DEFAULT_SHEET_NAME,
    cache_remote: bool = True,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load and aggregate Exhibit 2 monthly data into quarter-end poster rows."""

    monthly_data = load_monthly_capacity_records(
        path_or_url=path_or_url,
        sheet_name=sheet_name,
        cache_remote=cache_remote,
        force_download=force_download,
    )
    return aggregate_quarterly_capacity_records(monthly_data, start=start, end=end)


def build_poster_capacity_figure(
    quarterly_data: pd.DataFrame,
    title: str = "Puerto Rico Grid-Connected PV Capacity",
    subtitle: str | None = None,
) -> plt.Figure:
    """Create a poster-oriented quarter-end step chart for installed PV capacity."""

    plot_data = quarterly_data.sort_values("quarter_end").reset_index(drop=True)
    first_row = plot_data.iloc[0]
    last_row = plot_data.iloc[-1]
    starting_capacity_mw = float(first_row["capacity_mw"])
    final_capacity_mw = float(last_row["capacity_mw"])
    absolute_growth_mw = final_capacity_mw - starting_capacity_mw
    growth_ratio = (absolute_growth_mw / starting_capacity_mw) if starting_capacity_mw else None

    figure, axis = plt.subplots(figsize=(14.5, 7.1), constrained_layout=True)
    x_values = plot_data["quarter_end"]
    y_values = plot_data["capacity_mw"]
    year_end_rows = _year_end_rows(plot_data)

    axis.fill_between(x_values, y_values, step="post", color=CAPACITY_FILL_COLOR, alpha=0.88)
    axis.step(x_values, y_values, where="post", color=CAPACITY_MAIN_COLOR, linewidth=3.4)
    axis.plot(
        x_values,
        y_values,
        linestyle="none",
        marker="o",
        markersize=4.8,
        markerfacecolor="#f8fafc",
        markeredgewidth=1.0,
        markeredgecolor="#134e4a",
    )

    _style_capacity_axis(axis, "Quarter-end installed capacity (MW)")
    axis.set_ylim(0, float(y_values.max()) * 1.18)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))

    _set_year_ticks(axis, plot_data, "quarter_end")

    subtitle_text = subtitle or CAPACITY_SOURCE_NOTE
    _apply_chart_title(figure, title, subtitle_text)
    _annotate_growth_note(axis, first_row, absolute_growth_mw, growth_ratio)
    _annotate_year_end_capacity_labels(axis, year_end_rows)

    return figure


def build_capacity_clients_bar_figure(
    quarterly_data: pd.DataFrame,
    title: str = "Puerto Rico PV Capacity and Registered Clients",
    subtitle: str | None = None,
) -> plt.Figure:
    """Create a dual-axis clustered bar chart for capacity and client counts."""

    plot_data = quarterly_data.sort_values("quarter_end").reset_index(drop=True)
    x_positions = np.arange(len(plot_data), dtype=float)
    width = 0.38

    figure, axis_left = plt.subplots(figsize=(15.2, 7.2), constrained_layout=True)
    axis_right = axis_left.twinx()

    left_bars = axis_left.bar(
        x_positions - width / 2,
        plot_data["capacity_mw"],
        width=width,
        color=CAPACITY_MAIN_COLOR,
        alpha=0.92,
        label="Installed capacity (MW)",
    )
    right_bars = axis_right.bar(
        x_positions + width / 2,
        plot_data["client_count"].astype(float),
        width=width,
        color=CLIENT_COLOR,
        alpha=0.72,
        label="Registered clients",
    )

    _style_capacity_axis(axis_left, "Installed capacity (MW)")
    axis_left.set_xlim(-1.0, len(plot_data))
    axis_left.set_ylim(0, float(plot_data["capacity_mw"].max()) * 1.16)
    axis_left.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))

    axis_right.spines["top"].set_visible(False)
    axis_right.spines["left"].set_visible(False)
    axis_right.spines["right"].set_color("#334155")
    axis_right.set_ylabel("Registered clients", fontsize=12, color=POSTER_TEXT_COLOR)
    axis_right.tick_params(axis="y", colors="#334155", labelsize=10.5)
    axis_right.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis_right.set_ylim(0, float(plot_data["client_count"].astype(float).max()) * 1.16)

    year_end_positions = plot_data.index[plot_data["is_year_end"]].tolist()
    if not year_end_positions:
        year_end_positions = list(range(len(plot_data)))
    axis_left.set_xticks(year_end_positions)
    axis_left.set_xticklabels([str(plot_data.iloc[position]["quarter_end"].year) for position in year_end_positions])

    subtitle_text = subtitle or CAPACITY_SOURCE_NOTE
    _apply_chart_title(figure, title, subtitle_text)
    _annotate_growth_note(
        axis_left,
        plot_data.iloc[0],
        float(plot_data.iloc[-1]["capacity_mw"] - plot_data.iloc[0]["capacity_mw"]),
        (float(plot_data.iloc[-1]["capacity_mw"] / plot_data.iloc[0]["capacity_mw"] - 1.0) if plot_data.iloc[0]["capacity_mw"] else None),
    )

    handles = [left_bars, right_bars]
    labels = ["Installed capacity (MW)", "Registered clients"]
    axis_left.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.84), frameon=False, fontsize=10.5)

    for _, row in _year_end_rows(plot_data).iterrows():
        position = float(plot_data.index[plot_data["quarter_key"] == row["quarter_key"]][0])
        axis_left.annotate(
            f"{row['capacity_mw']:,.0f}",
            xy=(position - width / 2, row["capacity_mw"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.2,
            color=CAPACITY_MAIN_COLOR,
            fontweight="semibold",
        )

    return figure


def build_capacity_growth_combo_figure(
    quarterly_data: pd.DataFrame,
    title: str = "Puerto Rico PV Capacity and Quarterly Growth",
    subtitle: str | None = None,
) -> plt.Figure:
    """Create a dual-axis chart with capacity bars and quarterly growth line."""

    plot_data = quarterly_data.sort_values("quarter_end").reset_index(drop=True)
    x_positions = np.arange(len(plot_data), dtype=float)

    figure, axis_left = plt.subplots(figsize=(15.2, 7.2), constrained_layout=True)
    axis_right = axis_left.twinx()

    bars = axis_left.bar(
        x_positions,
        plot_data["capacity_mw"],
        width=0.68,
        color=CAPACITY_MAIN_COLOR,
        alpha=0.88,
        label="Installed capacity (MW)",
    )
    growth_values = plot_data["capacity_growth_pct"].fillna(0.0)
    line = axis_right.plot(
        x_positions,
        growth_values,
        color=CAPACITY_ACCENT_COLOR,
        linewidth=2.6,
        marker="o",
        markersize=4.8,
        markerfacecolor="#fff7ed",
        markeredgecolor="#9a3412",
        label="Quarterly growth (%)",
    )[0]

    _style_capacity_axis(axis_left, "Installed capacity (MW)")
    axis_left.set_xlim(-1.0, len(plot_data))
    axis_left.set_ylim(0, float(plot_data["capacity_mw"].max()) * 1.16)
    axis_left.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))

    axis_right.spines["top"].set_visible(False)
    axis_right.spines["left"].set_visible(False)
    axis_right.spines["right"].set_color("#334155")
    axis_right.set_ylabel("Quarterly growth (%)", fontsize=12, color=POSTER_TEXT_COLOR)
    axis_right.tick_params(axis="y", colors="#334155", labelsize=10.5)
    growth_ceiling = max(5.0, float(growth_values.max()) * 1.2)
    axis_right.set_ylim(0, growth_ceiling)
    axis_right.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))

    year_end_positions = plot_data.index[plot_data["is_year_end"]].tolist()
    if not year_end_positions:
        year_end_positions = list(range(len(plot_data)))
    axis_left.set_xticks(year_end_positions)
    axis_left.set_xticklabels([str(plot_data.iloc[position]["quarter_end"].year) for position in year_end_positions])

    subtitle_text = subtitle or CAPACITY_SOURCE_NOTE
    _apply_chart_title(figure, title, subtitle_text)
    _annotate_growth_note(
        axis_left,
        plot_data.iloc[0],
        float(plot_data.iloc[-1]["capacity_mw"] - plot_data.iloc[0]["capacity_mw"]),
        (float(plot_data.iloc[-1]["capacity_mw"] / plot_data.iloc[0]["capacity_mw"] - 1.0) if plot_data.iloc[0]["capacity_mw"] else None),
    )

    axis_left.legend([bars, line], ["Installed capacity (MW)", "Quarterly growth (%)"], loc="upper left", bbox_to_anchor=(0.01, 0.84), frameon=False, fontsize=10.5)

    for _, row in _year_end_rows(plot_data).iterrows():
        position = float(plot_data.index[plot_data["quarter_key"] == row["quarter_key"]][0])
        axis_left.annotate(
            f"{row['capacity_mw']:,.0f}",
            xy=(position, row["capacity_mw"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.2,
            color=CAPACITY_MAIN_COLOR,
            fontweight="semibold",
        )

    return figure


def build_capacity_figure(data: pd.DataFrame) -> plt.Figure:
    """Backward-compatible wrapper that builds the poster chart from monthly or quarterly data."""

    if "quarter_end" in data.columns:
        quarterly_data = data
    else:
        quarterly_data = aggregate_quarterly_capacity_records(data)
    return build_poster_capacity_figure(quarterly_data)


def save_capacity_figure_variants(
    figure: plt.Figure,
    output_stem: Path = OUTPUT_STEM,
    dpi: int = 300,
) -> dict[str, Path]:
    """Write PNG and SVG poster assets for the capacity figure."""

    base_path = output_stem if output_stem.is_absolute() else PROJECT_ROOT / output_stem
    if base_path.suffix:
        base_path = base_path.with_suffix("")
    base_path.parent.mkdir(parents=True, exist_ok=True)

    png_path = base_path.with_suffix(".png")
    svg_path = base_path.with_suffix(".svg")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    return {"png": png_path, "svg": svg_path}


if __name__ == "__main__":
    workbook_info = inspect_workbook()
    print(f"Workbook source: {workbook_info['source_path']}")
    print(f"Workbook period range: {workbook_info['period_min']} -> {workbook_info['period_max']}")

    monthly_capacity = load_monthly_capacity_records()
    quarterly_capacity = aggregate_quarterly_capacity_records(monthly_capacity)
    print(quarterly_capacity[["quarter_key", "capacity_mw", "client_count"]].tail(8).to_string(index=False))

    output_paths = save_capacity_figure_variants(build_poster_capacity_figure(quarterly_capacity))
    quarterly_csv_path = OUTPUT_STEM.with_suffix(".csv")
    quarterly_csv_path.parent.mkdir(parents=True, exist_ok=True)
    quarterly_capacity.to_csv(quarterly_csv_path, index=False)

    print(f"Quarterly capacity data saved to {quarterly_csv_path}")
    print(f"Capacity figure PNG saved to {output_paths['png']}")
    print(f"Capacity figure SVG saved to {output_paths['svg']}")