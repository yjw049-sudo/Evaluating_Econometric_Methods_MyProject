from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# ============================================================
# 1. 基本设置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "output" / "Speeches_final.csv"

OUT_DIR = BASE_DIR / "output" / "figures_tables_llm_kmeans"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 输出表格
# -----------------------------

TABLE_FILE = OUT_DIR / "table_high_score_relative_word_count_by_year.csv"
QUADRANT_TABLE_FILE = OUT_DIR / "table_quadrant_relative_word_count_by_year.csv"

QUADRANT_OVERALL_TABLE_FILE = OUT_DIR / "table_quadrant_overall_share.csv"

QUADRANT_CATEGORY_LONG_FILE = OUT_DIR / "table_quadrant_category_composition_long.csv"
QUADRANT_CATEGORY_WIDE_FILE = OUT_DIR / "table_quadrant_category_composition_wide_pct.csv"

CATEGORY_QUADRANT_LONG_FILE = OUT_DIR / "table_category_quadrant_composition_long.csv"
CATEGORY_QUADRANT_WIDE_FILE = OUT_DIR / "table_category_quadrant_composition_wide_pct.csv"

QUADRANT_CATEGORY_YEAR_FILE = OUT_DIR / "table_quadrant_category_composition_by_year.csv"

CATEGORY_WORD_COUNT_TABLE_FILE = OUT_DIR / "table_category_word_count_summary.csv"


# -----------------------------
# 输出图片
# -----------------------------

FIG_MEAN_FILE = OUT_DIR / "fig_high_score_relative_word_count_mean_by_year.png"
FIG_MEDIAN_FILE = OUT_DIR / "fig_high_score_relative_word_count_median_by_year.png"

FIG_QUAD_MEAN_FILE = OUT_DIR / "fig_quadrant_relative_word_count_mean_by_year.png"
FIG_QUAD_MEDIAN_FILE = OUT_DIR / "fig_quadrant_relative_word_count_median_by_year.png"

FIG_QUADRANT_BAR_FILE = OUT_DIR / "fig_quadrant_overall_horizontal_bar.png"

FIG_CATEGORY_QUADRANT_STACKED_BAR_FILE = (
    OUT_DIR / "fig_category_quadrant_stacked_horizontal_bar.png"
)

FIG_CATEGORY_MEAN_WORD_COUNT_BAR_FILE = (
    OUT_DIR / "fig_category_mean_word_count_horizontal_bar.png"
)


# -----------------------------
# 参数
# -----------------------------

THRESHOLD = 0.75
MIN_N = 100
REFORM_YEARS = [1968, 1982, 1986]

quadrant_order = [
    "Both high",
    "K-means only",
    "LLM only",
    "Neither high",
]


# ============================================================
# 2. 读取数据
# ============================================================

usecols = [
    "year",
    "category",
    "cluster_score",
    "procedural_score",
    "speechtext_word_count",
]

df = pd.read_csv(INPUT_FILE, usecols=usecols, low_memory=False)

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["cluster_score"] = pd.to_numeric(df["cluster_score"], errors="coerce")
df["procedural_score"] = pd.to_numeric(df["procedural_score"], errors="coerce")
df["speechtext_word_count"] = pd.to_numeric(
    df["speechtext_word_count"],
    errors="coerce",
)

df = df.dropna(
    subset=[
        "year",
        "category",
        "cluster_score",
        "procedural_score",
        "speechtext_word_count",
    ]
).copy()

df["year"] = df["year"].astype(int)

print(f"Loaded rows: {len(df):,}")


# ============================================================
# 3. 计算相对于当年平均长度的 word count
# ============================================================

year_mean_word_count = (
    df.groupby("year")["speechtext_word_count"]
    .mean()
    .rename("year_mean_word_count")
    .reset_index()
)

df = df.merge(year_mean_word_count, on="year", how="left")

df["relative_word_count"] = (
    df["speechtext_word_count"] / df["year_mean_word_count"]
)


# ============================================================
# 4. 构造 high-score indicators 和四象限变量
# ============================================================

df["cluster_high"] = df["cluster_score"] >= THRESHOLD
df["procedural_high"] = df["procedural_score"] >= THRESHOLD


def classify_quadrant(row):
    if row["cluster_high"] and row["procedural_high"]:
        return "Both high"
    elif row["cluster_high"] and not row["procedural_high"]:
        return "K-means only"
    elif not row["cluster_high"] and row["procedural_high"]:
        return "LLM only"
    else:
        return "Neither high"


df["high_score_quadrant"] = df.apply(classify_quadrant, axis=1)

df["high_score_quadrant"] = pd.Categorical(
    df["high_score_quadrant"],
    categories=quadrant_order,
    ordered=True,
)


# ============================================================
# 5. 导出：high-score 年度 relative word count
# ============================================================

cluster_yearly = (
    df[df["cluster_high"]]
    .groupby("year")
    .agg(
        cluster_high_n=("relative_word_count", "size"),
        cluster_high_mean_relative_wc=("relative_word_count", "mean"),
        cluster_high_median_relative_wc=("relative_word_count", "median"),
        cluster_high_raw_mean_wc=("speechtext_word_count", "mean"),
        cluster_high_raw_median_wc=("speechtext_word_count", "median"),
    )
    .reset_index()
)

procedural_yearly = (
    df[df["procedural_high"]]
    .groupby("year")
    .agg(
        procedural_high_n=("relative_word_count", "size"),
        procedural_high_mean_relative_wc=("relative_word_count", "mean"),
        procedural_high_median_relative_wc=("relative_word_count", "median"),
        procedural_high_raw_mean_wc=("speechtext_word_count", "mean"),
        procedural_high_raw_median_wc=("speechtext_word_count", "median"),
    )
    .reset_index()
)

yearly = pd.merge(
    cluster_yearly,
    procedural_yearly,
    on="year",
    how="outer",
).sort_values("year")

yearly = yearly.merge(year_mean_word_count, on="year", how="left")

yearly["cluster_high_below_min_n"] = yearly["cluster_high_n"] < MIN_N
yearly["procedural_high_below_min_n"] = yearly["procedural_high_n"] < MIN_N

yearly.to_csv(TABLE_FILE, index=False, encoding="utf-8-sig")
print(f"Saved: {TABLE_FILE}")


# ============================================================
# 6. 导出：四象限年度 relative word count
# ============================================================

quadrant_yearly = (
    df.groupby(["year", "high_score_quadrant"], observed=True)
    .agg(
        n=("relative_word_count", "size"),
        mean_relative_wc=("relative_word_count", "mean"),
        median_relative_wc=("relative_word_count", "median"),
        raw_mean_wc=("speechtext_word_count", "mean"),
        raw_median_wc=("speechtext_word_count", "median"),
    )
    .reset_index()
)

quadrant_yearly["below_min_n"] = quadrant_yearly["n"] < MIN_N

quadrant_yearly.to_csv(
    QUADRANT_TABLE_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(f"Saved: {QUADRANT_TABLE_FILE}")


# ============================================================
# 7. 导出：四象限总体占比
# ============================================================

quadrant_overall = (
    df["high_score_quadrant"]
    .value_counts(dropna=False)
    .reindex(quadrant_order)
    .rename_axis("high_score_quadrant")
    .reset_index(name="n")
)

quadrant_overall["share"] = quadrant_overall["n"] / quadrant_overall["n"].sum()
quadrant_overall["share_pct"] = quadrant_overall["share"] * 100

quadrant_overall.to_csv(
    QUADRANT_OVERALL_TABLE_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(f"Saved: {QUADRANT_OVERALL_TABLE_FILE}")


# ============================================================
# 8. 导出：四象限中各 category 的占比
# ============================================================

# ------------------------------------------------------------
# A. 每个 quadrant 内部，各 category 占比
# 分母：该 quadrant 的全部 speeches
# ------------------------------------------------------------

quadrant_category = (
    df.groupby(["high_score_quadrant", "category"], observed=True)
    .agg(
        n=("category", "size"),
        mean_relative_wc=("relative_word_count", "mean"),
        median_relative_wc=("relative_word_count", "median"),
        mean_word_count=("speechtext_word_count", "mean"),
        median_word_count=("speechtext_word_count", "median"),
    )
    .reset_index()
)

quadrant_totals = (
    df.groupby("high_score_quadrant", observed=True)
    .size()
    .rename("quadrant_total_n")
    .reset_index()
)

quadrant_category = quadrant_category.merge(
    quadrant_totals,
    on="high_score_quadrant",
    how="left",
)

quadrant_category["share_within_quadrant"] = (
    quadrant_category["n"] / quadrant_category["quadrant_total_n"]
)

quadrant_category["share_within_quadrant_pct"] = (
    quadrant_category["share_within_quadrant"] * 100
)

quadrant_category = quadrant_category.sort_values(
    ["high_score_quadrant", "share_within_quadrant"],
    ascending=[True, False],
)

quadrant_category.to_csv(
    QUADRANT_CATEGORY_LONG_FILE,
    index=False,
    encoding="utf-8-sig",
)

quadrant_category_wide = quadrant_category.pivot_table(
    index="high_score_quadrant",
    columns="category",
    values="share_within_quadrant_pct",
    aggfunc="sum",
    fill_value=0,
    observed=True,
)

quadrant_category_wide = quadrant_category_wide.reindex(quadrant_order)

quadrant_category_wide.to_csv(
    QUADRANT_CATEGORY_WIDE_FILE,
    encoding="utf-8-sig",
)

print(f"Saved: {QUADRANT_CATEGORY_LONG_FILE}")
print(f"Saved: {QUADRANT_CATEGORY_WIDE_FILE}")


# ------------------------------------------------------------
# B. 每个 category 内部，四象限各占多少
# 分母：该 category 的全部 speeches
# ------------------------------------------------------------

category_quadrant = (
    df.groupby(["category", "high_score_quadrant"], observed=True)
    .agg(
        n=("category", "size"),
        mean_relative_wc=("relative_word_count", "mean"),
        median_relative_wc=("relative_word_count", "median"),
        mean_word_count=("speechtext_word_count", "mean"),
        median_word_count=("speechtext_word_count", "median"),
    )
    .reset_index()
)

category_totals = (
    df.groupby("category", observed=True)
    .size()
    .rename("category_total_n")
    .reset_index()
)

category_quadrant = category_quadrant.merge(
    category_totals,
    on="category",
    how="left",
)

category_quadrant["share_within_category"] = (
    category_quadrant["n"] / category_quadrant["category_total_n"]
)

category_quadrant["share_within_category_pct"] = (
    category_quadrant["share_within_category"] * 100
)

category_quadrant = category_quadrant.sort_values(
    ["category", "share_within_category"],
    ascending=[True, False],
)

category_quadrant.to_csv(
    CATEGORY_QUADRANT_LONG_FILE,
    index=False,
    encoding="utf-8-sig",
)

category_quadrant_wide = category_quadrant.pivot_table(
    index="category",
    columns="high_score_quadrant",
    values="share_within_category_pct",
    aggfunc="sum",
    fill_value=0,
    observed=True,
)

category_quadrant_wide = category_quadrant_wide.reindex(
    columns=quadrant_order
)

category_quadrant_wide.to_csv(
    CATEGORY_QUADRANT_WIDE_FILE,
    encoding="utf-8-sig",
)

print(f"Saved: {CATEGORY_QUADRANT_LONG_FILE}")
print(f"Saved: {CATEGORY_QUADRANT_WIDE_FILE}")


# ------------------------------------------------------------
# C. 每一年、每个 quadrant 内部，各 category 占比
# 分母：某一年某个 quadrant 的全部 speeches
# ------------------------------------------------------------

quadrant_category_year = (
    df.groupby(["year", "high_score_quadrant", "category"], observed=True)
    .agg(
        n=("category", "size"),
        mean_relative_wc=("relative_word_count", "mean"),
        median_relative_wc=("relative_word_count", "median"),
    )
    .reset_index()
)

quadrant_year_totals = (
    df.groupby(["year", "high_score_quadrant"], observed=True)
    .size()
    .rename("quadrant_year_total_n")
    .reset_index()
)

quadrant_category_year = quadrant_category_year.merge(
    quadrant_year_totals,
    on=["year", "high_score_quadrant"],
    how="left",
)

quadrant_category_year["share_within_quadrant_year"] = (
    quadrant_category_year["n"]
    / quadrant_category_year["quadrant_year_total_n"]
)

quadrant_category_year["share_within_quadrant_year_pct"] = (
    quadrant_category_year["share_within_quadrant_year"] * 100
)

quadrant_category_year = quadrant_category_year.sort_values(
    ["year", "high_score_quadrant", "share_within_quadrant_year"],
    ascending=[True, True, False],
)

quadrant_category_year.to_csv(
    QUADRANT_CATEGORY_YEAR_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(f"Saved: {QUADRANT_CATEGORY_YEAR_FILE}")


# ============================================================
# 9. 新增：每个 category 的平均字数统计
# ============================================================

category_word_count = (
    df.groupby("category", observed=True)
    .agg(
        n=("speechtext_word_count", "size"),
        mean_word_count=("speechtext_word_count", "mean"),
        median_word_count=("speechtext_word_count", "median"),
        std_word_count=("speechtext_word_count", "std"),
        mean_relative_word_count=("relative_word_count", "mean"),
        median_relative_word_count=("relative_word_count", "median"),
    )
    .reset_index()
)

category_word_count = category_word_count.sort_values(
    "mean_word_count",
    ascending=False,
)

category_word_count.to_csv(
    CATEGORY_WORD_COUNT_TABLE_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(f"Saved: {CATEGORY_WORD_COUNT_TABLE_FILE}")


# ============================================================
# 10. 画图函数：两条线版本
# ============================================================

def plot_relative_word_count_trend(
    data,
    cluster_col,
    procedural_col,
    ylabel,
    title,
    output_file,
):
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.3,
        alpha=0.7,
        label="Yearly average speech length = 1.0",
    )

    line_cluster, = ax.plot(
        data["year"],
        data[cluster_col],
        linewidth=2,
        label=f"cluster_score >= {THRESHOLD}",
    )

    line_procedural, = ax.plot(
        data["year"],
        data[procedural_col],
        linewidth=2,
        label=f"procedural_score >= {THRESHOLD}",
    )

    cluster_color = line_cluster.get_color()
    procedural_color = line_procedural.get_color()

    cluster_high_n = data["cluster_high_n"] >= MIN_N
    cluster_low_n = data["cluster_high_n"] < MIN_N

    procedural_high_n = data["procedural_high_n"] >= MIN_N
    procedural_low_n = data["procedural_high_n"] < MIN_N

    ax.scatter(
        data.loc[cluster_high_n, "year"],
        data.loc[cluster_high_n, cluster_col],
        marker="o",
        s=36,
        color=cluster_color,
        zorder=3,
    )

    ax.scatter(
        data.loc[cluster_low_n, "year"],
        data.loc[cluster_low_n, cluster_col],
        marker="o",
        s=52,
        facecolors="none",
        edgecolors=cluster_color,
        linewidths=1.4,
        zorder=4,
    )

    ax.scatter(
        data.loc[procedural_high_n, "year"],
        data.loc[procedural_high_n, procedural_col],
        marker="s",
        s=36,
        color=procedural_color,
        zorder=3,
    )

    ax.scatter(
        data.loc[procedural_low_n, "year"],
        data.loc[procedural_low_n, procedural_col],
        marker="s",
        s=52,
        facecolors="none",
        edgecolors=procedural_color,
        linewidths=1.4,
        zorder=4,
    )

    ymin, ymax = ax.get_ylim()

    for yr in REFORM_YEARS:
        ax.axvline(
            yr,
            linestyle="--",
            linewidth=1.3,
            alpha=0.8,
        )

        ax.text(
            yr + 0.1,
            ymax * 0.98,
            str(yr),
            rotation=90,
            va="top",
            ha="left",
            fontsize=10,
        )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    ax.legend(
        loc="best",
        frameon=False,
        title=f"Hollow marker: n < {MIN_N}",
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_file}")


# ============================================================
# 11. 画图函数：四象限年度趋势
# ============================================================

def plot_quadrant_relative_word_count(
    data,
    value_col,
    ylabel,
    title,
    output_file,
):
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.3,
        alpha=0.7,
        label="Yearly average speech length = 1.0",
    )

    marker_map = {
        "Both high": "o",
        "K-means only": "^",
        "LLM only": "s",
        "Neither high": "D",
    }

    for quadrant in quadrant_order:
        sub = data[data["high_score_quadrant"] == quadrant].copy()

        if sub.empty:
            continue

        line, = ax.plot(
            sub["year"],
            sub[value_col],
            linewidth=2,
            label=quadrant,
        )

        color = line.get_color()

        high_n = sub["n"] >= MIN_N
        low_n = sub["n"] < MIN_N

        marker = marker_map.get(quadrant, "o")

        ax.scatter(
            sub.loc[high_n, "year"],
            sub.loc[high_n, value_col],
            marker=marker,
            s=38,
            color=color,
            zorder=3,
        )

        ax.scatter(
            sub.loc[low_n, "year"],
            sub.loc[low_n, value_col],
            marker=marker,
            s=56,
            facecolors="none",
            edgecolors=color,
            linewidths=1.4,
            zorder=4,
        )

    ymin, ymax = ax.get_ylim()

    for yr in REFORM_YEARS:
        ax.axvline(
            yr,
            linestyle="--",
            linewidth=1.3,
            alpha=0.8,
        )

        ax.text(
            yr + 0.1,
            ymax * 0.98,
            str(yr),
            rotation=90,
            va="top",
            ha="left",
            fontsize=10,
        )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    ax.legend(
        loc="best",
        frameon=False,
        title=f"Hollow marker: n < {MIN_N}",
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_file}")


# ============================================================
# 12. 画图函数：四象限总体横向 bar 图
# ============================================================

def plot_quadrant_overall_horizontal_bar(
    data,
    output_file,
):
    plot_df = data.sort_values("share", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.barh(
        plot_df["high_score_quadrant"],
        plot_df["share"],
    )

    for _, row in plot_df.iterrows():
        ax.text(
            row["share"] + 0.005,
            row["high_score_quadrant"],
            f"{row['share']:.1%} (n={int(row['n']):,})",
            va="center",
            fontsize=10,
        )

    ax.xaxis.set_major_formatter(PercentFormatter(1.0))

    ax.set_xlabel("Share of speeches")
    ax.set_ylabel("")
    ax.set_title(
        f"LLM–K-means High-Score Quadrants\n"
        f"threshold = {THRESHOLD}"
    )

    ax.grid(axis="x", alpha=0.3)

    max_share = plot_df["share"].max()
    ax.set_xlim(0, min(1.0, max_share + 0.12))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_file}")


# ============================================================
# 13. 画图函数：每个 category 内部四象限占比 stacked bar
# ============================================================

def plot_category_quadrant_stacked_horizontal_bar(
    category_quadrant_wide_pct,
    output_file,
):
    category_quad_plot = category_quadrant_wide_pct.copy() / 100

    category_quad_plot = category_quad_plot.reindex(
        columns=quadrant_order,
        fill_value=0,
    )

    if "K-means only" in category_quad_plot.columns:
        category_quad_plot = category_quad_plot.sort_values(
            "K-means only",
            ascending=True,
        )

    fig_height = max(6, 0.6 * len(category_quad_plot))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    left = pd.Series(0.0, index=category_quad_plot.index)

    for q in quadrant_order:
        ax.barh(
            category_quad_plot.index,
            category_quad_plot[q],
            left=left,
            label=q,
        )

        left = left + category_quad_plot[q]

    ax.xaxis.set_major_formatter(PercentFormatter(1.0))

    ax.set_xlabel("Share within category")
    ax.set_ylabel("")
    ax.set_title(
        f"Composition of LLM–K-means Quadrants within Each Category\n"
        f"threshold = {THRESHOLD}"
    )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
    )

    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_file}")


# ============================================================
# 14. 新增画图函数：每个 category 的平均字数横向 bar 图
# ============================================================

def plot_category_mean_word_count_horizontal_bar(
    data,
    output_file,
):
    plot_df = data.sort_values("mean_word_count", ascending=True).copy()

    fig_height = max(6, 0.6 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    ax.barh(
        plot_df["category"],
        plot_df["mean_word_count"],
    )

    for _, row in plot_df.iterrows():
        ax.text(
            row["mean_word_count"] + 2,
            row["category"],
            f"{row['mean_word_count']:.1f} (n={int(row['n']):,})",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Mean speechtext_word_count")
    ax.set_ylabel("")
    ax.set_title("Mean Speech Length by Category")

    ax.grid(axis="x", alpha=0.3)

    max_x = plot_df["mean_word_count"].max()
    ax.set_xlim(0, max_x * 1.18)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_file}")


# ============================================================
# 15. 生成图片
# ============================================================

plot_relative_word_count_trend(
    data=yearly,
    cluster_col="cluster_high_mean_relative_wc",
    procedural_col="procedural_high_mean_relative_wc",
    ylabel="Mean relative speech length\n(word count / yearly mean word count)",
    title=(
        "Mean Relative Speech Length among High-Score Speeches over Time\n"
        f"(cluster_score >= {THRESHOLD} vs procedural_score >= {THRESHOLD})"
    ),
    output_file=FIG_MEAN_FILE,
)

plot_relative_word_count_trend(
    data=yearly,
    cluster_col="cluster_high_median_relative_wc",
    procedural_col="procedural_high_median_relative_wc",
    ylabel="Median relative speech length\n(word count / yearly mean word count)",
    title=(
        "Median Relative Speech Length among High-Score Speeches over Time\n"
        f"(cluster_score >= {THRESHOLD} vs procedural_score >= {THRESHOLD})"
    ),
    output_file=FIG_MEDIAN_FILE,
)

plot_quadrant_relative_word_count(
    data=quadrant_yearly,
    value_col="mean_relative_wc",
    ylabel="Mean relative speech length\n(word count / yearly mean word count)",
    title=(
        "Mean Relative Speech Length by LLM–K-means Quadrant over Time\n"
        f"(threshold = {THRESHOLD})"
    ),
    output_file=FIG_QUAD_MEAN_FILE,
)

plot_quadrant_relative_word_count(
    data=quadrant_yearly,
    value_col="median_relative_wc",
    ylabel="Median relative speech length\n(word count / yearly mean word count)",
    title=(
        "Median Relative Speech Length by LLM–K-means Quadrant over Time\n"
        f"(threshold = {THRESHOLD})"
    ),
    output_file=FIG_QUAD_MEDIAN_FILE,
)

plot_quadrant_overall_horizontal_bar(
    data=quadrant_overall,
    output_file=FIG_QUADRANT_BAR_FILE,
)

plot_category_quadrant_stacked_horizontal_bar(
    category_quadrant_wide_pct=category_quadrant_wide,
    output_file=FIG_CATEGORY_QUADRANT_STACKED_BAR_FILE,
)

plot_category_mean_word_count_horizontal_bar(
    data=category_word_count,
    output_file=FIG_CATEGORY_MEAN_WORD_COUNT_BAR_FILE,
)


# ============================================================
# 16. 结束
# ============================================================

print("\nDone.")
print(f"All outputs saved to: {OUT_DIR}")