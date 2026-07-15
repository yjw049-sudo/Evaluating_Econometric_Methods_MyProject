from pathlib import Path
import math
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

OUTPUT_FILE = OUT_DIR / "fig_category_high_score_small_multiples_minN.png"
TABLE_FILE = OUT_DIR / "table_category_high_score_yearly_share_minN.csv"

# 阈值
THRESHOLD = 0.75

# 最小样本量限制
# 如果某个 category-year 的 speech 数量低于这个值，则不画该点
MIN_N = 100

# 只画 Top N categories
TOP_N = 10

# 改革年份
REFORM_YEARS = [1968, 1982, 1986]

# 每行几个图
N_COLS = 2

LABEL_CLUSTER = f"cluster_score >= {THRESHOLD}"
LABEL_PROCEDURAL = f"procedural_score >= {THRESHOLD}"


# ============================================================
# 2. 读取数据
# ============================================================

usecols = [
    "year",
    "category",
    "cluster_score",
    "procedural_score",
]

df = pd.read_csv(INPUT_FILE, usecols=usecols, low_memory=False)

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["cluster_score"] = pd.to_numeric(df["cluster_score"], errors="coerce")
df["procedural_score"] = pd.to_numeric(df["procedural_score"], errors="coerce")

df = df.dropna(
    subset=["year", "category", "cluster_score", "procedural_score"]
).copy()

df["year"] = df["year"].astype(int)

df["cluster_high"] = (df["cluster_score"] >= THRESHOLD).astype(int)
df["procedural_high"] = (df["procedural_score"] >= THRESHOLD).astype(int)


# ============================================================
# 3. 选取 Top categories
# ============================================================

category_order = (
    df["category"]
    .value_counts()
    .head(TOP_N)
    .index
    .tolist()
)

df_top = df[df["category"].isin(category_order)].copy()

df_top["category"] = pd.Categorical(
    df_top["category"],
    categories=category_order,
    ordered=True
)


# ============================================================
# 4. 计算 category-year 内部高分占比
# ============================================================

plot_df = (
    df_top.groupby(["category", "year"], observed=True)
    .agg(
        cluster_high_share=("cluster_high", "mean"),
        procedural_high_share=("procedural_high", "mean"),
        n=("cluster_high", "size"),
    )
    .reset_index()
)

# 保存未过滤前的结果，方便检查
plot_df["cluster_high_share_pct"] = plot_df["cluster_high_share"] * 100
plot_df["procedural_high_share_pct"] = plot_df["procedural_high_share"] * 100
plot_df["below_min_n"] = plot_df["n"] < MIN_N

plot_df.to_csv(TABLE_FILE, index=False, encoding="utf-8-sig")

# 低于最小样本量的点设为 NaN，画图时自动断开
plot_df.loc[
    plot_df["n"] < MIN_N,
    ["cluster_high_share", "procedural_high_share"]
] = pd.NA


# ============================================================
# 5. 补齐年份，保证每个小图 x 轴一致
# ============================================================

all_years = sorted(df_top["year"].unique())

full_index = pd.MultiIndex.from_product(
    [category_order, all_years],
    names=["category", "year"]
)

plot_df = (
    plot_df.set_index(["category", "year"])
    .reindex(full_index)
    .reset_index()
)


# ============================================================
# 6. 画图
# ============================================================

n_categories = len(category_order)
n_rows = math.ceil(n_categories / N_COLS)

fig, axes = plt.subplots(
    n_rows,
    N_COLS,
    figsize=(14, 3.2 * n_rows),
    sharex=True,
    sharey=False
)

axes = axes.flatten()

for i, category in enumerate(category_order):
    ax = axes[i]

    sub = plot_df[plot_df["category"] == category].copy()

    ax.plot(
        sub["year"],
        sub["cluster_high_share"],
        marker="o",
        linewidth=1.8,
        markersize=3.5,
        label=LABEL_CLUSTER,
    )

    ax.plot(
        sub["year"],
        sub["procedural_high_share"],
        marker="s",
        linewidth=1.8,
        markersize=3.5,
        label=LABEL_PROCEDURAL,
    )

    # --------------------------------------------------------
    # y 轴设置
    # 除 Parliamentary Procedure 外，其余图固定为 0-25%
    # --------------------------------------------------------

    if category == "Parliamentary Procedure":
        # Parliamentary Procedure 的数值明显更高，单独设定较合理的范围
        # 你也可以改成 ax.set_ylim(0, 1)
        ax.set_ylim(0.45, 0.90)
    else:
        ax.set_ylim(0, 0.25)

    # --------------------------------------------------------
    # 改革年份竖线
    # --------------------------------------------------------

    ymin, ymax = ax.get_ylim()

    for yr in REFORM_YEARS:
        ax.axvline(
            yr,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8
        )

        ax.text(
            yr + 0.1,
            ymax * 0.98,
            str(yr),
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
        )

    ax.set_title(category, fontsize=10)

    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3)

    # 在标题下方提醒低样本年份被隐藏
    ax.text(
        0.99,
        0.02,
        f"min n = {MIN_N}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        alpha=0.6,
    )


# 删除多余空白子图
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])


# ============================================================
# 7. 总标题、图例和保存
# ============================================================

fig.suptitle(
    f"Top {TOP_N} Categories: Yearly Share of High-Score Speeches\n"
    f"(cluster_score >= {THRESHOLD} vs procedural_score >= {THRESHOLD}; "
    f"category-year n >= {MIN_N})",
    fontsize=16,
    y=0.995,
)

fig.text(0.5, 0.02, "Year", ha="center", fontsize=12)
fig.text(0.02, 0.5, "Share", va="center", rotation="vertical", fontsize=12)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.955),
)

plt.tight_layout(rect=[0.04, 0.04, 1, 0.93])

plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()

print(f"Saved figure to: {OUTPUT_FILE}")
print(f"Saved yearly table to: {TABLE_FILE}")