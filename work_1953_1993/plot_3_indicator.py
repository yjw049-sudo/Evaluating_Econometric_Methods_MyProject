from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# =========================
# 1. 基本设置
# =========================

# 当前 .py 文件所在的文件夹，也就是 work_1953_1993
PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = PROJECT_DIR / "output" / "Speeches_final.csv"

OUTPUT_DIR = PROJECT_DIR / "output" / "yearly_three_indicators"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_COL = "year"
CATEGORY_COL = "category"
PROCEDURAL_SCORE_COL = "procedural_score"

# 如果你的 K-means / cluster 分数列不叫 cluster_score，请改这里
CLUSTER_SCORE_COL = "cluster_score"

THRESHOLD = 0.75

TARGET_CATEGORY = "Parliamentary Procedure"

SPECIAL_YEARS = {
    1968: "1968",
    1982: "1982",
    1986: "1986",
}


# =========================
# 2. 读取数据
# =========================

print(f"Project folder: {PROJECT_DIR}")
print(f"Input file: {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"找不到输入文件：{INPUT_FILE}\n"
        f"请确认 Speeches_final.csv 是否在 work_1953_1993/output/ 里面。"
    )

df = pd.read_csv(INPUT_FILE)

print("Data loaded.")
print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")


# =========================
# 3. 检查必要列是否存在
# =========================

required_cols = [
    YEAR_COL,
    CATEGORY_COL,
    PROCEDURAL_SCORE_COL,
    CLUSTER_SCORE_COL,
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(
        f"\n以下列在数据中不存在：{missing_cols}\n\n"
        f"当前数据中的列包括：\n{list(df.columns)}\n\n"
        f"请检查列名，尤其是 cluster 分数列。"
    )


# =========================
# 4. 清理数据
# =========================

df = df.dropna(subset=[YEAR_COL]).copy()
df[YEAR_COL] = df[YEAR_COL].astype(int)

df[CATEGORY_COL] = df[CATEGORY_COL].astype(str).str.strip()

df[PROCEDURAL_SCORE_COL] = pd.to_numeric(
    df[PROCEDURAL_SCORE_COL],
    errors="coerce"
)

df[CLUSTER_SCORE_COL] = pd.to_numeric(
    df[CLUSTER_SCORE_COL],
    errors="coerce"
)


# =========================
# 5. 构造三个指标
# =========================

# 1. LLM procedural_score >= 0.75
df["is_procedural_score_high"] = (
    df[PROCEDURAL_SCORE_COL] >= THRESHOLD
)

# 2. category == Parliamentary Procedure
df["is_parliamentary_procedure_category"] = (
    df[CATEGORY_COL] == TARGET_CATEGORY
)

# 3. cluster_score >= 0.75
df["is_cluster_score_high"] = (
    df[CLUSTER_SCORE_COL] >= THRESHOLD
)


# =========================
# 6. 按 year 计算占比
# =========================

yearly = (
    df.groupby(YEAR_COL)
    .agg(
        total_speeches=(YEAR_COL, "size"),

        procedural_score_high_count=(
            "is_procedural_score_high",
            "sum"
        ),
        parliamentary_procedure_category_count=(
            "is_parliamentary_procedure_category",
            "sum"
        ),
        cluster_score_high_count=(
            "is_cluster_score_high",
            "sum"
        ),

        procedural_score_high_share=(
            "is_procedural_score_high",
            "mean"
        ),
        parliamentary_procedure_category_share=(
            "is_parliamentary_procedure_category",
            "mean"
        ),
        cluster_score_high_share=(
            "is_cluster_score_high",
            "mean"
        ),
    )
    .reset_index()
    .sort_values(YEAR_COL)
)

# 保存年度结果表
output_table = OUTPUT_DIR / "yearly_three_measures_share.csv"
yearly.to_csv(output_table, index=False, encoding="utf-8-sig")

print(f"Saved yearly table to: {output_table}")


# =========================
# 7. 画一张合并图：自动放大 y 轴，并标注关键年份
# =========================

plt.figure(figsize=(12, 6))
ax = plt.gca()

series_cols = [
    "procedural_score_high_share",
    "parliamentary_procedure_category_share",
    "cluster_score_high_share",
]

# 根据三条线的实际数值自动设置 y 轴范围
all_y_values = yearly[series_cols].stack().dropna()

y_min = all_y_values.min()
y_max = all_y_values.max()
y_range = y_max - y_min

# padding 越小，y 轴越“放大”
# 如果觉得波动还是不明显，可以把 0.12 改成 0.08
padding = max(y_range * 0.12, 0.003)

lower_ylim = max(0, y_min - padding)
upper_ylim = y_max + padding

ax.set_ylim(lower_ylim, upper_ylim)

# 三条线
ax.plot(
    yearly[YEAR_COL],
    yearly["procedural_score_high_share"],
    marker="o",
    linewidth=2,
    label=f"Procedural score >= {THRESHOLD}",
)

ax.plot(
    yearly[YEAR_COL],
    yearly["parliamentary_procedure_category_share"],
    marker="s",
    linewidth=2,
    label='Category = "Parliamentary Procedure"',
)

ax.plot(
    yearly[YEAR_COL],
    yearly["cluster_score_high_share"],
    marker="^",
    linewidth=2,
    label=f"Cluster score >= {THRESHOLD}",
)

# 标注三个特殊年份
for special_year, label in SPECIAL_YEARS.items():
    ax.axvline(
        x=special_year,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
    )

    ax.text(
        special_year,
        upper_ylim - (upper_ylim - lower_ylim) * 0.03,
        label,
        rotation=90,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=9,
    )

ax.set_title(
    "Comparison of Procedural Speech Measures over Time, Zoomed Y-axis",
    fontsize=14,
)

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Share of speeches", fontsize=12)

# y 轴显示为百分比
ax.yaxis.set_major_formatter(PercentFormatter(1.0))

ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

output_fig = OUTPUT_DIR / "three_measures_share_by_year_with_key_years_zoomed_yaxis.png"
plt.savefig(output_fig, dpi=300)
plt.close()

print(f"Saved combined figure with zoomed y-axis to: {output_fig}")


# =========================
# 8. 同时导出一份更直观的百分比表
# =========================

yearly_percent = yearly.copy()

share_cols = [
    "procedural_score_high_share",
    "parliamentary_procedure_category_share",
    "cluster_score_high_share",
]

for col in share_cols:
    yearly_percent[col + "_percent"] = yearly_percent[col] * 100

output_percent_table = OUTPUT_DIR / "yearly_three_measures_share_percent.csv"

yearly_percent.to_csv(
    output_percent_table,
    index=False,
    encoding="utf-8-sig"
)

print(f"Saved percent table to: {output_percent_table}")

print("\nDone.")