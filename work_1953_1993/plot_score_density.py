from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm


# =========================
# 1. 基本设置
# =========================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = PROJECT_DIR / "output" / "Speeches_final.csv"

OUTPUT_DIR = PROJECT_DIR / "output" / "score_heatmap_only"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROCEDURAL_SCORE_COL = "procedural_score"
CLUSTER_SCORE_COL = "cluster_score"

SCORE_LEVELS = [0, 0.25, 0.5, 0.75, 1.0]


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
# 3. 检查列名
# =========================

required_cols = [PROCEDURAL_SCORE_COL, CLUSTER_SCORE_COL]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(
        f"\n以下列在数据中不存在：{missing_cols}\n\n"
        f"当前数据中的列包括：\n{list(df.columns)}\n\n"
        f"请检查 procedural_score 或 cluster_score 的列名。"
    )


# =========================
# 4. 清理数据
# =========================

df[PROCEDURAL_SCORE_COL] = pd.to_numeric(
    df[PROCEDURAL_SCORE_COL],
    errors="coerce"
)

df[CLUSTER_SCORE_COL] = pd.to_numeric(
    df[CLUSTER_SCORE_COL],
    errors="coerce"
)

plot_df = df[[PROCEDURAL_SCORE_COL, CLUSTER_SCORE_COL]].dropna().copy()

plot_df = plot_df[
    plot_df[PROCEDURAL_SCORE_COL].between(0, 1)
    & plot_df[CLUSTER_SCORE_COL].between(0, 1)
].copy()

print(f"Valid rows for plotting: {len(plot_df):,}")


# =========================
# 5. 将分数归入最近的 0, 0.25, 0.5, 0.75, 1
# =========================

def round_to_nearest_quarter(x):
    return round(x * 4) / 4


plot_df["procedural_score_discrete"] = (
    plot_df[PROCEDURAL_SCORE_COL]
    .apply(round_to_nearest_quarter)
    .clip(0, 1)
)

plot_df["cluster_score_discrete"] = (
    plot_df[CLUSTER_SCORE_COL]
    .apply(round_to_nearest_quarter)
    .clip(0, 1)
)


# =========================
# 6. 构造 joint distribution
# =========================

cross_tab = pd.crosstab(
    plot_df["procedural_score_discrete"],
    plot_df["cluster_score_discrete"],
    normalize="index"
)

cross_tab = cross_tab.reindex(
    index=SCORE_LEVELS,
    columns=SCORE_LEVELS,
    fill_value=0
)

print("\nJoint distribution table:")
print(cross_tab)


# =========================
# 7. 只画热力图
# =========================

plt.figure(figsize=(8, 7))
ax = plt.gca()

values = cross_tab.values

# 更柔和的渐变色，并增强中低值之间的区分度
norm = PowerNorm(gamma=0.6, vmin=0, vmax=values.max())

im = ax.imshow(
    values,
    cmap="YlGnBu",
    norm=norm,
    interpolation="nearest",
    aspect="auto"
)

ax.set_title(
    "Joint Distribution of Procedural Score and Cluster Score",
    fontsize=16,
    pad=12,
)

ax.set_xlabel("K-means cluster score", fontsize=13)
ax.set_ylabel("LLM procedural score", fontsize=13)

ax.set_xticks(range(len(SCORE_LEVELS)))
ax.set_yticks(range(len(SCORE_LEVELS)))

ax.set_xticklabels([str(s) for s in SCORE_LEVELS], fontsize=11)
ax.set_yticklabels([str(s) for s in SCORE_LEVELS], fontsize=11)

# 在格子中标注百分比
max_value = values.max()

for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        value = values[i, j]

        # 根据背景深浅自动调整文字颜色
        text_color = "white" if value > max_value * 0.45 else "black"

        ax.text(
            j,
            i,
            f"{value * 100:.1f}%",
            ha="center",
            va="center",
            fontsize=11,
            color=text_color,
        )

# 加 colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Share of speeches", fontsize=12)

plt.tight_layout()

output_fig = OUTPUT_DIR / "joint_distribution_heatmap_only.png"
plt.savefig(output_fig, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved heatmap to: {output_fig}")
print("\nDone.")