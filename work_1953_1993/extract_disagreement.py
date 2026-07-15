from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# =========================
# 1. 基本设置
# =========================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = PROJECT_DIR / "output" / "Speeches_final.csv"

OUTPUT_DIR = PROJECT_DIR / "output" / "kmeans_high_llm_low_cases"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_COL = "year"
CATEGORY_COL = "category"
PROCEDURAL_SCORE_COL = "procedural_score"
CLUSTER_SCORE_COL = "cluster_score"

# 如果你的 K-means cluster 编号列名不同，可以改这里
POSSIBLE_CLUSTER_ID_COLS = [
    "cluster",
    "kmeans_cluster",
    "cluster_id",
    "kmeans_label",
]

# 如果你的原文列名不同，可以在这里补充
POSSIBLE_TEXT_COLS = [
    "speechtext_original",
    "speechtext",
    "text",
]

# 重点关注的制度密集类别
INSTITUTIONAL_CATEGORIES = [
    "Justice, Rights & Immigration",
    "Constitutional & Intergovernmental Relations",
    "Trade & Foreign Affairs",
    "Government & Electoral Affairs",
    "Parliamentary Procedure",
]


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
# 3. 检查必要列
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
        f"当前数据中的列包括：\n{list(df.columns)}\n"
    )

# 自动寻找 cluster 编号列
cluster_id_col = None
for col in POSSIBLE_CLUSTER_ID_COLS:
    if col in df.columns:
        cluster_id_col = col
        break

# 自动寻找文本列
text_col = None
for col in POSSIBLE_TEXT_COLS:
    if col in df.columns:
        text_col = col
        break

print(f"Detected cluster id column: {cluster_id_col}")
print(f"Detected text column: {text_col}")


# =========================
# 4. 清理数据
# =========================

df = df.copy()

df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
df[PROCEDURAL_SCORE_COL] = pd.to_numeric(df[PROCEDURAL_SCORE_COL], errors="coerce")
df[CLUSTER_SCORE_COL] = pd.to_numeric(df[CLUSTER_SCORE_COL], errors="coerce")

df[CATEGORY_COL] = df[CATEGORY_COL].astype(str).str.strip()

df = df.dropna(
    subset=[
        YEAR_COL,
        CATEGORY_COL,
        PROCEDURAL_SCORE_COL,
        CLUSTER_SCORE_COL,
    ]
).copy()

df[YEAR_COL] = df[YEAR_COL].astype(int)

# 只保留 0-1 范围内的分数
df = df[
    df[PROCEDURAL_SCORE_COL].between(0, 1)
    & df[CLUSTER_SCORE_COL].between(0, 1)
].copy()

# 是否属于制度密集类别
df["is_institutional_category"] = df[CATEGORY_COL].isin(INSTITUTIONAL_CATEGORIES)


# =========================
# 5. 构造四类比较组
# =========================

df["comparison_group"] = "Other"

# 两者都高
df.loc[
    (df[PROCEDURAL_SCORE_COL] >= 0.75)
    & (df[CLUSTER_SCORE_COL] >= 0.75),
    "comparison_group"
] = "Both high"

# LLM 高，K-means 低
df.loc[
    (df[PROCEDURAL_SCORE_COL] >= 0.75)
    & (df[CLUSTER_SCORE_COL] < 0.75),
    "comparison_group"
] = "LLM high, K-means low"

# K-means 高，LLM 低：最重要
df.loc[
    (df[PROCEDURAL_SCORE_COL] < 0.75)
    & (df[CLUSTER_SCORE_COL] >= 0.75),
    "comparison_group"
] = "K-means high, LLM low"

# 两者都低
df.loc[
    (df[PROCEDURAL_SCORE_COL] < 0.75)
    & (df[CLUSTER_SCORE_COL] < 0.75),
    "comparison_group"
] = "Both low"


# =========================
# 6. 提取两个重点案例集
# =========================

# 重点案例 1：
# LLM procedural_score 不高，但 K-means cluster_score 高
kmeans_high_llm_low = df[
    (df[PROCEDURAL_SCORE_COL] < 0.75)
    & (df[CLUSTER_SCORE_COL] >= 0.75)
].copy()

# 重点案例 2：
# LLM 很低，但 K-means 中高
kmeans_mid_high_llm_very_low = df[
    (df[PROCEDURAL_SCORE_COL] <= 0.25)
    & (df[CLUSTER_SCORE_COL] >= 0.5)
].copy()

# 共同高分案例，作为对照
both_high = df[
    (df[PROCEDURAL_SCORE_COL] >= 0.75)
    & (df[CLUSTER_SCORE_COL] >= 0.75)
].copy()

# LLM 高但 K-means 低，作为另一个对照
llm_high_kmeans_low = df[
    (df[PROCEDURAL_SCORE_COL] >= 0.75)
    & (df[CLUSTER_SCORE_COL] < 0.75)
].copy()


# =========================
# 7. 选择导出的列
# =========================

base_export_cols = [
    YEAR_COL,
    CATEGORY_COL,
    PROCEDURAL_SCORE_COL,
    CLUSTER_SCORE_COL,
    "comparison_group",
    "is_institutional_category",
]

if cluster_id_col is not None:
    base_export_cols.append(cluster_id_col)

if text_col is not None:
    base_export_cols.append(text_col)

# 如果还有这些列，也一起导出
optional_cols = [
    "speakerposition",
    "speaker_position",
    "speakername",
    "party",
    "topic",
]

for col in optional_cols:
    if col in df.columns and col not in base_export_cols:
        base_export_cols.append(col)


# =========================
# 8. 导出案例 CSV
# =========================

def save_cases(case_df, filename):
    output_path = OUTPUT_DIR / filename

    # 为了方便 close reading，优先看：
    # cluster_score 高、procedural_score 低的文本
    case_df = case_df.sort_values(
        by=[CLUSTER_SCORE_COL, PROCEDURAL_SCORE_COL],
        ascending=[False, True]
    )

    case_df[base_export_cols].to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Saved: {output_path} | rows = {len(case_df):,}")


save_cases(
    kmeans_high_llm_low,
    "cases_kmeans_high_llm_low__procedural_lt075_cluster_ge075.csv"
)

save_cases(
    kmeans_mid_high_llm_very_low,
    "cases_kmeans_mid_high_llm_very_low__procedural_le025_cluster_ge050.csv"
)

save_cases(
    both_high,
    "cases_both_high__procedural_ge075_cluster_ge075.csv"
)

save_cases(
    llm_high_kmeans_low,
    "cases_llm_high_kmeans_low__procedural_ge075_cluster_lt075.csv"
)


# =========================
# 9. 总体 group summary
# =========================

group_summary = (
    df.groupby("comparison_group")
    .agg(
        count=("comparison_group", "size"),
        mean_procedural_score=(PROCEDURAL_SCORE_COL, "mean"),
        mean_cluster_score=(CLUSTER_SCORE_COL, "mean"),
        institutional_category_share=("is_institutional_category", "mean"),
    )
    .reset_index()
)

group_summary["share_of_all_speeches"] = (
    group_summary["count"] / len(df)
)

group_summary_output = OUTPUT_DIR / "summary_by_comparison_group.csv"
group_summary.to_csv(group_summary_output, index=False, encoding="utf-8-sig")

print(f"Saved group summary to: {group_summary_output}")


# =========================
# 10. category 分布函数
# =========================

def category_distribution(case_df, group_name, filename):
    dist = (
        case_df.groupby(CATEGORY_COL)
        .agg(
            count=(CATEGORY_COL, "size"),
            mean_procedural_score=(PROCEDURAL_SCORE_COL, "mean"),
            mean_cluster_score=(CLUSTER_SCORE_COL, "mean"),
            institutional_category_share=("is_institutional_category", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    dist["share_within_group"] = dist["count"] / len(case_df) if len(case_df) > 0 else 0

    output_path = OUTPUT_DIR / filename
    dist.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved category distribution for {group_name} to: {output_path}")

    return dist


dist_kmeans_high_llm_low = category_distribution(
    kmeans_high_llm_low,
    "K-means high, LLM low",
    "category_distribution_kmeans_high_llm_low.csv"
)

dist_kmeans_mid_high_llm_very_low = category_distribution(
    kmeans_mid_high_llm_very_low,
    "K-means medium/high, LLM very low",
    "category_distribution_kmeans_mid_high_llm_very_low.csv"
)

dist_both_high = category_distribution(
    both_high,
    "Both high",
    "category_distribution_both_high.csv"
)

dist_llm_high_kmeans_low = category_distribution(
    llm_high_kmeans_low,
    "LLM high, K-means low",
    "category_distribution_llm_high_kmeans_low.csv"
)


# =========================
# 11. 年份分布
# =========================

def year_distribution(case_df, group_name, filename):
    dist = (
        case_df.groupby(YEAR_COL)
        .agg(
            count=(YEAR_COL, "size"),
            mean_procedural_score=(PROCEDURAL_SCORE_COL, "mean"),
            mean_cluster_score=(CLUSTER_SCORE_COL, "mean"),
        )
        .reset_index()
        .sort_values(YEAR_COL)
    )

    # 加入每一年总 speech 数，计算该类案例占当年全部 speeches 的比例
    yearly_total = (
        df.groupby(YEAR_COL)
        .size()
        .reset_index(name="total_speeches_in_year")
    )

    dist = dist.merge(yearly_total, on=YEAR_COL, how="left")
    dist["share_within_year"] = dist["count"] / dist["total_speeches_in_year"]

    output_path = OUTPUT_DIR / filename
    dist.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved year distribution for {group_name} to: {output_path}")

    return dist


year_kmeans_high_llm_low = year_distribution(
    kmeans_high_llm_low,
    "K-means high, LLM low",
    "year_distribution_kmeans_high_llm_low.csv"
)

year_both_high = year_distribution(
    both_high,
    "Both high",
    "year_distribution_both_high.csv"
)


# =========================
# 12. 画图一：K-means 高、LLM 低案例的 category 分布
# =========================

def plot_top_categories(dist_df, title, output_filename, top_n=12):
    if len(dist_df) == 0:
        print(f"No data for plot: {title}")
        return

    plot_data = dist_df.head(top_n).copy()
    plot_data = plot_data.sort_values("share_within_group", ascending=True)

    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    ax.barh(
        plot_data[CATEGORY_COL],
        plot_data["share_within_group"],
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Share within this group", fontsize=12)
    ax.set_ylabel("Category", fontsize=12)

    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {output_path}")


plot_top_categories(
    dist_kmeans_high_llm_low,
    title="Categories of Speeches with High K-means Score but Low LLM Procedural Score",
    output_filename="plot_categories_kmeans_high_llm_low.png",
    top_n=12,
)

plot_top_categories(
    dist_kmeans_mid_high_llm_very_low,
    title="Categories of Speeches with Medium/High K-means Score but Very Low LLM Procedural Score",
    output_filename="plot_categories_kmeans_mid_high_llm_very_low.png",
    top_n=12,
)


# =========================
# 13. 画图二：重点案例随年份变化
# =========================

def plot_year_share(year_df, title, output_filename):
    if len(year_df) == 0:
        print(f"No data for plot: {title}")
        return

    plt.figure(figsize=(11, 6))
    ax = plt.gca()

    ax.plot(
        year_df[YEAR_COL],
        year_df["share_within_year"],
        marker="o",
        linewidth=2,
    )

    # 标注关键年份
    special_years = {
        1968: "1968",
        1982: "1982",
        1986: "1986",
    }

    y_min = year_df["share_within_year"].min()
    y_max = year_df["share_within_year"].max()
    y_range = y_max - y_min
    padding = max(y_range * 0.15, 0.002)

    lower_ylim = max(0, y_min - padding)
    upper_ylim = y_max + padding

    ax.set_ylim(lower_ylim, upper_ylim)

    for special_year, label in special_years.items():
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

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Share of speeches in year", fontsize=12)

    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {output_path}")


plot_year_share(
    year_kmeans_high_llm_low,
    title="Yearly Share of Speeches with High K-means Score but Low LLM Procedural Score",
    output_filename="plot_year_share_kmeans_high_llm_low.png",
)

plot_year_share(
    year_both_high,
    title="Yearly Share of Speeches with High Scores in Both Measures",
    output_filename="plot_year_share_both_high.png",
)


# =========================
# 14. 如果有 cluster 编号，输出 cluster × category 分布
# =========================

if cluster_id_col is not None:
    cluster_category_tab = pd.crosstab(
        kmeans_high_llm_low[cluster_id_col],
        kmeans_high_llm_low[CATEGORY_COL],
        normalize="index"
    )

    cluster_category_output = (
        OUTPUT_DIR / "cluster_by_category_distribution_kmeans_high_llm_low.csv"
    )

    cluster_category_tab.to_csv(
        cluster_category_output,
        encoding="utf-8-sig"
    )

    print(
        "Saved cluster-by-category distribution for K-means high, LLM low cases "
        f"to: {cluster_category_output}"
    )


# =========================
# 15. 打印几个关键结果
# =========================

print("\n=========================")
print("Key results")
print("=========================")

print(f"Total valid speeches: {len(df):,}")

print(
    "K-means high, LLM low "
    f"(procedural_score < 0.75 and cluster_score >= 0.75): "
    f"{len(kmeans_high_llm_low):,} "
    f"({len(kmeans_high_llm_low) / len(df) * 100:.2f}%)"
)

print(
    "K-means medium/high, LLM very low "
    f"(procedural_score <= 0.25 and cluster_score >= 0.5): "
    f"{len(kmeans_mid_high_llm_very_low):,} "
    f"({len(kmeans_mid_high_llm_very_low) / len(df) * 100:.2f}%)"
)

print(
    "Both high "
    f"(procedural_score >= 0.75 and cluster_score >= 0.75): "
    f"{len(both_high):,} "
    f"({len(both_high) / len(df) * 100:.2f}%)"
)

print("\nTop categories in K-means high, LLM low cases:")
print(
    dist_kmeans_high_llm_low[
        [CATEGORY_COL, "count", "share_within_group"]
    ].head(10)
)

institutional_share = (
    kmeans_high_llm_low["is_institutional_category"].mean()
    if len(kmeans_high_llm_low) > 0
    else 0
)

print(
    "\nInstitutional category share among K-means high, LLM low cases: "
    f"{institutional_share * 100:.2f}%"
)

print("\nDone.")