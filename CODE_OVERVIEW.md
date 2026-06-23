# Python 代码文件说明

这份文档用简单语言说明 `src` 文件夹中每个 Python 文件的作用。整体来看，项目代码可以分成五类：

1. 数据准备
2. KMeans 建模
3. 聚类结果解释与一致性分析
4. 作图和抽样
5. 共用配置和旧版实验脚本

## 推荐阅读顺序

如果你想理解整个项目流程，建议按这个顺序看：

1. `prepare_speeches_Claude_v.py`
2. `build_1920_1949_dataset.py`
3. `model_1920_1949_text.py`
4. `plot_category_cluster_heatmap.py`
5. `plot_overall_cluster_type_trends.py`
6. `plot_agriculture_cluster_type_trends.py`
7. `analyze_cluster_label_consistency.py`

## project_config.py

这个文件保存整个项目常用的路径和列名。

它的作用是避免每个脚本都重复写项目路径，例如：

- `PROJECT_DIR`
- `OUTPUT_DIR`
- `FULL_CORPUS_OUTPUT_DIR`
- `CSV_SEPARATOR`
- `BASEPK_COLUMN`
- `YEAR_COLUMN`
- `TEXT_COLUMN`
- `AI_LABEL_COLUMN`
- `CLUSTER_COLUMN`

简单说，它是项目的“统一配置文件”。如果以后项目文件夹移动了，通常只需要检查这里，而不需要到每个脚本里改路径。

## prepare_speeches_Claude_v.py

这个文件负责最早的数据清洗。

它从 `input/lipad` 读取原始 parliamentary speeches，然后对文本做基本预处理，例如：

- 读取每天的 CSV 文件
- 清洗 `speechtext`
- 分词（tokenization）
- 去掉 stopwords
- 词干化（stemming）
- 保存清洗后的文本到 `output/processed_data`

简单说，它把原始文本变成后续模型可以使用的干净文本。

注意：这个文件目前保持原样，没有参与本次整理。

## build_1920_1949_dataset.py

这个文件负责构建 1920-1949 年的研究数据集。

它主要做三件事：

1. 从 `output/processed_data` 中找出 1920 到 1949 年的清洗后文本。
2. 把这些文本合并成一个完整的 1920-1949 数据表。
3. 根据 `basepk` 把清洗后文本和 AI 分类结果合并。

输出文件包括：

- `processed_data_1920_1949.csv`
- `speeches_1920_1949_merged.csv`

简单说，它把“清洗后的文本”和“AI 主题标签”合并到同一张表里。

## model_1920_1949_text.py

这个文件是当前主要的 KMeans 建模脚本。

它使用 1920-1949 年的完整语料进行文本聚类。主要步骤是：

1. 读取 `speeches_1920_1949_merged.csv`
2. 读取原始 `speechtext`，方便后面人工阅读原文
3. 删除太短的 speeches
4. 使用 TF-IDF 把文本转成数值特征
5. 使用 KMeans 把文本分成 20 个 clusters
6. 保存每个 speech 的 cluster 结果
7. 保存 cluster top terms
8. 保存 cluster 和 AI label 之间的交叉表
9. 保存模型文件和 TF-IDF matrix

输出结果主要放在：

- `output/1920-1949-full-corpus-v1`

简单说，它是项目中最核心的模型脚本：用 KMeans 发现 speeches 的 lexical / discourse structure。

## cluster_type_analysis.py

这个文件保存和 `cluster_type` 相关的共用函数。

这里的 `cluster_type` 是对 20 个 KMeans clusters 的进一步解释。例如：

- `substantive_policy_discourse`
- `general_parliamentary_discourse`
- `institutional_procedural_discourse`

它主要提供：

- cluster 到 cluster type 的映射
- historical period 的定义
- 按年份计算 cluster type share
- 按历史时期计算 cluster type share
- 按时期抽样 speeches
- 绘制 cluster type 趋势图

简单说，它把多个绘图脚本里重复的逻辑集中到一个地方。

## plot_overall_cluster_type_trends.py

这个文件画所有 speeches 的 cluster type 年度趋势图。

它读取建模后的完整数据，然后：

1. 根据 `kmeans_cluster` 添加 `cluster_type`
2. 计算每一年中三类 discourse type 的占比
3. 输出年度表格
4. 画年度趋势图

输出包括：

- `overall_cluster_type_share_by_year.csv`
- `overall_cluster_type_share_by_year.png`

简单说，它回答的问题是：

> 在所有 speeches 中，不同 discourse type 的比例如何随时间变化？

## plot_agriculture_cluster_type_trends.py

这个文件虽然名字里有 `agriculture`，但现在不只分析 agriculture。

它会分别分析几个重要 AI category：

- `Agriculture and Food Policy`
- `Transportation and Communications`
- `Public Finance and Taxation`

对每个 category，它都会：

1. 筛选该 category 的 speeches
2. 添加 `cluster_type`
3. 计算每年的 cluster type 占比
4. 计算每个历史时期的 cluster type 占比
5. 每个历史时期抽样若干 speeches
6. 输出表格和趋势图

简单说，它回答的问题是：

> 在同一个 AI 主题内部，不同 discourse type 如何随时间变化？

## plot_category_cluster_heatmap.py

这个文件画 AI category 和 KMeans cluster 之间的 heatmap。

它主要生成两类比较：

1. 每个 AI category 分布到哪些 KMeans clusters
2. 每个 KMeans cluster 包含哪些 AI categories

输出包括：

- counts table
- category-to-cluster share table
- cluster-to-category share table
- 两张 heatmap 图片

简单说，它帮助你看 AI label 和 KMeans cluster 是一致、分散，还是交叉混合。

## plot_cluster_statistics.py

这个文件用于查看较早版本模型结果中的 cluster 统计信息。

它读取已经保存的：

- second-stage 数据
- TF-IDF vectorizer
- TF-IDF matrix

然后输出：

- 每个 cluster 的 speech 数量和占比
- 每个 cluster 中 TF-IDF 权重最高的词
- cluster share bar plot

简单说，它用来快速检查 clusters 的大小和代表性词汇。

## analyze_cluster_label_consistency.py

这个文件分析 KMeans cluster 和 AI label 在不同时间段中的一致性。

它把年份分成三个部分：

- `test_pre`: 1920-1929
- `train`: 1930-1945
- `test_post`: 1946-1949

主要思路是：

1. 在 train period 中，找出每个 cluster 最主要对应的 AI label。
2. 把这个映射应用到 pre 和 post period。
3. 检查其他时期的 speeches 是否仍然符合 train period 的主要 label。
4. 输出一致性 summary 和交叉表。

简单说，它回答的问题是：

> cluster 和 AI label 的关系在不同历史时期是否稳定？

## export_basepk_topic_for_ai_labeling.py

这个文件用于导出给 AI label 检查或补充使用的 topic 文件。

它从完整建模结果中提取：

- `basepk`
- `topic`

并另外生成一个 unique topic 列表。

输出包括：

- `basepk_topic_for_ai_labeling.csv`
- `unique_topics_for_ai_labeling.csv`

简单说，它是为了方便后续 AI labeling 或人工检查 topic。

## sample_agriculture_by_kmeans_cluster.py

这个文件从 agriculture category 中按指定 KMeans clusters 抽样。

当前设置是：

- category: `Agriculture and Food Policy`
- clusters: `11, 15, 17, 5, 18`
- 每个 cluster 抽 10 条 speeches

输出文件：

- `agriculture_kmeans_clusters_sample.csv`

简单说，它用于 close reading：从不同 clusters 中抽一些 agriculture speeches，方便人工比较它们的语言结构。

## model_speech.py

这个文件是较早版本的 KMeans 建模脚本。

它使用指定年份的数据，例如当前设置中的：

- 1948
- 1949
- 1950

然后进行：

- train/test split
- TF-IDF vectorization
- KMeans clustering
- cluster top words 输出
- prediction 输出
- cluster frequency plot
- word frequency plot

输出主要放在 `Final` 文件夹。

简单说，它是早期模型实验脚本，适合用来测试不同年份和不同 cluster 数量的效果。

## cv_cluster_groups.py

这个文件用于比较不同 cluster 数量的表现。

它会测试不同的 `cluster_groups`，例如：

- 15
- 20
- 25
- 30

并用 temporal validation 的方式比较模型表现，例如：

- 用 1948 年训练，1949 年验证
- 用 1948-1949 年训练，1950 年验证

它会记录：

- inertia
- average distance to center
- silhouette score
- cluster size balance
- top words

简单说，它用来帮助判断 KMeans 应该分成多少个 clusters。

## 总体流程总结

可以把项目理解成下面这条线：

1. `prepare_speeches_Claude_v.py` 清洗原始文本。
2. `build_1920_1949_dataset.py` 构建 1920-1949 年研究数据。
3. `model_1920_1949_text.py` 使用 TF-IDF 和 KMeans 建模。
4. `plot_category_cluster_heatmap.py` 比较 AI label 和 KMeans cluster。
5. `plot_overall_cluster_type_trends.py` 看总体 discourse type 趋势。
6. `plot_agriculture_cluster_type_trends.py` 看重点 AI categories 内部的 discourse type 趋势。
7. `sample_agriculture_by_kmeans_cluster.py` 抽样做 close reading。
8. `analyze_cluster_label_consistency.py` 检查 cluster-label 关系是否随时间稳定。

核心研究逻辑是：

> AI label 主要回答 speech “讲什么主题”，KMeans cluster 主要帮助观察 speech “如何被书写和组织”。

