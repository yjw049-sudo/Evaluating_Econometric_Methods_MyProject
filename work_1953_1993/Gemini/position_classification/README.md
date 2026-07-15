# Speaker Position Classification with Gemini

把整个 `position_classification` 文件夹放到：

```text
work_1953_1993/Gemini/position_classification
```

默认输入文件是：

```text
work_1953_1993/output/Speeches_final.csv
```

## 运行顺序

```bash
cd E:\study\S2\Evaluating-Econometric-Methods_data\Evaluating_Econometric_Methods_MyProject\work_1953_1993\Gemini\position_classification

python 01_extract_unique_positions.py
python 02_discover_position_taxonomy_gemini.py
python 03_classify_positions_with_taxonomy_gemini.py
```

## 三个脚本的作用

### 01_extract_unique_positions.py

从 `Speeches_final.csv` 里只读取 `speakerposition`，提取 unique values，并给每一个 unique position 一个稳定的 `position_id`。

输出：

```text
speakerposition_unique_values.csv
```

包含：

```text
position_id, speakerposition, n_speeches_with_position
```

### 02_discover_position_taxonomy_gemini.py

把所有 unique positions 的 `position_id` 和 `speakerposition` 发给 Gemini，请它提出少量 institutional position categories。

这个 prompt 尽量保持中性：

- 不要求它按 frontbench/backbench 预设分类；
- 不要求它按照程序性高低分类；
- 不把 procedural_score、cluster_score、disagreement 或研究结果放进 prompt；
- 不按党派、部门、政策领域分类。

输出：

```text
position_taxonomy_gemini.json
position_taxonomy_gemini.raw_response.txt
```

### 03_classify_positions_with_taxonomy_gemini.py

读取第二步生成的 taxonomy，然后分 batch 给每一个 `position_id` 分类。Gemini 返回的是 `position_id` 和分类，脚本再 merge 回原始 `speakerposition`，降低文本匹配风险。

输出：

```text
speakerposition_institutional_types_gemini.csv
speakerposition_institutional_types_gemini.json
speakerposition_institutional_types_gemini.raw_response.txt
```

最终最重要的是：

```text
speakerposition_institutional_types_gemini.csv
```

它包含：

```text
position_id, speakerposition, n_speeches_with_position, institutional_position_type, confidence, rationale
```

## 常用参数

如果第三步 batch 太大：

```bash
python 03_classify_positions_with_taxonomy_gemini.py --batch-size 50
```

如果想换模型：

```bash
python 02_discover_position_taxonomy_gemini.py --model gemini-2.5-flash-lite
python 03_classify_positions_with_taxonomy_gemini.py --model gemini-2.5-flash-lite
```

## API key

脚本会从当前文件夹或上级文件夹中的 `.env` 读取：

```text
GEMINI_API_KEY=你的key
```
