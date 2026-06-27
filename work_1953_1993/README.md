# 合并和清理 Lipad 数据

`merge_processed_with_lipad.py` 在一次运行中完成数据合并与清理。

## 修改年份

打开脚本，在文件开头修改：

```python
START_YEAR = 1953
END_YEAR = 1993
```

也可以修改文本长度过滤参数：

```python
MIN_WORD_COUNT = 50
MAX_WORD_COUNT_PERCENTILE = 99
```

脚本会先在满足其他清理条件的数据中计算全样本 `speechtext`
单词数量的第 99 百分位数，然后删除高于该阈值的超长文本。

## 处理步骤

1. 读取 `output/processed_data` 中所选年份的日度 CSV。
2. 找到 `input/lipad/年份/月` 中对应的原始 CSV。
3. 使用 `basepk` 进行 left join。
4. 加入 `speechdate`、`speakerparty`、`speakerposition` 和 `maintopic`。
5. 加入 Lipad 的原始 `speechtext`，命名为 `speechtext_oringinal`。
6. 删除 `speechtext` 单词数量少于 `MIN_WORD_COUNT` 的记录。
7. 计算全样本单词数量的第 99 百分位数，并删除高于该阈值的文本。
8. 删除 `speakername`、`speechtext_oringinal`、`speechdate`、
   `speakerparty` 或 `speakerposition` 缺失、为空或只包含空格的记录。
   `maintopic` 会保留在输出中，但不因其缺失而删除记录。
9. 增加 `speechtext_word_count` 列并输出最终数据。

脚本按日处理和写入数据，因此不会一次性将全部年份加载进内存。

## 运行

```powershell
python work_1953_1993/merge_processed_with_lipad.py
```

输出文件名会根据年份自动变化。例如：

```text
work_1953_1993/output/processed_lipad_1953_1993_filtered.csv
```
