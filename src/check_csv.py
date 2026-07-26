import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 100)

df = pd.read_csv(r"output\Speech_chunk.csv")


print(df.info())
print(df.head())

print(list(df.columns))