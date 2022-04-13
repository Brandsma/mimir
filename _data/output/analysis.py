import pandas as pd

df = pd.read_csv("answering_run.csv")
values = df["true_answer_ROUGE_L"].to_list()
print(values)

values = [v for v in values if v != 0.0]
print(f"avg: {sum(values) / len(values)}")
