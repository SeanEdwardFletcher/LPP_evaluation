import json
import pandas as pd
import matplotlib.pyplot as plt

# Load data from JSON file
with open("z_bert_embeddings_all_k_metrics.json", "r") as file:
    bert_data = json.load(file)

with open("z_lpp_embeddings_all_k_metrics.json", "r") as file:
    lpp_data = json.load(file)

# Convert data into DataFrames
bm25_df = pd.DataFrame(bert_data["bm25"]).T.astype(float)
lpp_df = pd.DataFrame(lpp_data["reranked"]).T.astype(float)
bert_df = pd.DataFrame(bert_data["reranked"]).T.astype(float)

# Plot each graph separately
metrics = ["micro_precision", "micro_recall", "micro_f2"]
for metric in metrics:
    plt.figure(figsize=(10, 5))
    plt.plot(bm25_df.index, bm25_df[metric], label="BM25", marker="o")
    plt.plot(lpp_df.index, lpp_df[metric], label="LPP", marker="x")
    plt.plot(bert_df.index, bert_df[metric], label="BERT", marker="^")
    plt.title(f"Comparison of {metric.replace('_', ' ').title()}")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xlabel("For K at")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()