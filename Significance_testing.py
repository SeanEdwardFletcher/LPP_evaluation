import json
import scipy.stats as stats
import pandas as pd

# Load data from JSON files
with open("z_bert_embeddings_all_k_metrics.json", "r") as file:
    bert_data = json.load(file)

with open("z_lpp_embeddings_all_k_metrics.json", "r") as file:
    lpp_data = json.load(file)

# Convert data into DataFrames
bm25_df = pd.DataFrame(bert_data["bm25"]).T.astype(float)
lpp_df = pd.DataFrame(lpp_data["reranked"]).T.astype(float)
bert_df = pd.DataFrame(bert_data["reranked"]).T.astype(float)

# Extract micro precision as an example metric for paired t-tests
bm25_precision = bm25_df["micro_precision"].values
lpp_precision = lpp_df["micro_precision"].values
bert_precision = bert_df["micro_precision"].values


# Perform paired t-tests
def perform_t_test(group1, group2, label1, label2):
    t_stat, p_value = stats.ttest_rel(group1, group2)
    alpha = 0.05
    print(f"\nPaired t-test: {label1} vs {label2}")
    print(f"T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")
    if p_value < alpha:
        print(f"Result: Reject the null hypothesis (Significant difference between {label1} and {label2}).")
    else:
        print(f"Result: Fail to reject the null hypothesis (No significant difference between {label1} and {label2}).")


# BM25 vs LPP
perform_t_test(bm25_precision, lpp_precision, "BM25", "LPP")

# BM25 vs BERT
perform_t_test(bm25_precision, bert_precision, "BM25", "BERT")

# BERT vs LPP
perform_t_test(bert_precision, lpp_precision, "BERT", "LPP")
