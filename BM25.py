import json
import os
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import time

start_time = time.time()  # Record the start time

# Paths to your files
input_json_path = r"C:\Users\fletc\OneDrive\Desktop\MS_ComSci\Independent_Study_F24\COLIEE_dataset\LPP_validation.json"  # Input JSON file
documents_dir = r"C:\Users\fletc\OneDrive\Desktop\MS_ComSci\Independent_Study_F24\COLIEE_dataset\task1_train_files_2024\task1_train_files_2024_tokenized"  # Directory with tokenized documents
output_json_path = 'bm25_output_top_200_LPP_validation.json'  # Output JSON file

# Load the input JSON
with open(input_json_path, 'r') as f:
    input_data = json.load(f)


# Function to read a document
def read_document(doc_path):
    with open(doc_path, 'r', encoding='utf-8') as file:
        return file.read()


# Get all files in the documents directory
all_docs = [f for f in os.listdir(documents_dir) if os.path.isfile(os.path.join(documents_dir, f))]

# Load pre-tokenized documents
tokenized_docs = []
doc_names = []

for doc_file in tqdm(all_docs, desc="Loading Tokenized Documents", unit="doc"):
    doc_path = os.path.join(documents_dir, doc_file)
    with open(doc_path, 'r', encoding='utf-8') as f:
        tokens = f.read().strip().split()  # Assuming tokens are space-separated
        tokenized_docs.append(tokens)
        doc_names.append(doc_file)  # Track document filenames

# Initialize BM25 with pre-tokenized documents
bm25 = BM25Okapi(tokenized_docs)

# Process each query file (key in JSON)
bm25_results = {}
for query_file in tqdm(input_data.keys(), desc="Processing Queries", unit="query"):
    # Read the pre-tokenized query
    query_path = os.path.join(documents_dir, query_file)
    with open(query_path, 'r', encoding='utf-8') as f:
        tokenized_query = f.read().strip().split()  # Assuming tokens are space-separated

    # Calculate BM25 scores for the query
    scores = bm25.get_scores(tokenized_query)

    # Prepare the BM25 results for the top 200 scored documents
    scored_docs = sorted(zip(doc_names, scores), key=lambda x: x[1], reverse=True)[:200]
    top_docs = [doc_file for doc_file, score in scored_docs]

    # Add to results dictionary
    bm25_results[query_file] = top_docs

# Record the end time
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")

# Save the BM25 results to the output JSON file
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(bm25_results, f, indent=4)

print("Done!")
