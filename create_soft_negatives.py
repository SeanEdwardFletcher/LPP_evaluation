import json
import os
import random


def create_negative_examples(json_file, bm25_results_file, folder_path, output_file):
    # Load queries and relevant texts
    with open(json_file, 'r') as f:
        query_relevant_texts = json.load(f)

    # Load BM25 results
    with open(bm25_results_file, 'r') as f:
        bm25_results = json.load(f)

    # Get all available .txt files
    all_txt_files = set(os.listdir(folder_path))

    # Dictionary to store negative examples
    negative_examples = {}

    for query, relevant_texts in query_relevant_texts.items():
        relevant_set = set(relevant_texts)
        bm25_top_200 = set(bm25_results.get(query, []))  # Get top 200 results for the query

        # Exclude relevant texts and BM25 top 200 from all files
        excluded_files = relevant_set.union(bm25_top_200)
        eligible_negatives = list(all_txt_files - excluded_files)

        if len(eligible_negatives) < len(relevant_set):
            print(f"Warning: Not enough eligible negatives for query {query}. Adjusting count.")

        # Randomly sample negatives, ensuring they match the count of relevant examples
        negative_samples = random.sample(eligible_negatives, min(len(relevant_set), len(eligible_negatives)))

        # Save negative examples for the query
        negative_examples[query] = negative_samples

    # Save the negative examples to a file
    with open(output_file, 'w') as f:
        json.dump(negative_examples, f, indent=4)

    print(f"Negative examples saved to {output_file}")


# Parameters
json_file = r"C:\Users\fletc\OneDrive\Desktop\MS_ComSci\Independent_Study_F24\COLIEE_dataset\LPP_validation.json"  # Path to the JSON file with queries and relevant texts
bm25_results_file = r"C:\Users\fletc\PycharmProjects\IRfall2024\bm25_output_top_200_LPP_validation.json"  # Path to the BM25 results file
folder_path = r"C:\Users\fletc\OneDrive\Desktop\MS_ComSci\Independent_Study_F24\COLIEE_dataset\task1_train_files_2024\task1_train_files_2024"  # Path to the folder containing .txt files
output_file = 'soft_negative_validation_examples.json'  # Output file to save the negative examples

# Create negative examples
create_negative_examples(json_file, bm25_results_file, folder_path, output_file)
