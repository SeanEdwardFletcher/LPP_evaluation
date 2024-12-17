import os
import json
import torch
from rank_bm25 import BM25Okapi
from bi_lstm_model import TwoLayerBiLSTMCrossEncoder
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, fbeta_score
import argparse
from linear_pyramid_pooling import PyramidPooling1D


# Step 1: Load Pre-Tokenized Documents for BM25
def load_pre_tokenized_documents(folder):
    documents = {}
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            with open(filepath, "r") as f:
                documents[filename] = f.read().split()  # Tokenized document
        except Exception as e:
            print(f"Error loading tokenized document {filename}: {e}")
    return documents


# Step 2: Load Pre-Embedded Documents
def load_pre_embedded_documents(folder, embedding_type):
    """
    Load pre-embedded documents.

    Args:
        folder (str): Path to the folder containing document subfolders.
        embedding_type (str): Type of embedding to load ("document" or "paragraph").

    Returns:
        dict: A dictionary mapping document IDs to their embeddings.
    """
    embeddings = {}
    for doc_id in os.listdir(folder):
        doc_folder = os.path.join(folder, doc_id)
        if not os.path.isdir(doc_folder):
            continue  # Skip files that are not folders

        # Select the correct file name based on embedding_type
        if embedding_type == "document":
            embedding_file = os.path.join(doc_folder, "document_embedding.pt")
        elif embedding_type == "paragraph":
            embedding_file = os.path.join(doc_folder, "paragraph_embeddings.pt")
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

        try:
            embedding = torch.load(embedding_file)
            if embedding_type == "paragraph":
                # Concatenate paragraph embeddings into a single 1D vector
                if isinstance(embedding, list):  # Ensure it's a list of embeddings
                    concatenated_embedding = torch.cat(embedding, dim=0)  # Concatenate along dimension 0
                elif isinstance(embedding, torch.Tensor):  # Ensure it's a tensor
                    concatenated_embedding = embedding.view(-1)  # Flatten into 1D
                else:
                    raise ValueError(f"Unexpected format in {embedding_file}: {type(embedding)}")
                embeddings[doc_id] = concatenated_embedding
            else:
                # Directly load the document embedding
                embeddings[doc_id] = embedding
        except Exception as e:
            print(f"Error loading embedding for {doc_id} from {embedding_file}: {e}")
    return embeddings


# Step 3: BM25 Ranking
def bm25_ranking(query_name, query_text, bm25_model, doc_ids, top_k=1000):
    try:
        # Get BM25 scores for the query
        scores = bm25_model.get_scores(query_text)

        # Pair document IDs with their scores
        doc_scores = list(zip(doc_ids, scores))

        # Sort the document-score pairs by scores in descending order
        top_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:top_k]

        # Return only the document names (or IDs) of the top_k results
        return [doc for doc, _ in top_docs]
    except Exception as e:
        print(f"Error in BM25 ranking for query {query_name}: {e}")
        return []


# Step 4: Rerank Using Bi-LSTM Cross Encoder
def rerank_with_bilstm(query_embedding, candidate_embeddings, model, device):
    scores = []
    try:
        # Ensure the model is on the correct device
        model.to(device)
        model.eval()

        # Move the query embedding to the correct device
        query_embedding = query_embedding.to(device)

        with torch.no_grad():
            for candidate_name, candidate_embedding in candidate_embeddings.items():
                # Move the candidate embedding to the correct device
                candidate_embedding = candidate_embedding.to(device)

                # Add batch dimension and ensure all tensors are on the same device
                query_repeated = query_embedding.unsqueeze(0)  # [1, embedding_dim]
                candidate_embedding = candidate_embedding.unsqueeze(0)  # [1, embedding_dim]

                # Compute similarity score
                score = model(query_repeated, candidate_embedding).item()
                scores.append((candidate_name, score))

        # Sort the scores in descending order
        scores.sort(key=lambda x: x[1], reverse=True)

    except Exception as e:
        print(f"Error during Bi-LSTM reranking: {e}")

    return scores  # Return list of tuples (document_name, score)


# Step 5: Calculate Metrics
def calculate_metrics(queries, results, output_file=None):
    results_list = []
    y_true_micro = []
    y_pred_micro = []

    for query, relevant_docs in queries.items():
        query_no_ext = os.path.splitext(query)[0]  # Remove the file extension from the test ground truth JSON file
        relevant_doc_list = [os.path.splitext(doc)[0] for doc in relevant_docs]
        predicted_docs = results.get(query_no_ext, {})
        predicted_doc_list = [doc for doc in predicted_docs.keys()]
        all_considered_docs = set(relevant_doc_list + predicted_doc_list)

        # Generate y_true and y_pred for all possible documents
        y_true = [1 if doc in relevant_doc_list else 0 for doc in all_considered_docs]
        y_pred = [1 if doc in predicted_doc_list else 0 for doc in all_considered_docs]

        y_true_micro.extend(y_true)
        y_pred_micro.extend(y_pred)

        # Per-query metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

        results_list.append({
            "query": query,
            "precision": precision,
            "recall": recall,
            "f2_score": f2,
        })

    # Aggregate micro metrics
    micro_precision = precision_score(y_true_micro, y_pred_micro, zero_division=0)
    micro_recall = recall_score(y_true_micro, y_pred_micro, zero_division=0)
    micro_f2 = fbeta_score(y_true_micro, y_pred_micro, beta=2, zero_division=0)

    aggregated_metrics = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f2": micro_f2,
    }

    if output_file:
        with open(output_file, "w") as f:
            json.dump({"per_query": results_list, "aggregated": aggregated_metrics}, f, indent=4)

    return aggregated_metrics


# Step 6: Save Results to File
def save_results_to_file(results, output_file):
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving results to file {output_file}: {e}")


# Step 7: Main Execution
if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="BM25 and Bi-LSTM Testing Script")
    parser.add_argument("--pre_tokenized_folder", type=str, default="../task1_train_files_2024_tokenized", help="Path to the folder containing pre-tokenized documents")
    parser.add_argument("--embeddings", type=str, default="bert", choices=["bert", "sailer"], help="oscillates between two embedding folders")
    parser.add_argument("--query_file", type=str, default="./LPP_test.json", help="Path to the query JSON file")
    parser.add_argument("--model-path", type=int, default=1, choices=[1, 2], help="oscillates between two paths to pretrained Bi-LSTM models")
    parser.add_argument("--embedding_type", type=str, default="document", choices=["document", "paragraph"], help="Type of embedding to load (document or paragraph)")
    parser.add_argument("--lpp", action="store_true", help="Enable Linear Pyramid Pooling (LPP)")
    parser.add_argument("--pooling", type=str, default="avg", choices=["avg", "max"], help="Pooling method for LPP (avg or max)")
    parser.add_argument("--prefix", type=str, default="", help="String to prepend to all saved filenames")

    args = parser.parse_args()

    if args.embeddings == "bert":
        args.embeddings = "../training_data_embeddings_bert"
    elif args.embeddings == "sailer":
        args.embeddings = "../training_data_embeddings_sailer"
    else:
        raise ValueError(f"Unsupported embeddings argument call: {args.embeddings}")

    if args.model_path == 1:
        args.model_path = "./ex_01_bert_max_pool_best_model_epoch_40.pth"
    elif args.model_path == 2:
        args.model_path = "./ex_02_bert_lpp_avg_best_model_epoch_28.pth"
    else:
        raise ValueError(f"Unsupported pretrained_model_path argument call: {args.model_path}")

    # Load data
    print("Loading pre-tokenized documents...")
    pre_tokenized_docs = load_pre_tokenized_documents(args.pre_tokenized_folder)

    # # Print 3 samples from pre_tokenized_docs
    # print("Sample pre-tokenized documents:")
    # for idx, (doc_name, tokens) in enumerate(pre_tokenized_docs.items()):
    #     print(f"Document Name: {doc_name}, Tokens: {tokens[:10]}")  # Print only the first 10 tokens for brevity
    #     if idx >= 2:  # Stop after printing 3 samples
    #         break

    # Convert pre-tokenized documents into a list for BM25
    tokenized_doc_list = list(pre_tokenized_docs.values())  # List of tokenized document contents
    doc_ids = list(pre_tokenized_docs.keys())  # List of corresponding document IDs

    # Load pre-embedded documents
    print("Loading pre-embedded documents...")
    pre_embedded_docs = load_pre_embedded_documents(args.embeddings, args.embedding_type)

    # Load the testing data json file
    try:
        with open(args.query_file, "r") as f:
            queries = json.load(f)
    except Exception as e:
        print(f"Error loading queries file {args.query_file}: {e}")
        queries = {}

    # Initialize BM25
    try:
        bm25 = BM25Okapi(tokenized_doc_list)  # Pass only the list of tokenized documents
    except Exception as e:
        print(f"Error initializing BM25: {e}")

    # Initialize and load Bi-LSTM model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine the embedding dimension
    if args.lpp:
        lpp_levels = [16, 32, 64, 128, 256]  # Define LPP levels
        embedding_dim = sum(lpp_levels)  # Adjust embedding dimension
        lpp_module = PyramidPooling1D(levels=lpp_levels, pooling=args.pooling)  # Initialize LPP with dynamic pooling
    else:
        embedding_dim = 768  # Default embedding dimension for precomputed embeddings

    # Instantiate the model with the adjusted embedding dimension
    model = TwoLayerBiLSTMCrossEncoder(embedding_dim=embedding_dim, hidden_dim=256)

    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
    except Exception as e:
        print(f"Error loading pretrained model: {e}")

    # Step 1: Generate BM25 Results
    bm25_results = {}

    # counter = 0

    for query_name, relevant_docs in tqdm(queries.items(), desc="Generating BM25 Results"):
        query_name = os.path.splitext(query_name)[0]  # Remove query file extension
        try:
            query_text = pre_tokenized_docs.get(f"{query_name}.txt", [])

            # if counter <= 5:
            #     print(query_text)
            #     counter += 1

            bm25_candidates = bm25_ranking(query_name, query_text, bm25, doc_ids)

            # Remove file extensions from BM25 candidates
            bm25_candidates = [os.path.splitext(candidate)[0] for candidate in bm25_candidates]

            # Store BM25 candidates
            bm25_results[query_name] = bm25_candidates
        except Exception as e:
            print(f"Error generating BM25 results for query {query_name}: {e}")


    # Step 2: Rerank Using Bi-LSTM
    reranked_results = {}

    for query_name, bm25_candidates in tqdm(bm25_results.items(), desc="Reranking with Bi-LSTM"):
        try:
            query_embedding = pre_embedded_docs.get(query_name)
            if query_embedding is not None:
                query_embedding = query_embedding.to(device)

                # Apply LPP if enabled
                if args.lpp:
                    assert query_embedding.dim() == 1, "Expected a 1D embedding"
                    query_embedding = lpp_module(
                        query_embedding.unsqueeze(0).unsqueeze(0))  # Add batch and channel dims
                    query_embedding = query_embedding.squeeze(0).squeeze(0)  # Remove added dims
                    assert query_embedding.dim() == 1, "Expected a 1D embedding after LPP"

                candidate_embeddings = {}
                for doc_key in bm25_candidates:  # No need to strip extensions again; already handled in BM25 loop
                    if doc_key in pre_embedded_docs:
                        candidate_embedding = pre_embedded_docs[doc_key].to(device)

                        # Apply LPP if enabled
                        if args.lpp:
                            assert candidate_embedding.dim() == 1, "Expected a 1D embedding"
                            candidate_embedding = lpp_module(
                                candidate_embedding.unsqueeze(0).unsqueeze(0))  # Add batch and channel dims
                            candidate_embedding = candidate_embedding.squeeze(0).squeeze(0)  # Remove added dims
                            assert candidate_embedding.dim() == 1, "Expected a 1D embedding after LPP"

                        candidate_embeddings[doc_key] = candidate_embedding
                    else:
                        print("Candidate embedding not found in pre_embedded document folder")

                # Perform reranking
                reranked_candidates = rerank_with_bilstm(query_embedding, candidate_embeddings, model, device)
                reranked_results[query_name] = {doc: score for doc, score in reranked_candidates}
            else:
                print(f"Embedding for query {query_name} not found.")
        except Exception as e:
            print(f"Error reranking query {query_name}: {e}")

    # Save results
    save_results_to_file(bm25_results, f"{args.prefix}_bm25_results.json")
    save_results_to_file(reranked_results, f"{args.prefix}_reranked_results.json")

    k_values = list(range(50, 501, 25))

    # Initialize containers for metrics
    all_metrics = {"bm25": {}, "reranked": {}}  # Separate BM25 and reranked metrics

    for k in k_values:
        bm25_top_k_results = {
            query: {doc: 1 for doc in docs[:k]}
            for query, docs in bm25_results.items()
        }

        reranked_top_k_results = {}  # Initialize the final dictionary
        for query, docs in reranked_results.items():
            # Step 2: Sort documents by their scores in descending order
            sorted_docs = sorted(docs.items(), key=lambda item: item[1], reverse=True)
            # Step 3: Take the top-k results
            top_k_docs = sorted_docs[:k]
            # Step 4: Convert the top-k results back into a dictionary
            top_k_dict = {doc: score for doc, score in top_k_docs}
            # Step 5: Add the top-k dictionary to the final results
            reranked_top_k_results[query] = top_k_dict

        # Calculate metrics for BM25 top-k results
        # print(f"Calculating metrics for BM25 top-{k} results...")
        bm25_metrics = calculate_metrics(
            queries,
            bm25_top_k_results,
            output_file=None  # Disable individual saving
        )
        # print(f"BM25 Top-{k} Metrics:", bm25_metrics)
        all_metrics["bm25"][k] = bm25_metrics  # Store metrics for this `k`

        # Calculate metrics for BiLSTM reranked top-k results
        # print(f"Calculating metrics for BiLSTM reranked top-{k} results...")
        reranked_metrics = calculate_metrics(
            queries,
            reranked_top_k_results,
            output_file=None  # Disable individual saving
        )
        # print(f"BiLSTM Reranked Top-{k} Metrics:", reranked_metrics)
        all_metrics["reranked"][k] = reranked_metrics  # Store metrics for this `k`

    # Save all metrics to a single file
    metrics_file = f"{args.prefix}_all_k_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Saved all metrics to {metrics_file}")

# python get_and_evaluate_results.py --prefix z_testing_testing_08

# python get_and_evaluate_results.py --prefix z_bert_embeddings
# python get_and_evaluate_results.py --prefix z_lpp_embeddings --model-path 2 --lpp

# cdscsd