# Evaluating Linear Pyramid Pooling: A Novel Approach for Legal Case Retrieval

This repository contains all the code for preprocessing case law text files, training and evaluating a custom **Bi-LSTM Cross-Encoder** model, and generating result graphs

## Table of Contents

- [Installation](#installation)
- [Steps to Run](#steps-to-run)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Significance Testing and Graph Generation](#significance-testing-and-graph-generation)

---

## Installation

To run this code, you need access to the [COLIEE-2024 Task 1 dataset](https://coliee.org/application/caseMemorandumWaiver). Make sure you have the dataset downloaded before proceeding.

### Required Libraries

Install the following Python libraries:

```
pip install torch tqdm langdetect transformers sklearn scipy pandas matplotlib rank_bm25
```

Other dependencies (standard libraries):  
`re`, `argparse`, `csv`, `json`, `os`, `time`

### Repository Setup

Clone this repository to your machine:

```
git clone <repository-url>
cd <repository-directory>
```

Ensure file paths and directories are updated to match your local setup.

---

## Steps to Run

### 1. **Get the Dataset**
   - Download the dataset from the [COLIEE website](https://coliee.org/application/caseMemorandumWaiver).

### 2. **Preprocess the Case Files**
   - **Modify file paths** in `create_and_save_embeddings.py`.
   - Run the script to create embeddings:

     ```
     python create_and_save_embeddings.py
     ```

   - Outputs:
     - Four folders containing embeddings (train/test) using Legal-BERT and SAILER.

---

### 3. **Tokenize the Case Files**
   - **Update file paths** in `tokenize_case_files.py`.
   - Run the script:

     ```
     python tokenize_case_files.py
     ```

   - Outputs:
     - Folders for tokenized training and test data.

---

### 4. **Generate Negative Examples**
   - Update file paths in `BM25.py` (JSON file paths for COLIEE data and tokenized training data).
   - Run BM25:

     ```
     python BM25.py
     ```

   - Use the BM25 results to generate soft negatives:

     ```
     python create_soft_negative.py
     ```

   - Outputs:
     - JSON file with BM25 results.
     - JSON file with negative training examples.

---

### 5. **Split Training Data**
   *(Optional: skip this if submitting to COLIEE without additional splits)*

   - Update file paths in `training_data_split_80_10_10.py`.
   - Run the script to split the dataset:

     ```
     python training_data_split_80_10_10.py
     ```

   - Outputs:
     - Train, validation, and test JSON files (default split: 80/10/10).

---

### 6. **Train the Bi-LSTM Model**
   - Update the default file paths in `bi_lstm_training.py` (check lines 48–53 for embedding paths).
   - Run the script with desired arguments:

     ```
     python bi_lstm_training.py --arg1 value1 --arg2 value2 ...
     ```

   - Outputs:
     - Configuration file (`.txt`) and training metrics (`.csv`).

---

### 7. **Evaluate the Models**
   - Update file paths in `get_and_evaluate_results.py` (lines 183–208).
   - Run the script with desired arguments:

     ```
     python get_and_evaluate_results.py --arg1 value1 --arg2 value2 ...
     ```

   - Outputs:
     - BM25 results (`.json`).
     - Reranked results (`.json`).
     - Evaluation metrics (`.json`).

---

### 8. **Significance Testing**
   - Update file paths in `Significance_testing.py`.
   - Tailor the script for the number of models being compared.
   - Run the script:

     ```
     python Significance_testing.py
     ```

   - Outputs:
     - Results printed in the terminal.

---

## Optional: Graph Generation

### Training Process Graphs
   - Update file paths in `graph_the_csv_files.py`.
   - Run the script:

     ```
     python graph_the_csv_files.py
     ```

### Model Performance Graphs
   - Update file paths in `graph_the_json_files.py`.
   - Run the script:

     ```
     python graph_the_json_files.py
     ```

---

## Notes
- Replace placeholder paths in each script before running.
- All scripts are designed to be run sequentially for a smooth workflow.
- For detailed explanations of arguments, refer to the comments in each script.

---

## Contact

For questions or issues, please contact:  


---

