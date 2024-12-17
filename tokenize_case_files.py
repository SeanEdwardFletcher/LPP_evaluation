"""
this file is for tokenizing the case files using a bert-based tokenizer.
By running this file on the folder of case files (.txt files) the french will be removed
the text in the files will be tokenized by LegalBERT-uncased
then each file will be saved as space-separated tokens in a new folder

pyterrier uses a space-based tokenization method for indexing
doing this to all the case files beforehand will guarantee that pyterrier will use the same tokenization
methods by whichever model you choose to use with it.

it would be a lot faster, but the french removal part takes more time...
"""

# imports
import os
from transformers import AutoTokenizer
from preprocessing_functions import remove_french
from tqdm import tqdm


model_name = "nlpaueb/legal-bert-base-uncased"  # LegalBERT model name
the_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Path to the folder containing the text documents

train_folder = r"C:\Users\fletc\OneDrive\Desktop\MS_ComSci\Independent_Study_F24\COLIEE_dataset\task1_train_files_2024\task1_train_files_2024"
train_tokenized_folder = r"C:\Users\fletc\OneDrive\Desktop\MS_ComSci\Independent_Study_F24\COLIEE_dataset\task1_train_files_2024\task1_train_files_2024_tokenized"

path_to_text_folder = r"C:\Users\fletc\OneDrive\Desktop\MS_ComSci\Independent_Study_F24\COLIEE_dataset\task1_test_files_2024\task1_test_files_2024"
path_to_text_folder_tokenized = r"C:\Users\fletc\OneDrive\Desktop\MS_ComSci\Independent_Study_F24\COLIEE_dataset\task1_test_files_2024\task1_test_files_2024_tokenized"



def tokenize_and_save_documents(input_folder, output_folder, tokenizer):
    """
    Tokenizes .txt files in the input folder using the provided tokenizer
    and saves the tokenized documents to the output folder.

    Args:
        input_folder (str): Path to the folder containing .txt files.
        output_folder (str): Path to the folder where tokenized files will be saved.
        tokenizer: A tokenizer object (e.g., from transformers) to use for tokenization.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of .txt files in the input folder
    txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

    # Iterate through each .txt file with a progress bar
    for filename in tqdm(txt_files, desc="Processing Files", unit="file"):
        # Read the content of the file
        input_file_path = os.path.join(input_folder, filename)
        with open(input_file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        content_no_french = remove_french(text_content)  # from preprocessing_functions.py

        # Tokenize the content using the provided tokenizer
        tokens = tokenizer.tokenize(content_no_french)
        tokenized_text = " ".join(tokens)  # Convert tokens back into a space-separated string for pyterrier indexing

        # Save the tokenized content to a new file in the output folder
        output_file_path = os.path.join(output_folder, filename)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(tokenized_text)


tokenize_and_save_documents(train_folder, train_tokenized_folder, tokenizer=the_tokenizer)
tokenize_and_save_documents(path_to_text_folder, path_to_text_folder_tokenized, tokenizer=the_tokenizer)
