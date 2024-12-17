from preprocessing_functions import *
from embedding_creation_functions import *
from transformers import (AutoTokenizer, AutoModel)
import os
from tqdm import tqdm

# Get the root directory of the project
def get_project_root():
    """
    Get the root directory of the project (assumes this script is in the 'code' folder).
    """
    current_file_path = os.path.dirname(__file__)  # Current script's directory
    return os.path.abspath(os.path.join(current_file_path, os.pardir))  # One level up (project root)


# Global variables
project_root = get_project_root()

# Define the paths dynamically
case_folder_path_train = os.path.join(project_root, "training_data")  # Training data folder
save_folder_path_train = os.path.join(project_root, "training_data_embeddings_bert")  # Save folder for embeddings

case_folder_path_test = os.path.join(project_root, "test_data")
save_folder_path_test = os.path.join(project_root, "test_data_embeddings_bert")

items_to_scrub = ["<FRAGMENT_SUPPRESSED>", "FRAGMENT_SUPPRESSED", "[Translation]"]

# Load models and tokenizer
model_bert_uncased = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")  # LegalBERT model name
model_sailer = AutoModel.from_pretrained('CSHaitao/SAILER_en')
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")


def process_case_files_to_embeddings(folder_path, save_path, words_to_remove, the_model, the_tokenizer, name00):
    """
    Iterates through a folder of .txt files, applies processing functions to each file,
    and saves paragraph and document embeddings in a specified folder.

    Args:
        folder_path (str): Path to the folder containing .txt files.
        save_path (str): Path to the folder where embeddings will be saved.
        words_to_remove (list of str): List of words to remove using regex.
        the_model: Pre-trained model for generating embeddings.
        the_tokenizer: Tokenizer corresponding to the model.
    """
    # Ensure the input folder exists
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(name00, "w") as file:
        file.write("start.\n")

    # Iterate through all .txt files in the folder
    for file_name in tqdm(os.listdir(folder_path), desc="Processing case files"):

        file_path = os.path.join(folder_path, file_name)
        print(file_name)

        # Read the file content
        content = read_file(file_path)

        # Apply the processing functions
        content = remove_french(content)  # Remove French paragraphs
        content = remove_words_regex(content, words_to_remove)  # Remove specified words
        subsections = split_into_subsections(content)  # Split into subsections

        if len(subsections) < 3:
            with open(name00, "a") as file:
                file.write(f"{file_name} \n")

        paragraph_embeddings = []

        for s in subsections:
            paragraph_embedding = tokenize_and_process(s, the_model, the_tokenizer)
            paragraph_embeddings.append(paragraph_embedding)

        document_embedding = average_cls_embeddings(paragraph_embeddings)

        # Add saving functionality
        file_folder_name = os.path.join(save_path, os.path.splitext(file_name)[0])
        os.makedirs(file_folder_name, exist_ok=True)  # Create a folder for the file

        # Save paragraph embeddings
        paragraph_embeddings_tensor = torch.stack(paragraph_embeddings)  # Convert to tensor
        torch.save(paragraph_embeddings_tensor, os.path.join(file_folder_name, "paragraph_embeddings.pt"))

        # Save document embedding
        torch.save(document_embedding, os.path.join(file_folder_name, "document_embedding.pt"))



# Run the function with dynamic paths
process_case_files_to_embeddings(case_folder_path_train, save_folder_path_train, items_to_scrub, model_bert_uncased, tokenizer, "train_data_no_p_bert")
process_case_files_to_embeddings(case_folder_path_test, save_folder_path_test, items_to_scrub, model_bert_uncased, tokenizer, "test_data_no_p_bert")
