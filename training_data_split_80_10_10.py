import json
import random
import os

def split_dataset(json_file_path, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    # Set the random seed for reproducibility
    random.seed(seed)

    # Load the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Shuffle the data to randomize the splits
    keys = list(data.keys())
    random.shuffle(keys)

    # Calculate split indices
    total = len(keys)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Split the data
    train_keys = keys[:train_end]
    val_keys = keys[train_end:val_end]
    test_keys = keys[val_end:]

    train_data = {key: data[key] for key in train_keys}
    val_data = {key: data[key] for key in val_keys}
    test_data = {key: data[key] for key in test_keys}

    # Save the splits to files
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'LPP_train.json'), 'w') as train_file:
        json.dump(train_data, train_file, indent=4)

    with open(os.path.join(output_dir, 'LPP_validation.json'), 'w') as val_file:
        json.dump(val_data, val_file, indent=4)

    with open(os.path.join(output_dir, 'LPP_test.json'), 'w') as test_file:
        json.dump(test_data, test_file, indent=4)

    print(f"Dataset split completed: {len(train_keys)} training, {len(val_keys)} validation, {len(test_keys)} test samples.")

# Example usage
json_file_path = r"C:\Users\fletc\OneDrive\Desktop\MS_ComSci\Independent_Study_F24\COLIEE_dataset\task1_train_labels_2024.json"
output_dir = r"C:\Users\fletc\OneDrive\Desktop\MS_ComSci\Independent_Study_F24\COLIEE_dataset"
split_dataset(json_file_path, output_dir)
