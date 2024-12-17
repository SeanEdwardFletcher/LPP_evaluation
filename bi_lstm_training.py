from bi_lstm_model import *
from data_set_class import *
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv
import os
import random
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description="BiLSTM Cross Encoder Training")
parser.add_argument("--epochs", type=int, default=300, help="Number of epochs to train")
parser.add_argument("--savemodel", default="./", help="Directory to save the model")
parser.add_argument("--patience", type=int, default=15, help="Early stopping after how many epochs without improvement")
parser.add_argument("--no-cuda", action="store_true", default=False, help="Disables CUDA training")
parser.add_argument("--seed", type=int, default=1, metavar="S", help="Random seed (default: 1)")
parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--train-positive", type=str, default="./LPP_train.json", help="Path to training positive examples")
parser.add_argument("--train-negative", type=str, default="./soft_negative_training_examples.json", help="Path to training negative examples")
parser.add_argument("--val-positive", type=str, default="./LPP_validation.json", help="Path to validation positive examples")
parser.add_argument("--val-negative", type=str, default="./soft_negative_validation_examples.json", help="Path to validation negative examples")
parser.add_argument("--embeddings", type=str, default="bert", help="oscillate  between two embedding folders")
parser.add_argument("--log-name", type=str, required=True, help="Base name for log files, configuration, and saved models")
parser.add_argument("--embedding-type", type=str, default="document_embedding", help="Type of embeddings to use")  # "document_embedding" or "paragraph_embeddings"
parser.add_argument("--lpp", action="store_true", help="Enable Linear Pyramid Pooling (LPP)")  # if --lpp is passed as an argument, llp is set to True
parser.add_argument("--pooling", type=str, default="max", help="pooling over the sequence dimension in the bi-lstm, 'max' or 'avg'")
parser.add_argument("--lp-pooling", type=str, default="avg", help="pooling in the lpp class, 'max' or 'avg'")
parser.add_argument("--lpp-levels", type=str, default="16, 32, 64, 128, 256", help="pooling levels the lpp class")
parser.add_argument("--hidden", type=int, default=256, help="bi-lstm's hidden dimension")

args = parser.parse_args()

# Set seed for reproducibility
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

args.lpp_levels = list(map(int, args.lpp_levels.split(",")))  # turn the input into a list

if not args.no_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

if args.embeddings == "bert":
    args.embeddings = "../training_data_embeddings_bert"
elif args.embeddings == "sailer":
    args.embeddings = "../training_data_embeddings_sailer"
else:
    raise ValueError(f"Unsupported embeddings argument call: {args.embeddings}")

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

# Paths
os.makedirs(args.savemodel, exist_ok=True)
LOG_FILE = os.path.join(args.savemodel, f"{args.log_name}_training_validation_metrics.csv")
CONFIG_FILE = os.path.join(args.savemodel, f"{args.log_name}_training_config.txt")


def log_metrics(log_file, epoch, train_loss, train_accuracy, val_loss, val_accuracy):
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # If file is empty, write the header
            writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        writer.writerow([epoch, train_loss, train_accuracy, val_loss, val_accuracy])


def save_config(config_path):
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "patience": args.patience,
        "seed": args.seed,
        "device": device.type,
        "embedding_type": args.embedding_type,
        "lpp": args.lpp,
        "lpp-levels": args.lpp_levels,
        "hidden-dimension": args.hidden
    }
    with open(config_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for query_embeddings, doc_embeddings, labels in dataloader:
        query_embeddings, doc_embeddings, labels = (
            query_embeddings.to(device),
            doc_embeddings.to(device),
            labels.to(device),
        )

        # Forward pass
        outputs = model(query_embeddings, doc_embeddings)  # Logits
        loss = criterion(outputs, labels.unsqueeze(1))  # BCEWithLogitsLoss expects [batch_size, 1] labels

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

        # Accuracy calculation
        preds = (outputs > 0).float()
        correct_predictions += (preds.squeeze(1) == labels).sum().item()
        total_samples += labels.size(0)

    # Compute average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for query_embeddings, doc_embeddings, labels in dataloader:
            query_embeddings, doc_embeddings, labels = (
                query_embeddings.to(device),
                doc_embeddings.to(device),
                labels.to(device),
            )

            # Forward pass
            outputs = model(query_embeddings, doc_embeddings)  # Logits
            loss = criterion(outputs, labels.unsqueeze(1))  # BCEWithLogitsLoss expects [batch_size, 1] labels

            # Accumulate loss
            total_loss += loss.item()

            # Accuracy calculation
            preds = (outputs > 0).float()
            correct_predictions += (preds.squeeze(1) == labels).sum().item()
            total_samples += labels.size(0)

    # Compute average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy


# Main script
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    EMBEDDING_DIM = sum(args.lpp_levels) if args.lpp else 768  # Match the embedding size of your precomputed embeddings

    # Save configuration
    save_config(CONFIG_FILE)

    # Dataset and DataLoader for training
    train_dataset = LegalTextEmbeddingDataset(
        positive_file=args.train_positive,
        negative_file=args.train_negative,
        embedding_folder=args.embeddings,
        embedding_type=args.embedding_type,
        lpp=args.lpp,
        lpp_levels=args.lpp_levels,
        lpp_pooling=args.lp_pooling,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Dataset and DataLoader for validation
    val_dataset = LegalTextEmbeddingDataset(
        positive_file=args.val_positive,
        negative_file=args.val_negative,
        embedding_folder=args.embeddings,
        embedding_type=args.embedding_type,
        lpp=args.lpp,
        lpp_levels=args.lpp_levels,
        lpp_pooling=args.lp_pooling,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model with LPP
    model = TwoLayerBiLSTMCrossEncoder(embedding_dim=EMBEDDING_DIM, hidden_dim=args.hidden)  #, lpp=args.lpp, pooling=args.pooling, llp_pooling=args.lp_pooling)
    model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Early stopping
    best_val_accuracy = 0.0
    early_stop_counter = 0

    # Training loop with tqdm
    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs"):
        # Train the model
        train_loss, train_accuracy = train(model, train_dataloader, optimizer, criterion, device)

        # Validate the model
        val_loss, val_accuracy = validate(model, val_dataloader, criterion, device)

        # Print results
        print(f"Epoch {epoch}/{EPOCHS}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}")
        print(f"                 Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")

        # Log metrics
        log_metrics(LOG_FILE, epoch, train_loss, train_accuracy, val_loss, val_accuracy)

        model_save_path = os.path.join(args.savemodel, f"{args.log_name}_best_model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path} with validation accuracy: {val_accuracy:.4f}")

        # Save the model if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement in validation accuracy for {early_stop_counter} epoch(s).")

        # Trigger early stopping
        if early_stop_counter >= args.patience:
            print("Early stopping triggered. Training terminated.")
            break

    print(f"Training completed. Best Validation Accuracy: {best_val_accuracy:.4f}. Metrics saved to {LOG_FILE}.")

#  python bi_lstm_training.py --log-name ex_03_bert_paragraph_lpp_max --embedding-type paragraph_embeddings --lpp --lp-pooling max
# python bi_lstm_training.py --log-name ex_03_sailer_paragraph_lpp_max --embeddings sailer --embedding-type paragraph_embeddings --lpp --lp-pooling max

