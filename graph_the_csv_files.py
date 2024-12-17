import pandas as pd
import matplotlib.pyplot as plt

# Path to the CSV file
# csv_file = "ex_01_bert_max_pool_training_validation_metrics.csv"
# csv_file = "ex_01_bert_avg_pool_training_validation_metrics.csv"
# csv_file = "ex_01_sailer_avg_pool_training_validation_metrics.csv"
# csv_file = "ex_01_sailer_max_pool_training_validation_metrics.csv"
# csv_file = "ex_02_bert_lpp_avg_training_validation_metrics.csv"
# csv_file = "ex_02_sailer_lpp_avg_training_validation_metrics.csv"
# csv_file = "ex_02_bert_lpp_max_training_validation_metrics.csv"
# csv_file = "ex_02_sailer_lpp_max_training_validation_metrics.csv"
# csv_file = "ex_03_bert_paragraph_lpp_avg_training_validation_metrics.csv"
# csv_file = "ex_03_sailer_paragraph_lpp_avg_training_validation_metrics.csv"
# csv_file = "ex_03_bert_paragraph_lpp_max_training_validation_metrics.csv"
# csv_file = "ex_03_sailer_paragraph_lpp_max_training_validation_metrics.csv"



title_part = "Bert Average Pooling"

# Load the CSV data into a DataFrame
data = pd.read_csv(csv_file)

# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(data['epoch'], data['train_loss'], label='Train Loss', marker='o')
plt.plot(data['epoch'], data['val_loss'], label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'{title_part} Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot Training and Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(data['epoch'], data['train_accuracy'], label='Train Accuracy', marker='o')
plt.plot(data['epoch'], data['val_accuracy'], label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'{title_part} Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()
