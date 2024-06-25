import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
from utils import read_multiple_jsonl_files, convert_labels_to_indices


# Configure logging
logging.basicConfig(filename='/root/Multi-Vec-Detection/logs/train_BERT_multi.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Define hyperparameters
MAX_LENGTH = 512
PARTITIAL = 1
LR = 1e-5
BATCH_SIZE = 32
NUM_EPOCHS = 8
LOG_ITER = 100

# Define label to index mapping
label_to_index = {'llama': 0, 'human': 1, 'gptneo': 2, 'gpt3re': 3, 'gpt2': 4}
index_to_label = {v: k for k, v in label_to_index.items()}
num_classes = len(label_to_index)

# Define custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define model class
class AIGTClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(AIGTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 128)  
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        intermediate_output = self.fc1(pooled_output)
        intermediate_output = torch.relu(intermediate_output)
        logits = self.fc2(intermediate_output)
        return logits


# Training function with progress printing
def train_BERT_multi(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, log_iter, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Print progress
            if batch_idx % log_iter == 0:
                logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        scheduler.step()
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        per_class_metrics = precision_recall_fscore_support(all_labels, all_preds, labels=list(label_to_index.values()))
        logging.info("="*50)
        print("="*50)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")
        
        for i, label in index_to_label.items():
            precision, recall, fscore, _ = per_class_metrics[0][i], per_class_metrics[1][i], per_class_metrics[2][i], per_class_metrics[3][i]
            logging.info(f"Train {label}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {fscore:.4f}")
            print(f"Train {label}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {fscore:.4f}")
        logging.info("="*50)
        print("="*50)
        # Evaluate the model on the validation set
        val_loss, val_acc, val_per_class_metrics = evaluate_BERT_multi(model, val_dataloader, criterion, device)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        for i, label in index_to_label.items():
            precision, recall, fscore, _ = val_per_class_metrics[0][i], val_per_class_metrics[1][i], val_per_class_metrics[2][i], val_per_class_metrics[3][i]
            logging.info(f"Val {label}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {fscore:.4f}")
            print(f"Val {label}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {fscore:.4f}")
        logging.info("="*50)
        print("="*50)

# Evaluation function
def evaluate_BERT_multi(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    per_class_metrics = precision_recall_fscore_support(all_labels, all_preds, labels=list(label_to_index.values()))

    return epoch_loss, epoch_acc, per_class_metrics  

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('/root/Multi-Vec-Detection/bert-base-cased')

# Define maximum length
max_length = MAX_LENGTH

# Read data from JSONL files
file_paths = [
    '/root/Multi-Vec-Detection/dataset/en_gpt2_lines.jsonl',
    '/root/Multi-Vec-Detection/dataset/en_gpt3_lines.jsonl',
    '/root/Multi-Vec-Detection/dataset/en_gptneo_lines.jsonl',
    '/root/Multi-Vec-Detection/dataset/en_human_lines.jsonl',
    '/root/Multi-Vec-Detection/dataset/en_llama_lines.jsonl'
]  
texts, labels = read_multiple_jsonl_files(file_paths)
# Convert labels to indices
labels = convert_labels_to_indices(labels, label_to_index)


# Create dataset
dataset = TextDataset(texts, labels, tokenizer, max_length)
# Split dataset into train and test sets 
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# Use a smaller portion of the dataset to speed up training and testing
subset_train_size = int(PARTITIAL * train_size)
subset_test_size = int(PARTITIAL * test_size)
train_dataset, _ = torch.utils.data.random_split(train_dataset, [subset_train_size, train_size - subset_train_size])
test_dataset, _ = torch.utils.data.random_split(test_dataset, [subset_test_size, test_size - subset_test_size])
# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define model
pretrained_model_name = '/root/Multi-Vec-Detection/bert-base-cased'
model = AIGTClassifier(pretrained_model_name, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)


# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
logging.info("="*50)
print("="*50)
logging.info(f"Training Pure BERT with LR-{LR}...")
print(f"Training Pure BERT with LR-{LR}...")
# Train the model
train_BERT_multi(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, NUM_EPOCHS, LOG_ITER, device)
logging.info(f"Training finished!")
print(f"Training finished!")
logging.info("="*50)
print("="*50)
