import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np



MAX_LENGTH = 512
PCA_COMPUNENTS = 128
PARTITIAL = 0.1
LR = 2e-5
BATCH_SIZE = 32
NUM_EPOCHS = 3



# Function to read multiple JSONL files and extract texts and labels
def read_multiple_jsonl_files(file_paths):
    texts = []
    labels = []

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                texts.append(data['text'])
                labels.append(data['label'])

    return texts, labels

# Define label to index mapping
label_to_index = {'llama': 0, 'human': 1, 'gptneo': 2, 'gpt3re': 3, 'gpt2': 4}
num_classes = len(label_to_index)

# Convert labels to indices
def convert_labels_to_indices(labels):
    return [label_to_index[label] for label in labels]

# Define custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, tfidf_features, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.tfidf_features = tfidf_features
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
            'tfidf_features': torch.tensor(self.tfidf_features[idx], dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('/root/yyx/Multiple Detection/bert-base-uncased')

# Define maximum length
max_length = MAX_LENGTH

# Read data from JSONL files
file_paths = [
    'yyx/Multiple Detection/qxp/en_gpt2_lines.jsonl',
    'yyx/Multiple Detection/qxp/en_gpt3_lines.jsonl',
    'yyx/Multiple Detection/qxp/en_gptneo_lines.jsonl',
    'yyx/Multiple Detection/qxp/en_human_lines.jsonl',
    'yyx/Multiple Detection/qxp/en_llama_lines.jsonl'
]  # Replace with your actual file paths
texts, labels = read_multiple_jsonl_files(file_paths)


# Convert labels to indices
labels = convert_labels_to_indices(labels)

# Compute TFIDF features
print("TF-IDF featuring...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_features = tfidf_vectorizer.fit_transform(texts).toarray()

# Reduce dimensionality of TFIDF features using PCA
pca = PCA(n_components=PCA_COMPUNENTS)
print("PCA for TF-IDF...")
tfidf_features_reduced = pca.fit_transform(tfidf_features)

# Create dataset
dataset = TextDataset(texts, labels, tokenizer, tfidf_features_reduced, max_length)

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

# Define model class
class AIGTClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(AIGTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.fc1 = nn.Linear(self.bert.config.hidden_size + PCA_COMPUNENTS, 128)  # Adjust input size to include TFIDF features
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask, tfidf_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Standardize the BERT and TFIDF features
        standardizer = StandardScaler()
        pooled_output = standardizer.fit_transform(pooled_output.cpu().detach().numpy())
        tfidf_features = standardizer.fit_transform(tfidf_features.cpu().detach().numpy())

        # Concatenate BERT and TFIDF features
        concat_features = torch.tensor(
            np.concatenate((pooled_output, tfidf_features), axis=1),
            dtype=torch.float,
            device=input_ids.device
        )

        intermediate_output = self.fc1(concat_features)
        intermediate_output = torch.relu(intermediate_output)
        logits = self.fc2(intermediate_output)
        return logits

# Training function with progress printing
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tfidf_features = batch['tfidf_features'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, tfidf_features=tfidf_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        scheduler.step()
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")

        # Evaluate the model on the validation set
        val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

# Evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tfidf_features = batch['tfidf_features'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, tfidf_features=tfidf_features)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc

# Define model
pretrained_model_name = '/root/yyx/Multiple Detection/bert-base-uncased'
model = AIGTClassifier(pretrained_model_name, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
num_epochs = NUM_EPOCHS

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("Training...")
# Train the model
train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs, device)
