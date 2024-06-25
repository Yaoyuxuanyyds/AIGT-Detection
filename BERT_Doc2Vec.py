import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.decomposition import PCA
import logging
from utils import read_multiple_jsonl_files, convert_labels_to_indices, train_concat
from No_bert_classification import compute_doc2vec_features




# 配置根日志记录器
logging.basicConfig(
    filename='/root/NLP/YYX/NLP-AI-Detection/Multiple_Detection/logs/train_BERT_Dov2Vec_Multidimension.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

# 获取根日志记录器（这是可选的，只需确保日志记录器配置正确）
root_logger = logging.getLogger()


root_logger.info('This is a data of bert+doc2vec---dimension:2048/8.')
MAX_LENGTH = 512
PARTITIAL = 1
LR = 1e-5
BATCH_SIZE = 32
NUM_EPOCHS = 12
LOG_ITER = 50
#Doc2Vec_DIMENSIONs = [4096, 2048, 1024, 512, 256, 64, 8]
Doc2Vec_DIMENSIONs = [2048, 8,2048, 8]

# Define custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, extra_features, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.extra_features = extra_features
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
            'extra_features': torch.tensor(self.extra_features[idx], dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define model class
class AIGTClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes, extra_dimension):
        super(AIGTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.fc1 = nn.Linear(self.bert.config.hidden_size + extra_dimension, 128)  
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask, extra_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # # Standardize the BERT and TFIDF features
        # standardizer = StandardScaler()
        # pooled_output = standardizer.fit_transform(pooled_output.cpu().detach().numpy())
        # extra_features = standardizer.fit_transform(extra_features.cpu().detach().numpy())

        concat_features = torch.cat((pooled_output, extra_features), dim=1)
        intermediate_output = self.fc1(concat_features)
        intermediate_output = torch.relu(intermediate_output)
        logits = self.fc2(intermediate_output)
        return logits

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('/root/NLP/google-bert/bert-base-cased')

# Define maximum length
max_length = MAX_LENGTH

# Read data from JSONL files
file_paths = [
    '/root/NLP/qxp/en_gpt2_lines.jsonl',
    '/root/NLP/qxp/en_gpt3_lines.jsonl',
    '/root/NLP/qxp/en_gptneo_lines.jsonl',
    '/root/NLP/qxp/en_human_lines.jsonl',
    '/root/NLP/qxp/en_llama_lines.jsonl'
]

texts, labels = read_multiple_jsonl_files(file_paths)
# Convert labels to indices
# Define label to index mapping
label_to_index = {'llama': 0, 'human': 1, 'gptneo': 2, 'gpt3re': 3, 'gpt2': 4}
num_classes = len(label_to_index)
labels = convert_labels_to_indices(labels, label_to_index)


# Training
for DIMENSION in Doc2Vec_DIMENSIONs:

    # Compute TFIDF features
    print("Doc2Vec featuring...")
    doc2vec_features = compute_doc2vec_features(texts, DIMENSION, labels)
    print(f"Have TF-IDF features with shape: {doc2vec_features.shape}.")

    #print(f"First feature: \n {doc2vec_features[0]}")

    # Create dataset
    dataset = TextDataset(texts, labels, tokenizer, doc2vec_features, max_length)
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
    pretrained_model_name = '/root/NLP/google-bert/bert-base-cased'
    model = AIGTClassifier(pretrained_model_name, num_classes, DIMENSION)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)


    # Move model to GPU if available
    device = 0
    model.to(device)
    logging.info("="*50)
    print("="*50)
    logging.info(f"Training BERT + Doc2Vec features with Dimension-{DIMENSION}...")
    print(f"Training BERT + Doc2Vec features with Dimension-{DIMENSION}...")
    # Train the model
    train_concat(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, NUM_EPOCHS, LOG_ITER, device)
    logging.info(f"Training finished!")
    print(f"Training finished!")
    logging.info("="*50)
    print("="*50)