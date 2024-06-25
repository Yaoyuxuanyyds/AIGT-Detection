import json
import re
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from torch.utils.data import Dataset



# 保存结果
logging.basicConfig(filename='/root/yyx/Multiple_Detection/logs/train_Final.log', level=logging.INFO, format='%(asctime)s %(message)s')
# 调试参数
FEATURE_DIMENSIONS = [4096, 2048, 1024, 512, 256, 64, 8]



# Define function to preprocess text
def process_text(texts):
    processed_text = re.sub(r'[^\w\s]', '', texts)
    processed_text = processed_text.lower()
    words = word_tokenize(processed_text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

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

# Function to compute TF-IDF features
def compute_tfidf_features(texts, dimention):
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_features = tfidf_vectorizer.fit_transform(texts).toarray()
    if dimention < 10000:
        # Reduce dimensionality of TF-IDF features using PCA
        pca = PCA(n_components=dimention)
        print("PCA for TF-IDF...")
        tfidf_features = pca.fit_transform(tfidf_features)
    
    return tfidf_features

def compute_tfidf_features_without_pca(texts, dimention):
    tfidf_vectorizer = TfidfVectorizer(max_features=dimention)
    tfidf_features = tfidf_vectorizer.fit_transform(texts).toarray()
    
    return tfidf_features

# Dataset class for Doc2Vec
class Dataset_Doc2vec(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label


# Function to compute Doc2Vec features
def compute_doc2vec_features(texts, vector_size, labels):
    processed_texts = [process_text(text) for text in texts]
    dataset_doc2vec = Dataset_Doc2vec(processed_texts, labels)
    tagged_data = [TaggedDocument(words=data, tags=[label]) for data, label in dataset_doc2vec]

    model = Doc2Vec(vector_size=vector_size, window=2, min_count=1, workers=4)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=10)

    doc2vec_features = [model.infer_vector(doc.words) for doc in tagged_data]
    return np.array(doc2vec_features)


# Function to generate Gaussian noise as features
def generate_noise_features(num_samples, feature_dim):
    return np.random.normal(loc=0.0, scale=1.0, size=(num_samples, feature_dim))


# Function to train classifiers and evaluate their accuracy
def train_and_evaluate_classifiers(X_train, X_test, y_train, y_test):
    rf_classifier = RandomForestClassifier()
    log_reg_classifier = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs')

    print("Training the Random Forest classifier...")
    rf_classifier.fit(X_train, y_train)
    rf_pred = rf_classifier.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f'Random Forest Accuracy: {rf_acc}')

    print("Training the Logistic Regression classifier...")
    log_reg_classifier.fit(X_train, y_train)
    lr_pred = log_reg_classifier.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f'Logistic Regression Accuracy: {lr_acc}')

    vote_classifier = VotingClassifier(estimators=[
        ('rf', rf_classifier),
        ('lr', log_reg_classifier),
    ], voting='soft')
    print("Training the Voting classifier...")
    vote_classifier.fit(X_train, y_train)
    vote_pred = vote_classifier.predict(X_test)
    vote_acc = accuracy_score(y_test, vote_pred)
    print(f'Voting Classifier Accuracy: {vote_acc}')

    return rf_acc, lr_acc, vote_acc

def main():
    file_paths = [
        '/root/NLP/qxp/en_gpt2_lines.jsonl',
        '/root/NLP/qxp/en_gpt3_lines.jsonl',
        '/root/NLP/qxp/en_gptneo_lines.jsonl',
        '/root/NLP/qxp/en_human_lines.jsonl',
        '/root/NLP/qxp/en_llama_lines.jsonl'
    ]

    texts, labels_list = read_multiple_jsonl_files(file_paths)
    print(f"Number of samples: {len(labels_list)}")

    labels = convert_labels_to_indices(labels_list)

    for dimension in FEATURE_DIMENSIONS:
        print(f"\nProcessing for feature dimension = {dimension}")
        logging.info(f"\nProcessing for feature dimension = {dimension}")
        tfidf_features = compute_tfidf_features_without_pca(texts, dimension)
        #doc2vec_features = compute_doc2vec_features(texts, dimension, labels_list)
        num_samples = len(labels_list)
        #noise_features = generate_noise_features(num_samples, dimension)

        X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(tfidf_features, labels, test_size=0.2, random_state=42)
        #X_train_doc2vec, X_test_doc2vec, _, _ = train_test_split(doc2vec_features, labels, test_size=0.2, random_state=42)
        #X_train_noise, X_test_noise, _, _ = train_test_split(noise_features, labels, test_size=0.2, random_state=42)

        print(f"\n==== TF-IDF Features (dim={dimension}) ====")
        logging.info(f"\n==== TF-IDF Features (dim={dimension}) ====")
        rf_acc_tfidf, lr_acc_tfidf, vote_acc_tfidf = train_and_evaluate_classifiers(X_train_tfidf, X_test_tfidf, y_train, y_test)

        #print(f"\n==== Doc2Vec Features (dim={dimension}) ====")
        #logging.info(f"\n==== Doc2Vec Features (dim={dimension}) ====")
        #rf_acc_doc2vec, lr_acc_doc2vec, vote_acc_doc2vec = train_and_evaluate_classifiers(X_train_doc2vec, X_test_doc2vec, y_train, y_test)

        #print(f"\n==== Noise Features (dim={dimension}) ====")
        #logging.info(f"\n==== Noise Features (dim={dimension}) ====")
        #rf_acc_noise, lr_acc_noise, vote_acc_noise = train_and_evaluate_classifiers(X_train_noise, X_test_noise, y_train, y_test)

        print("\n==== Final Comparison ====")
        print(f'TF-IDF Random Forest Accuracy (dim={dimension}): {rf_acc_tfidf}')
        print(f'TF-IDF Logistic Regression Accuracy (dim={dimension}): {lr_acc_tfidf}')
        print(f'TF-IDF Voting Classifier Accuracy (dim={dimension}): {vote_acc_tfidf}')
        #print(f'Doc2Vec Random Forest Accuracy (dim={dimension}): {rf_acc_doc2vec}')
        #print(f'Doc2Vec Logistic Regression Accuracy (dim={dimension}): {lr_acc_doc2vec}')
        #print(f'Doc2Vec Voting Classifier Accuracy (dim={dimension}): {vote_acc_doc2vec}')
        #print(f'Noise Random Forest Accuracy (dim={dimension}): {rf_acc_noise}')
        #print(f'Noise Logistic Regression Accuracy (dim={dimension}): {lr_acc_noise}')
        #print(f'Noise Voting Classifier Accuracy (dim={dimension}): {vote_acc_noise}')

        logging.info("\n==== Final Comparison ====")
        logging.info(f'TF-IDF Random Forest Accuracy (dim={dimension}): {rf_acc_tfidf}')
        logging.info(f'TF-IDF Logistic Regression Accuracy (dim={dimension}): {lr_acc_tfidf}')
        logging.info(f'TF-IDF Voting Classifier Accuracy (dim={dimension}): {vote_acc_tfidf}')
        #logging.info(f'Doc2Vec Random Forest Accuracy (dim={dimension}): {rf_acc_doc2vec}')
        #logging.info(f'Doc2Vec Logistic Regression Accuracy (dim={dimension}): {lr_acc_doc2vec}')
        #logging.info(f'Doc2Vec Voting Classifier Accuracy (dim={dimension}): {vote_acc_doc2vec}')
        #logging.info(f'Noise Random Forest Accuracy (dim={dimension}): {rf_acc_noise}')
        #logging.info(f'Noise Logistic Regression Accuracy (dim={dimension}): {lr_acc_noise}')
        #logging.info(f'Noise Voting Classifier Accuracy (dim={dimension}): {vote_acc_noise}')


if __name__ == "__main__":
    main()