utf-8import os
import torch
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import load_data
import joblib
MODEL_NAME = 'distilbert-base-uncased'
BATCH_SIZE = 32
MAX_LEN = 64
DATA_OUTPUT_DIR = '../data'
MODEL_OUTPUT_DIR = '../models'
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
def generate_embeddings():
    print("Loading data...")
    df = load_data()
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['predicted_action'])
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    joblib.dump(le, os.path.join(MODEL_OUTPUT_DIR, 'label_encoder.pkl'))
    print(f"Saved LabelEncoder. Classes: {le.classes_}")
    X = df['input_command'].tolist()
    y = df['label'].tolist()
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Loading {MODEL_NAME}...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertModel.from_pretrained(MODEL_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    def compute_embeddings(text_list):
        embeddings = []
        for i in tqdm(range(0, len(text_list), BATCH_SIZE), desc="Computing Embeddings"):
            batch_text = text_list[i : i + BATCH_SIZE]
            encoded_input = tokenizer(
                batch_text, 
                padding=True, 
                truncation=True, 
                max_length=MAX_LEN, 
                return_tensors='pt'
            )
            input_ids = encoded_input['input_ids'].to(device)
            attention_mask = encoded_input['attention_mask'].to(device)
            with torch.no_grad():
                model_output = model(input_ids, attention_mask=attention_mask)
            sentence_embeddings = mean_pooling(model_output, attention_mask)
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            embeddings.append(sentence_embeddings.cpu())
        return torch.cat(embeddings, dim=0)
    print("Generating Training Embeddings...")
    X_train_emb = compute_embeddings(X_train_text)
    print("Generating Test Embeddings...")
    X_test_emb = compute_embeddings(X_test_text)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
    torch.save(X_train_emb, os.path.join(DATA_OUTPUT_DIR, 'X_train.pt'))
    torch.save(X_test_emb, os.path.join(DATA_OUTPUT_DIR, 'X_test.pt'))
    torch.save(y_train_tensor, os.path.join(DATA_OUTPUT_DIR, 'y_train.pt'))
    torch.save(y_test_tensor, os.path.join(DATA_OUTPUT_DIR, 'y_test.pt'))
    print(f"Saved embeddings to {DATA_OUTPUT_DIR}")
    print(f"X_train shape: {X_train_emb.shape}")
    print(f"X_test shape: {X_test_emb.shape}")
if __name__ == "__main__":
    generate_embeddings()