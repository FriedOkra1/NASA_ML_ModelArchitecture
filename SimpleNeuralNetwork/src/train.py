utf-8import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from model import SimpleNN
DATA_DIR = '../data'
MODEL_DIR = '../models'
RESULTS_DIR = '../results'
INPUT_DIM = 768 
HIDDEN_LAYERS = [512, 256]
OUTPUT_DIM = 0 
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32
NOISE_STD = 0.05 
def load_tensors():
    X_train = torch.load(os.path.join(DATA_DIR, 'X_train.pt'))
    X_test = torch.load(os.path.join(DATA_DIR, 'X_test.pt'))
    y_train = torch.load(os.path.join(DATA_DIR, 'y_train.pt'))
    y_test = torch.load(os.path.join(DATA_DIR, 'y_test.pt'))
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    num_classes = len(le.classes_)
    return X_train, X_test, y_train, y_test, num_classes
def add_noise(features, std=NOISE_STD):
    noise = torch.randn_like(features) * std
    return features + noise
def train_model():
    print("Loading data...")
    X_train_full, X_test, y_train_full, y_test, num_classes = load_tensors()
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    model = SimpleNN(INPUT_DIM, HIDDEN_LAYERS, num_classes, DROPOUT_RATE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = add_noise(inputs)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Val Loss: {epoch_val_loss:.4f} - Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'simple_nn.pth'))
    print("\nEvaluating on Test Set...")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'simple_nn.pth')))
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
    acc = accuracy_score(test_labels, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro')
    try:
        auroc = roc_auc_score(test_labels, test_probs, multi_class='ovr', average='macro')
    except ValueError:
        auroc = 0.0 
    print(f"\nTest Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    joblib.dump(history, os.path.join(RESULTS_DIR, 'training_history.pkl'))
    report_path = '../../report/SimpleNeuralNetwork/hyperparameters.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("Simple Neural Network Training Log\n")
        f.write("==================================\n")
        f.write(f"Model Architecture: {INPUT_DIM} -> {HIDDEN_LAYERS} -> {num_classes}\n")
        f.write(f"Input Features: {INPUT_DIM} (DistilBERT Embeddings)\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Dropout Rate: {DROPOUT_RATE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Noise Std (Augmentation): {NOISE_STD}\n")
        f.write("\nResults:\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUROC: {auroc:.4f}\n")
    if acc < 0.80:
        print("\nWARNING: Accuracy is below 80%. Consider tuning hyperparameters.")
    else:
        print("\nSUCCESS: Accuracy target met.")
if __name__ == "__main__":
    train_model()