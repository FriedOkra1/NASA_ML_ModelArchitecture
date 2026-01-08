utf-8import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from data_loader import load_data
from visualize import plot_confusion_matrix, plot_feature_importance
def train_and_evaluate():
    print("Loading data...")
    df = load_data()
    X = df['input_command']
    y = df['predicted_action']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,  
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_tfidf, y_train)
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    classes = model.classes_
    plot_confusion_matrix(y_test, y_pred, classes, output_path='../results/confusion_matrix.png')
    plot_feature_importance(model, vectorizer, output_path='../results/feature_importance.png')
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/rf_model.pkl')
    joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')
    print("Model and vectorizer saved to ../models/ directory.")
    if acc < 0.80:
        print("\nWARNING: Accuracy is below 80%. Consider tuning hyperparameters.")
    else:
        print("\nSUCCESS: Accuracy target met.")
if __name__ == "__main__":
    train_and_evaluate()