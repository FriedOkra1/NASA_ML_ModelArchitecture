utf-8import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from data_loader import load_data
from visualize import plot_confusion_matrix
def train_enhanced_model():
    print("Loading data...")
    df = load_data()
    X = df['input_command']
    y = df['predicted_action']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    custom_stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) +                        ['okay', 'roger', 'copy', 'stand', 'by', 'hey', 'uh', 'oh']
    print("Vectorizing with enhanced settings (N-grams, Custom Stop Words)...")
    vectorizer = TfidfVectorizer(
        stop_words=custom_stop_words,
        max_features=7000,      
        ngram_range=(1, 2),     
        min_df=2                
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("Training Enhanced Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=60,           
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_tfidf, y_train)
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nNew Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    os.makedirs('../results', exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, model.classes_, output_path='../results/confusion_matrix_enhanced.png')
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/rf_model_enhanced.pkl')
    joblib.dump(vectorizer, '../models/tfidf_vectorizer_enhanced.pkl')
    print("Enhanced model saved.")
if __name__ == "__main__":
    train_enhanced_model()