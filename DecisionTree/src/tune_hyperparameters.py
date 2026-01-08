utf-8import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from data_loader import load_data
def tune_hyperparameters():
    print("Loading and preprocessing data for tuning...")
    df = load_data()
    X = df['input_command']
    y = df['predicted_action']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    param_grid = [
        {'n_estimators': 50, 'max_depth': None, 'class_weight': 'balanced'},
        {'n_estimators': 100, 'max_depth': None, 'class_weight': 'balanced'}, 
        {'n_estimators': 200, 'max_depth': None, 'class_weight': 'balanced'},
        {'n_estimators': 100, 'max_depth': 20, 'class_weight': 'balanced'},  
        {'n_estimators': 100, 'max_depth': 50, 'class_weight': 'balanced'},
        {'n_estimators': 100, 'max_depth': None, 'class_weight': None},      
    ]
    results = []
    print(f"\n{'n_est':<10} {'max_depth':<10} {'class_w':<12} | {'Acc':<8} {'F1-Macro':<8}")
    print("-" * 60)
    for params in param_grid:
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            class_weight=params['class_weight'],
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"{params['n_estimators']:<10} {str(params['max_depth']):<10} {str(params['class_weight']):<12} | {acc:.4f}   {f1:.4f}")
        results.append({
            **params,
            : acc,
            : f1
        })
    return results
if __name__ == "__main__":
    tune_hyperparameters()