#added proper formatting this time haha

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from matplotlib.colors import LinearSegmentedColormap
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import sys

# Add src to python path to import modules
sys.path.append('../src')
from data_loader import load_data
from style_config import UMICH_BLUE, UMICH_MAIZE, METRIC_PALETTE

def load_artifacts_and_predict():
    print("Loading data and model...")
    df = load_data()
    labels = df['predicted_action'].unique().tolist()
    labels.sort()
    label2id = {label: i for i, label in enumerate(labels)}
    df['label'] = df['predicted_action'].map(label2id)
    
    _, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    X_test = test_df['input_command'].tolist()
    y_test = test_df['label'].tolist()
    
    # Path is relative to where script is run
    model_path = 'models'
    
    if os.path.exists(model_path):
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        
        print("Running predictions...")
        inputs = tokenizer(X_test, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits
        y_pred = torch.argmax(logits, dim=-1).tolist()
        
        id2label = {i: label for label, i in label2id.items()}
        y_test_labels = [id2label[i] for i in y_test]
        y_pred_labels = [id2label[i] for i in y_pred]
        
        return y_test_labels, y_pred_labels, labels
    else:
        raise FileNotFoundError(f"Model artifacts not found in {model_path}")

def save_plot(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

def plot_heatmap(matrix, classes, title, output_path, cmap_colors=None):
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    if cmap_colors:
        cmap = LinearSegmentedColormap.from_list("CustomCmap", cmap_colors)
    else:
        cmap = LinearSegmentedColormap.from_list("MichiganBlueGrad", ["white", UMICH_BLUE])
        
    sns.heatmap(matrix, annot=True, fmt='.1%', cmap=cmap, 
                xticklabels=classes, yticklabels=classes, vmin=0, vmax=1,
                linewidths=0.5, linecolor='gray')
    
    plt.title(title, fontsize=16)
    plt.ylabel('True Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.tight_layout()
    save_plot(output_path)

def plot_bar_chart(df, x_col, y_col, title, output_path, color=None, palette=None, hue=None):
    plt.figure(figsize=(10, 6) if not hue else (12, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(data=df, x=x_col, y=y_col, hue=hue, color=color, palette=palette)
    plt.title(title, fontsize=16)
    plt.ylim(0, 1.15)
    plt.ylabel(y_col if not hue else 'Score', fontsize=12)
    plt.xlabel(x_col, fontsize=12)
    
    if hue:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
    plt.xticks(rotation=45, ha='right')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
        
    plt.tight_layout()
    if not hue:
        plt.subplots_adjust(bottom=0.25)
    save_plot(output_path)

def generate_confusion_matrices(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    # Recall Matrix (Row-normalized)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_recall = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    plot_heatmap(np.nan_to_num(cm_recall), classes, 
                 "Recall Matrix", 
                 "results/recall_matrix.png")

    # Precision Matrix (Column-normalized)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_precision = cm.astype('float') / (cm.sum(axis=0)[np.newaxis, :] + 1e-10)
    plot_heatmap(np.nan_to_num(cm_precision), classes, 
                 "Precision Matrix", 
                 "results/precision_matrix.png")

    # F1 Matrix (Harmonic Mean)
    row_sums = cm.sum(axis=1)[:, np.newaxis] + 1e-10
    col_sums = cm.sum(axis=0)[np.newaxis, :] + 1e-10
    prec_mat = cm.astype('float') / col_sums
    rec_mat = cm.astype('float') / row_sums
    
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_matrix = 2 * (prec_mat * rec_mat) / (prec_mat + rec_mat)
    plot_heatmap(np.nan_to_num(f1_matrix), classes, 
                 "F1 Matrix", 
                 "results/f1_matrix.png")

def generate_metric_charts(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    metrics_data = {
        "macro_precision": report['macro avg']['precision'],
        "class_precision": {cls: report[cls]['precision'] for cls in classes}
    }
    
    with open('results/metrics.json', 'w') as f:
        import json
        json.dump(metrics_data, f, indent=4)
    print("Saved metrics to results/metrics.json")
    
    data_long = [] 
    data_wide = [] 
    
    for cls in classes:
        p = report[cls]['precision']
        r = report[cls]['recall']
        f1 = report[cls]['f1-score']
        
        data_wide.append({'Category': cls, 'Precision': p, 'Recall': r, 'F1-Score': f1})
        data_long.extend([
            {'Category': cls, 'Metric': 'Precision', 'Score': p},
            {'Category': cls, 'Metric': 'Recall', 'Score': r},
            {'Category': cls, 'Metric': 'F1-Score', 'Score': f1}
        ])
        
    df_wide = pd.DataFrame(data_wide).set_index('Category')
    df_long = pd.DataFrame(data_long)
    
    plot_bar_chart(df_long, 'Category', 'Score', 
                   "Class Metrics Breakdown", 
                   "results/class_metrics_bar.png", 
                   palette=METRIC_PALETTE, hue='Metric')
                   
    for metric in ['Precision', 'Recall', 'F1-Score']:
        folder_map = {
            'Precision': 'precision_metrics', 
            'Recall': 'recall_metrics', 
            'F1-Score': 'f1_metrics'
        }
        filename = f"class_metrics_{metric.split('-')[0].lower()}_bar.png"
        path = f"results/{folder_map[metric]}/{filename}"
        
        df_sub = df_long[df_long['Metric'] == metric].sort_values('Score', ascending=False)
        plot_bar_chart(df_sub, 'Category', 'Score', 
                       f"{metric} by Class", path, color=UMICH_BLUE)
                       
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")
    cmap = LinearSegmentedColormap.from_list("MichiganBlueGrad", ["white", UMICH_BLUE])
    sns.heatmap(df_wide, annot=True, fmt='.1%', cmap=cmap, vmin=0, vmax=1,
                linewidths=0.5, linecolor='gray')
    plt.title('Class Performance Heatmap', fontsize=14)
    plt.tight_layout()
    save_plot('results/f1_metrics/metrics_heatmap_michigan.png')
    
    return accuracy_score(y_test, y_pred)

def generate_final_comparison(current_acc):
    data = [
        {'Model': 'Decision Tree (Baseline)', 'Metric': 'Accuracy', 'Score': 0.8804},
        {'Model': 'Decision Tree (Best)',     'Metric': 'Accuracy', 'Score': 0.9309},
        {'Model': 'DistilBERT (Transformer)', 'Metric': 'Accuracy', 'Score': current_acc},
    ]
    df = pd.DataFrame(data)
    
    plot_bar_chart(df, 'Model', 'Score', 
                   "Model Comparison", 
                   "results/model_comparison.png", 
                   palette="viridis") 

if __name__ == "__main__":
    try:
        y_test, y_pred, classes = load_artifacts_and_predict()
        
        print("\n--- Generating Confusion Matrices ---")
        generate_confusion_matrices(y_test, y_pred, classes)
        
        print("\n--- Generating Metric Charts ---")
        current_acc = generate_metric_charts(y_test, y_pred)
        
        print("\n--- Generating Comparison Charts ---")
        generate_final_comparison(current_acc)
        
        print("\nVisualization complete. All assets saved to results/ subdirectories.")
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
