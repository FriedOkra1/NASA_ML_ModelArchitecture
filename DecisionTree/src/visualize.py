utf-8import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
from data_loader import load_data
from style_config import UMICH_BLUE, UMICH_MAIZE, METRIC_PALETTE
def load_artifacts_and_predict():
    print("Loading data and model...")
    df = load_data()
    X = df['input_command']
    y = df['predicted_action']
    if os.path.exists('../models/rf_model_enhanced.pkl') and os.path.exists('../models/tfidf_vectorizer_enhanced.pkl'):
        model = joblib.load('../models/rf_model_enhanced.pkl')
        vectorizer = joblib.load('../models/tfidf_vectorizer_enhanced.pkl')
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_test_tfidf = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_tfidf)
        return y_test, y_pred, model.classes_
    else:
        raise FileNotFoundError("Enhanced model artifacts not found in ../models/ directory.")
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
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_recall = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    plot_heatmap(np.nan_to_num(cm_recall), classes, 
                 , 
                 )
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_precision = cm.astype('float') / (cm.sum(axis=0)[np.newaxis, :] + 1e-10)
    plot_heatmap(np.nan_to_num(cm_precision), classes, 
                 , 
                 )
    row_sums = cm.sum(axis=1)[:, np.newaxis] + 1e-10
    col_sums = cm.sum(axis=0)[np.newaxis, :] + 1e-10
    prec_mat = cm.astype('float') / col_sums
    rec_mat = cm.astype('float') / row_sums
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_matrix = 2 * (prec_mat * rec_mat) / (prec_mat + rec_mat)
    plot_heatmap(np.nan_to_num(f1_matrix), classes, 
                 , 
                 )
import json
def generate_metric_charts(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    metrics_data = {
        : report['macro avg']['precision'],
        : {cls: report[cls]['precision'] for cls in classes}
    }
    with open('../results/metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print("Saved metrics to ../results/metrics.json")
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
                   , 
                   , 
                   palette=METRIC_PALETTE, hue='Metric')
    for metric in ['Precision', 'Recall', 'F1-Score']:
        folder_map = {
            : 'precision_metrics', 
            : 'recall_metrics', 
            : 'f1_metrics'
        }
        filename = f"class_metrics_{metric.split('-')[0].lower()}_bar.png"
        path = f"../results/{folder_map[metric]}/{filename}"
        df_sub = df_long[df_long['Metric'] == metric].sort_values('Score', ascending=False)
        plot_bar_chart(df_sub, 'Category', 'Score', 
                       , path, color=UMICH_BLUE)
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")
    cmap = LinearSegmentedColormap.from_list("MichiganBlueGrad", ["white", UMICH_BLUE])
    sns.heatmap(df_wide, annot=True, fmt='.1%', cmap=cmap, vmin=0, vmax=1,
                linewidths=0.5, linecolor='gray')
    plt.title('Class Performance Heatmap', fontsize=14)
    plt.tight_layout()
    save_plot('../results/f1_metrics/metrics_heatmap_michigan.png')
def generate_tuning_charts():
    data = [
        {'n_estimators': 50,  'max_depth': 'None', 'class_weight': 'balanced', 'Accuracy': 0.8819, 'F1_Macro': 0.8081, 'Label': 'Faster (50 est)'},
        {'n_estimators': 100, 'max_depth': 'None', 'class_weight': 'balanced', 'Accuracy': 0.8804, 'F1_Macro': 0.8086, 'Label': 'Baseline'},
        {'n_estimators': 200, 'max_depth': 'None', 'class_weight': 'balanced', 'Accuracy': 0.8762, 'F1_Macro': 0.8036, 'Label': 'More Trees (200)'},
        {'n_estimators': 100, 'max_depth': '20',   'class_weight': 'balanced', 'Accuracy': 0.8718, 'F1_Macro': 0.7573, 'Label': 'Shallow (Depth 20)'},
        {'n_estimators': 100, 'max_depth': '50',   'class_weight': 'balanced', 'Accuracy': 0.9067, 'F1_Macro': 0.8111, 'Label': 'Medium Depth (50)'},
        {'n_estimators': 100, 'max_depth': 'None', 'class_weight': 'None',     'Accuracy': 0.9309, 'F1_Macro': 0.8402, 'Label': 'No Class Weights'},
    ]
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars=['Label'], value_vars=['Accuracy', 'F1_Macro'], 
                        var_name='Metric', value_name='Score')
    plot_bar_chart(df_melted, 'Label', 'Score', 
                   , 
                   , 
                   palette="viridis", hue='Metric') 
    depth_data = df[df['Label'].isin(['Shallow (Depth 20)', 'Medium Depth (50)', 'Baseline'])]
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=depth_data, x='max_depth', y='Accuracy', marker='o', label='Accuracy')
    sns.lineplot(data=depth_data, x='max_depth', y='F1_Macro', marker='s', label='F1 Macro')
    plt.title('Impact of Max Depth (with Balanced Weights)', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    save_plot('../results/tuning/impact_of_depth.png')
def generate_final_comparison():
    data = [
        {'Model': 'Baseline (RF)', 'Metric': 'Overall Accuracy', 'Score': 0.8804},
        {'Model': 'Enhanced (RF + N-grams)', 'Metric': 'Overall Accuracy', 'Score': 0.8574},
        {'Model': 'Baseline (RF)', 'Metric': 'Nav Precision', 'Score': 0.38},
        {'Model': 'Enhanced (RF + N-grams)', 'Metric': 'Nav Precision', 'Score': 0.88},
        {'Model': 'Baseline (RF)', 'Metric': 'Nav Recall', 'Score': 0.49},
        {'Model': 'Enhanced (RF + N-grams)', 'Metric': 'Nav Recall', 'Score': 0.39},
        {'Model': 'Baseline (RF)', 'Metric': 'Nav F1', 'Score': 0.43},
        {'Model': 'Enhanced (RF + N-grams)', 'Metric': 'Nav F1', 'Score': 0.54},
    ]
    df = pd.DataFrame(data)
    plot_bar_chart(df, 'Metric', 'Score', 
                   , 
                   , 
                   palette="Paired", hue='Model')
if __name__ == "__main__":
    try:
        y_test, y_pred, classes = load_artifacts_and_predict()
        print("\n--- Generating Confusion Matrices ---")
        generate_confusion_matrices(y_test, y_pred, classes)
        print("\n--- Generating Metric Charts ---")
        generate_metric_charts(y_test, y_pred)
        print("\n--- Generating Tuning & Comparison Charts ---")
        generate_tuning_charts()
        generate_final_comparison()
        print("\nVisualization complete. All assets saved to ../results/ subdirectories.")
    except Exception as e:
        print(f"Visualization failed: {e}")