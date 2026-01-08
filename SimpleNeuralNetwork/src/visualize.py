utf-8import os
import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
from style_config import UMICH_BLUE, UMICH_MAIZE, METRIC_PALETTE
from model import SimpleNN
DATA_DIR = '../data'
MODEL_DIR = '../models'
RESULTS_DIR = '../results'
INPUT_DIM = 768
HIDDEN_LAYERS = [512, 256]
DROPOUT_RATE = 0.3 
def load_artifacts_and_predict():
    print("Loading data and model...")
    X_test = torch.load(os.path.join(DATA_DIR, 'X_test.pt'))
    y_test = torch.load(os.path.join(DATA_DIR, 'y_test.pt'))
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    classes = le.classes_
    num_classes = len(classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    model = SimpleNN(INPUT_DIM, HIDDEN_LAYERS, num_classes, DROPOUT_RATE).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'simple_nn.pth'), map_location=device))
    model.eval()
    X_test = X_test.to(device)
    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    return y_test.numpy(), preds, classes
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
    cm = confusion_matrix(y_test, y_pred)
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
def generate_metric_charts(y_test, y_pred, classes):
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    report_classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    metrics_data = {
        : report['macro avg']['precision'],
        : {cls: report[cls]['precision'] for cls in report_classes}
    }
    with open('../results/metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print("Saved metrics to ../results/metrics.json")
    data_long = []
    for cls in report_classes:
        p = report[cls]['precision']
        r = report[cls]['recall']
        f1 = report[cls]['f1-score']
        data_long.extend([
            {'Category': cls, 'Metric': 'Precision', 'Score': p},
            {'Category': cls, 'Metric': 'Recall', 'Score': r},
            {'Category': cls, 'Metric': 'F1-Score', 'Score': f1}
        ])
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
        path = f"{RESULTS_DIR}/{folder_map[metric]}/{filename}"
        df_sub = df_long[df_long['Metric'] == metric].sort_values('Score', ascending=False)
        plot_bar_chart(df_sub, 'Category', 'Score', 
                       , path, color=UMICH_BLUE)
def plot_training_history():
    history_path = os.path.join(RESULTS_DIR, 'training_history.pkl')
    if not os.path.exists(history_path):
        print("No history found.")
        return
    history = joblib.load(history_path)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', color=UMICH_BLUE)
    plt.plot(epochs, history['val_loss'], label='Val Loss', color=UMICH_MAIZE)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    save_plot(f'{RESULTS_DIR}/tuning/loss_curve.png')
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_acc'], label='Val Accuracy', color=METRIC_PALETTE[2])
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    save_plot(f'{RESULTS_DIR}/tuning/accuracy_curve.png')
if __name__ == "__main__":
    try:
        y_test, y_pred, classes = load_artifacts_and_predict()
        print("\n--- Generating Confusion Matrices ---")
        generate_confusion_matrices(y_test, y_pred, classes)
        print("\n--- Generating Metric Charts ---")
        generate_metric_charts(y_test, y_pred, classes)
        print("\n--- Plotting Training History ---")
        plot_training_history()
        print("\nVisualization complete.")
    except Exception as e:
        print(f"Visualization failed: {e}")