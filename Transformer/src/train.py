import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
from data_loader import load_data
MODEL_NAME = 'distilbert-base-uncased'
OUTPUT_DIR = '../NewResults/models'
RESULTS_DIR = '../NewResults/results'
LOG_FILE = '../NewResults/report/hyperparameters.txt'
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
def main():
    print("Loading data...")
    df = load_data()
    labels = df['predicted_action'].unique().tolist()
    labels.sort() 
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    print(f"Classes: {labels}")
    df['label'] = df['predicted_action'].map(label2id)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_dataset = Dataset.from_pandas(train_df[['input_command', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['input_command', 'label']])
    print("Tokenizing...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_function(examples):
        return tokenizer(examples['input_command'], truncation=True, padding=False) 
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # Freeze embeddings
    for param in model.distilbert.embeddings.parameters():
        param.requires_grad = False
    
    # Freeze first 4 layers of the transformer
    for layer in model.distilbert.transformer.layer[:4]:
        for param in layer.parameters():
            param.requires_grad = False

    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"All parameters: {all_params}")
    print(f"Percentage trainable: {100 * trainable_params / all_params:.2f}%")
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",  
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=f"{RESULTS_DIR}/logs",
        logging_steps=10,
        report_to="none", 
        use_cpu=True 
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print("Starting training...")
    train_result = trainer.train()
    print("Evaluating...")
    metrics = trainer.evaluate()
    print(f"Test Set Metrics: {metrics}")
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("DistilBERT Training Log\n")
        f.write("=======================\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Epochs: {training_args.num_train_epochs}\n")
        f.write(f"Batch Size: {training_args.per_device_train_batch_size}\n")
        f.write(f"Learning Rate: {training_args.learning_rate}\n")
        f.write("\nResults:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print("Done!")
if __name__ == "__main__":
    main()