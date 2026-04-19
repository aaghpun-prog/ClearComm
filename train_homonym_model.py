import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from modules.wic_loader import get_wic_hf_dataset
import evaluate

# Path to the WiC dataset
DATA_DIR = "WiC_dataset"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "saved_models/wic_model"

def train():
    # 1. Load the WiC dataset
    dataset_dict = get_wic_hf_dataset(DATA_DIR)
    if 'train' not in dataset_dict or 'dev' not in dataset_dict:
        print("Error: WiC dataset not found or incomplete in WiC_dataset directory.")
        return

    # 2. Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. Preprocess the dataset
    def preprocess_function(examples):
        # WiC is a sentence pair task: (sentence1, sentence2)
        # We also have the target word, but for simple classification, 
        # BERT handles the pair. More advanced versions might mask the target word.
        return tokenizer(
            examples["sentence1"], 
            examples["sentence2"],
            truncation=True,
            padding=True,
            max_length=128
        )

    tokenized_datasets = {}
    for split in dataset_dict:
        tokenized_datasets[split] = dataset_dict[split].map(preprocess_function, batched=True)

    # 4. Load the model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # 5. Define metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir="./wic_checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # 8. Train the model
    print("Starting training on WiC dataset...")
    trainer.train()

    # 9. Save the model
    print(f"Saving fine-tuned model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete!")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    
    train()
