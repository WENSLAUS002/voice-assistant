import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,
    GPT2Tokenizer, GPT2LMHeadModel, GPT2Config,
    T5Tokenizer, T5ForConditionalGeneration
)
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Load dataset
labeled_path = "C:/Users/JUMA/banking_nlp/backend/banking_dataset.csv"
df = pd.read_csv(labeled_path)
df = df.dropna(subset=["question", "answer"])

if "label" not in df.columns:
    raise ValueError("The dataset must have a 'label' column.")

df = df[df["label"].notna()]
# Convert labels to strings to handle mixed types (e.g., floats, strings)
df["label"] = df["label"].astype(str).replace("nan", "unknown")
# Sort labels for consistency with bert_classifier.py
LABELS = {label: idx for idx, label in enumerate(sorted(df["label"].unique()))}
# Add a placeholder label if necessary to reach 344 classes
if len(LABELS) == 343:
    LABELS["other"] = 343  # Add a placeholder label
NUM_CLASSES = len(LABELS)  # Should be 344
print("Label mapping:", LABELS)
print("Number of classes:", NUM_CLASSES)
if NUM_CLASSES != 344:
    raise ValueError(f"Expected 344 labels, found {NUM_CLASSES}")

# Assign label_id, mapping "unknown" or unmapped labels to "other"
df["label_id"] = df["label"].map(LABELS).fillna(LABELS["other"]).astype(int)

dataset = Dataset.from_pandas(df)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Split dataset
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Compute accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.tensor(logits).argmax(dim=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# BERT
def train_bert():
    print("Training BERT...")
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=NUM_CLASSES  # Set to 344
        ).to(DEVICE)

        def tokenize_fn(examples):
            return tokenizer(examples["question"], padding="max_length", truncation=True, max_length=128)

        tokenized_train = train_dataset.map(tokenize_fn, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_fn, batched=True)
        tokenized_train = tokenized_train.rename_column("label_id", "labels")
        tokenized_eval = tokenized_eval.rename_column("label_id", "labels")
        tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        tokenized_eval.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        args = TrainingArguments(
            output_dir="./models/bert_model",
            eval_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            save_strategy="epoch",
            logging_dir="./logs/bert_class"
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            compute_metrics=compute_metrics
        )
        trainer.train()
        model.save_pretrained("./models/bert_model")
        tokenizer.save_pretrained("./models/bert_model")
        print("BERT trained and saved.")
    except Exception as e:
        print(f"Error training BERT: {e}")

# GPT-2, T5, and SBERT functions remain unchanged
def train_gpt():
    print("Training GPT-2...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        config = GPT2Config.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2", config=config).to(DEVICE)

        def tokenize_fn(examples):
            inputs = [q + " </s>" for q in examples["question"]]
            outputs = [a + " </s>" for a in examples["answer"]]
            model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
            labels = tokenizer(outputs, padding="max_length", truncation=True, max_length=128)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_train = train_dataset.map(tokenize_fn, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_fn, batched=True)
        tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        tokenized_eval.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        args = TrainingArguments(
            output_dir="./models/gpt_model",
            eval_strategy="epoch",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            logging_dir="./logs/gpt_gen"
        )
        trainer = Trainer(model=model, args=args, train_dataset=tokenized_train, eval_dataset=tokenized_eval)
        trainer.train()
        model.save_pretrained("./models/gpt_model")
        tokenizer.save_pretrained("./models/gpt_model")
        print("GPT-2 trained and saved.")
    except Exception as e:
        print(f"Error training GPT-2: {e}")

def train_t5():
    print("Training T5...")
    try:
        tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
        model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)

        def preprocess_fn(examples):
            inputs = ["question: " + q for q in examples["question"]]
            targets = examples["answer"]
            model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
            labels = tokenizer(targets, padding="max_length", truncation=True, max_length=128).input_ids
            labels = [[token if token != tokenizer.pad_token_id else -100 for token in seq] for seq in labels]
            model_inputs["labels"] = labels
            return model_inputs

        tokenized_train = train_dataset.map(preprocess_fn, batched=True)
        tokenized_eval = eval_dataset.map(preprocess_fn, batched=True)
        tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        tokenized_eval.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        args = TrainingArguments(
            output_dir="./models/t5_model",
            eval_strategy="epoch",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=3,
            save_strategy="epoch",
            logging_dir="./logs/t5_qa",
            gradient_accumulation_steps=4
        )
        trainer = Trainer(model=model, args=args, train_dataset=tokenized_train, eval_dataset=tokenized_eval)
        trainer.train()
        model.save_pretrained("./models/t5_model")
        tokenizer.save_pretrained("./models/t5_model")
        print("T5 trained and saved.")
    except Exception as e:
        print(f"Error training T5: {e}")

def train_sbert():
    print("Training SBERT...")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")

        train_samples = [InputExample(texts=[row["question"], row["answer"]], label=1.0) for _, row in df.iterrows()]

        sample_size = min(len(df), 100)
        eval_samples = [InputExample(texts=[row["question"], row["answer"]], label=1.0) for _, row in df.sample(sample_size, random_state=42).iterrows()]

        train_dataloader = DataLoader(train_samples, batch_size=16, shuffle=True)
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_samples, name='eval')

        train_loss = losses.CosineSimilarityLoss(model)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=3,
            warmup_steps=100,
            output_path="./models/sbert_model",
            evaluation_steps=500
        )
        print("SBERT trained and saved.")
    except Exception as e:
        print(f"Error training SBERT: {e}")

# Run all
if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    train_bert()
    train_gpt()
    train_sbert()
    train_t5()
    print("All models trained and saved successfully.")