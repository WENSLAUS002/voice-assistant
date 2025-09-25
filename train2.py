import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
)
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

# Load dataset
labeled_path = "C:/Users/JUMA/banking_nlp/backend/banking_dataset.csv"
df = pd.read_csv(labeled_path)
df = df.dropna(subset=["question", "answer"])

if "label" not in df.columns:
    raise ValueError("The dataset must have a 'label' column.")

# Filter out 'nan' labels
df = df[df["label"].notna()]
LABELS = {label: idx for idx, label in enumerate(df["label"].unique())}
df["label_id"] = df["label"].map(LABELS)
print("Label mapping:", LABELS)

dataset = Dataset.from_pandas(df)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Split dataset into train and eval
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# T5 for summarization and QA
def train_t5():
    print("Training T5 for QA/Summarization/Text Classification...")
    try:
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
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
            logging_steps=500,
            gradient_accumulation_steps=4
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval
        )
        trainer.train()
        model.save_pretrained("./models/t5_model")
        tokenizer.save_pretrained("./models/t5_model")
        print("T5 model saved to ./models/t5_model")
    except Exception as e:
        print(f"Error training T5: {str(e)}")

# SBERT for semantic search / paraphrase / classification
def train_sbert():
    print("Training SBERT for Semantic Tasks...")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Create training and evaluation samples
        train_samples = [InputExample(texts=[row["question"], row["answer"]], label=1.0) for _, row in df.iterrows()]
        eval_samples = [
            InputExample(texts=[row["question"], row["answer"]], label=1.0)
            for _, row in df.sample(n=min(100, len(df)), random_state=42).iterrows()
        ]

        train_dataloader = DataLoader(train_samples, batch_size=16, shuffle=True)
        train_loss = losses.CosineSimilarityLoss(model)
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_samples, name='eval')

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=3,
            warmup_steps=100,
            output_path="./models/sbert_model",
            evaluation_steps=500
        )
        model.save("./models/sbert_model")
        print("SBERT model saved to ./models/sbert_model")
    except Exception as e:
        print(f"Error training SBERT: {str(e)}")

# Run selected models only
if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    train_sbert()
    train_t5()
    print("SBERT and T5 models trained and saved successfully.")
