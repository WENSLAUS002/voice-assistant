from transformers import (
    BertTokenizer, BertForSequenceClassification,
    GPT2Tokenizer, GPT2LMHeadModel,
    T5Tokenizer, T5ForConditionalGeneration
)
from sentence_transformers import SentenceTransformer

# BERT for intent classification
print("Downloading BERT...")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model.save_pretrained("./colab/bert_model")
bert_tokenizer.save_pretrained("./colab/bert_model")

# GPT-2 for response generation
print("Downloading GPT-2...")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model.save_pretrained("./colab/gpt_model")
gpt_tokenizer.save_pretrained("./colab/gpt_model")

# T5 for summarization
print("Downloading T5...")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model.save_pretrained("./colab")
t5_tokenizer.save_pretrained("./colab/t5_model")

# SBERT for semantic similarity
print("Downloading SBERT...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
sbert_model.save("./colab/sbert_model")

print("All models downloaded and saved successfully.")
