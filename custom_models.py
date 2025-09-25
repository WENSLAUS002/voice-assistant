import torch
import torch.nn as nn
from transformers import BertModel, GPT2Model, T5EncoderModel
from sentence_transformers import SentenceTransformer

NUM_CLASSES = 307  # Set this to match your dataset's label range

# BERT Classifier
class CustomBERTClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=NUM_CLASSES):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)

# SBERT Classifier
class SBERTClassifier(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', num_classes=NUM_CLASSES):
        super().__init__()
        self.sbert = SentenceTransformer(model_name)
        self.classifier = nn.Linear(self.sbert.get_sentence_embedding_dimension(), num_classes)

    def forward(self, input_sentences):
        embeddings = self.sbert.encode(input_sentences, convert_to_tensor=True)
        return self.classifier(embeddings)

# GPT-2 Classifier
class GPT2Classifier(nn.Module):
    def __init__(self, model_name='gpt2', num_classes=NUM_CLASSES):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, -1, :]  # Use last token
        return self.classifier(cls_token)

# T5 Classifier
class T5Classifier(nn.Module):
    def __init__(self, model_name='t5-small', num_classes=NUM_CLASSES):
        super().__init__()
        self.t5 = T5EncoderModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.t5.config.d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # Use first token
        return self.classifier(cls_token)
