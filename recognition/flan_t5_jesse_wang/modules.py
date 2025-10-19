import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

'''
https://www.datacamp.com/tutorial/flan-t5-tutorial
https://www.kaggle.com/code/paultimothymooney/fine-tune-flan-t5-with-peft-lora-deeplearning-ai
'''

MODEL_NAME = "google/flan-t5-small"

class BioLaySummT5Flan(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """
        Generate sequences (summaries) using the underlying seq2seq model.
        """
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str):
        instance = cls.__new__(cls)
        instance.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        return instance

    def to(self, device):
        self.model.to(device)
        return self

    def train(self):
        """Set the model in training mode."""
        self.model.train(True)
        return self

    def eval(self):
        """Set the model to evaluation mode."""
        self.model.train(False)
        return self