import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

'''
https://www.datacamp.com/tutorial/flan-t5-tutorial
https://www.kaggle.com/code/paultimothymooney/fine-tune-flan-t5-with-peft-lora-deeplearning-ai
'''

class BioLaySummT5Flan(nn.Module):
    def __init__(self, model_name=None, model_path=None):
        super().__init__()
        if model_path is not None:
            # Load fine-tuned model from local checkpoint
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif model_name is not None:
            # Load pre-trained model from HuggingFace
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            raise ValueError("Either model_name or model_path must be provided.")

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

class PretrainedT5(BioLaySummT5Flan):
    def __init__(self, model_name):
        super().__init__(model_name=model_name)

class FineTunedT5(BioLaySummT5Flan):
    def __init__(self, checkpoint_path):
        super().__init__(model_path=checkpoint_path)
