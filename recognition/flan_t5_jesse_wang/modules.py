import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

class BioLaySummT5Base(nn.Module):
    """Base class with common functionality"""
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    def to(self, device):
        self.model.to(device)
        return self
    
    def train(self):
        self.model.train(True)
        return self
    
    def eval(self):
        self.model.train(False)
        return self

class PretrainedT5(BioLaySummT5Base):
    """Load a pretrained T5 model from HuggingFace"""
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.fine_tuned = False

class FineTunedT5(BioLaySummT5Base):
    """Load a fully fine-tuned T5 model (non-LoRA)"""
    def __init__(self, model_path):
        super().__init__()
        checkpoint = torch.load(model_path, map_location='cpu')
        base_model = checkpoint['base_model_name']
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        
        # Remove 'model.' prefix from state_dict keys
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {
            k[6:] if k.startswith('model.') else k: v 
            for k, v in state_dict.items()
        }
        
        self.model.load_state_dict(new_state_dict)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model_name = base_model
        self.fine_tuned = True
        self.lora = False

class FineTunedT5LoRA(BioLaySummT5Base):
    """Load a LoRA fine-tuned T5 model"""
    def __init__(self, model_path):
        super().__init__()
        
        checkpoint = torch.load(model_path, map_location='cpu')
        base_model = checkpoint['base_model_name']
        lora_params = checkpoint['lora_params']
        
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        
        peft_config = LoraConfig(
            r=lora_params["r"],
            lora_alpha=lora_params["alpha"],
            target_modules=lora_params["target_modules"],
            lora_dropout=lora_params["dropout"],
            task_type="SEQ_2_SEQ_LM"
        )
        self.model = get_peft_model(base, peft_config)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        self.model = self.model.merge_and_unload()
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model_name = base_model
        self.fine_tuned = True
        self.lora = False