from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def load_bio_lay_summ_data(model_name, batch_size=8, eval_batch_size=4, train_split_ratio=0.8):
    """
    Returns tokenized datasets ready for Trainer
    """
    dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Split train into train + validation
    train_val_split = dataset["train"].train_test_split(
        test_size=1 - train_split_ratio, seed=0
    )
    
    tokenized_train = train_val_split["train"].map(
        lambda batch: preprocess(batch, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    tokenized_val = train_val_split["test"].map(
        lambda batch: preprocess(batch, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Treat validation set as the test set (has target column)
    tokenized_test = dataset["validation"].map(
        lambda batch: preprocess(batch, tokenizer),
        batched=True,
        remove_columns=dataset["validation"].column_names
    )
    
    # # Set format for Trainer / PyTorch
    for d in [tokenized_train, tokenized_val, tokenized_test]:
        d.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    print(f"Shapes of the datasets:")
    print(f"Training: {tokenized_train.shape}")
    print(f"Validation: {tokenized_val .shape}")
    print(f"Test: {tokenized_test.shape}")

    print(tokenized_train)
    print(tokenized_val)
    print(tokenized_test)

    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_val, batch_size=eval_batch_size)
    test_loader = DataLoader(tokenized_test, batch_size=eval_batch_size)
    
    return train_loader, val_loader, test_loader

def preprocess(batch, tokenizer, max_input_length=512, max_output_length=128):
    # Prepend instruction prompt
    prompt = [f"Provide a layman's interpretation of this medical report:\n{r}" 
              for r in batch["radiology_report"]]
    
    inputs = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_input_length
    )
    
    labels = tokenizer(
        batch["layman_report"],
        padding="max_length",
        truncation=True,
        max_length=max_output_length
    )
    
    batch["input_ids"] = inputs["input_ids"]
    batch["attention_mask"] = inputs["attention_mask"]
    batch["labels"] = labels["input_ids"]
    
    return batch
