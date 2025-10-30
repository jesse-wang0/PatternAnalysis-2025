"""
Handles data loading and preprocessing for the BioLay-Summ dataset.

Includes functions to load, tokenize, and split the dataset into train/val/test DataLoaders, and to
prepare tokenized inputs and labels for model training.

Supports a `sanity_check` mode for quick verification using a small subset of samples.
"""

from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Tuple

# --- SANITY CHECK SIZE ---
TINY_SIZE = 20

def load_bio_lay_summ_data(
    tokenizer,
    batch_size: int = 8, 
    eval_batch_size: int = 4, 
    train_split_ratio: float = 0.8, 
    max_input_length: int = 512, 
    max_output_length: int = 128,
    seed: int = 0,
    sanity_check: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads, tokenizes, and splits the BioLaySumm dataset into train/val/test sets.

    Applies preprocessing, builds PyTorch DataLoaders, and supports a `sanity_check` mode
    for quick pipeline testing with limited samples.

    Args:
        tokenizer: Hugging Face tokenizer for text encoding.
        batch_size (int, optional): Training batch size. Defaults to 8.
        eval_batch_size (int, optional): Evaluation batch size. Defaults to 4.
        train_split_ratio (float, optional): Fraction of data for training. Defaults to 0.8.
        max_input_length (int, optional): Max token length for inputs. Defaults to 512.
        max_output_length (int, optional): Max token length for outputs. Defaults to 128.
        seed (int, optional): Random seed. Defaults to 0.
        sanity_check (bool, optional): Use small subset for debugging. Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, val, and test loaders.
    """

    dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
    
    # Split train into train + validation
    train_val_split = dataset["train"].train_test_split(
        test_size=1 - train_split_ratio, seed=seed
    )

    if sanity_check:
        print(f"--- SANITY CHECK MODE: Loading only the first {TINY_SIZE} samples of Train and Val. ---")
        
        # Select only a tiny fraction of the split data
        train_data = train_val_split["train"].select(range(TINY_SIZE))
        val_data = train_val_split["test"].select(range(TINY_SIZE))
        
        # Only use a tiny fraction of the test set if we are in sanity mode
        test_data = dataset["validation"].select(range(TINY_SIZE)) 
    else:
        train_data = train_val_split["train"]
        val_data = train_val_split["test"]
        test_data = dataset["validation"]
    
    tokenized_train = train_data.map(
        lambda batch: preprocess(batch, tokenizer, max_input_length, max_output_length),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    tokenized_val = val_data.map(
        lambda batch: preprocess(batch, tokenizer, max_input_length, max_output_length),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Treat validation set as the test set (has target column)
    tokenized_test = test_data.map(
        lambda batch: preprocess(batch, tokenizer, max_input_length, max_output_length),
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

    train_loader = DataLoader(
        tokenized_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        tokenized_val,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        tokenized_test,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader

def preprocess(
    batch: dict, 
    tokenizer,
    max_input_length: int, 
    max_output_length: int
) -> dict:
    """
    Tokenizes and formats a batch of radiology/layman text pairs.

    Encodes inputs and targets, applies padding/truncation, and replaces label padding tokens
    with -100 to ignore them in loss computation.

    Args:
        batch (dict): Batch with 'radiology_report' and 'layman_report' fields.
        tokenizer: Hugging Face tokenizer.
        max_input_length (int): Max input length.
        max_output_length (int): Max output length.

    Returns:
        dict: Tokenized batch with `input_ids`, `attention_mask`, and `labels`.
    """
    
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

    # Replace padding tokens in labels with -100 (ignored in loss)
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in labels["input_ids"]
    ]
    
    batch["input_ids"] = inputs["input_ids"]
    batch["attention_mask"] = inputs["attention_mask"]
    batch["labels"] = labels["input_ids"]
    
    return batch
