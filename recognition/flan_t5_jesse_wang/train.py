import torch
import pandas as pd
from torch import amp
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer

from dataset import load_bio_lay_summ_data
from modules import PretrainedT5

# --------------------------
# HYPERPARAMETERS
# --------------------------
MODEL_NAME = "google/flan-t5-small"

# Data
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
TRAIN_SPLIT_RATIO = 0.8
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128
SEED = 0
MAX_GRAD_NORM = 1.0

# Training
L_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3

# Saving & Logging
OUTPUT_DIR = "./t5flan-checkpoints"
LOGGING_STEPS = 50
SAVE_TOTAL_LIMIT = 3

# Generation (for evaluation)
GENERATION_MAX_LENGTH = MAX_OUTPUT_LENGTH  # Must match preprocessing
GENERATION_NUM_BEAMS = 4

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    results_folder = Path("./results")
    results_folder.mkdir(parents=True, exist_ok=True)

    data = load_bio_lay_summ_data(
        AutoTokenizer.from_pretrained(MODEL_NAME),
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        train_split_ratio=TRAIN_SPLIT_RATIO,
        max_input_length=MAX_INPUT_LENGTH,
        max_output_length=MAX_OUTPUT_LENGTH,
        seed=SEED
    )
    train_loader = data["loaders"]["train"]
    val_loader = data["loaders"]["val"]

    train_dataset = data["datasets"]["train"]
    val_dataset = data["datasets"]["val"]

    model = PretrainedT5(MODEL_NAME).to(device)

    # No manual criterion - handled by huggingface T5ForConditionalGeneration

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=L_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # train_t5_flan(device, model, optimiser, train_loader, val_loader)
    trainer_obj = train_with_trainer(model, train_dataset, val_dataset)

    logs = []
    for log in trainer_obj.state.log_history:
        logs.append({
            'epoch': log.get('epoch'),
            'step': log.get('step'),
            'train_loss': log.get('loss'),
            'eval_loss': log.get('eval_loss'),
            'learning_rate': log.get('learning_rate')
        })

    df = pd.DataFrame(logs)
    df.to_csv(results_folder / "training_history.csv", index=False)
    print("Training history saved to training_history.csv")

    # Extract data with proper step alignment
    train_data = df[df['train_loss'].notna()].copy()
    eval_data = df[df['eval_loss'].notna()].copy()

    # Plot with correct x-axis (steps)
    plt.figure(figsize=(12, 6))
    plt.plot(train_data['step'], train_data['train_loss'], 
            label='Train Loss', alpha=0.7, linewidth=1)
    plt.plot(eval_data['step'], eval_data['eval_loss'], 
            label='Eval Loss', marker='o', markersize=10, 
            linewidth=2, color='orange')

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Optional: Add epoch boundaries
    for idx, step in enumerate(eval_data['step'].values):
        plt.axvline(x=step, color='gray', linestyle='--', 
                alpha=0.3, linewidth=1)
        plt.text(step, plt.ylim()[1]*0.95, f'Epoch {idx+1}', 
                rotation=0, ha='center', fontsize=9, alpha=0.7)

    plt.tight_layout()
    plt.savefig(results_folder / 'loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Training points logged: {len(train_data)}")
    print(f"Evaluation points: {len(eval_data)}")

    # Save final model
    trainer_obj.save_model(results_folder / "final_model")

def train_with_trainer(model, train_dataset, val_dataset):
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=L_RATE,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        
        # Logging
        logging_dir="./logs",
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        
        # Evaluation & Saving
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Generation (for seq2seq)
        predict_with_generate=True,
        generation_max_length=GENERATION_MAX_LENGTH,
        generation_num_beams=GENERATION_NUM_BEAMS,
        
        # Performance
        fp16 = False,
        bf16 = True,
        report_to="tensorboard"  # ← Enable TensorBoard
    )

    trainer = Seq2SeqTrainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=model.tokenizer,
    )

    trainer.train()
    return trainer

def train_t5_flan(device, model, optimiser, train_loader, val_loader):
    print(f"Starting training using {NUM_EPOCHS} epochs")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimiser.zero_grad()

            with amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            running_loss += loss.item()
         
        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {avg_train_loss:.4f}')

        # --- Validation Phase ---
        model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass (loss computed automatically)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                validation_loss += loss.item()

        avg_val_loss = validation_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}')

    print("Training and Validation Done!")

if __name__ == "__main__":
    main()