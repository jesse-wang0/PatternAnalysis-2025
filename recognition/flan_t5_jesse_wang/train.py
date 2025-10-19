import torch
from torch import amp
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

from dataset import load_bio_lay_summ_data
from modules import BioLaySummT5Flan

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

# Training
L_RATE = 3e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 2

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

    data = load_bio_lay_summ_data(
        MODEL_NAME,
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

    model = BioLaySummT5Flan().to(device)

    # No manual criterion - handled by huggingface T5ForConditionalGeneration

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=L_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # train_t5_flan(device, model, optimiser, train_loader, val_loader)
    trainer_obj = train_with_trainer(model, train_dataset, val_dataset)
    # Save final model
    trainer_obj.save_model("./final_model")

def train_with_trainer(model, train_dataset, val_dataset):
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=L_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        generation_max_length=GENERATION_MAX_LENGTH,
        generation_num_beams=GENERATION_NUM_BEAMS,
        fp16=True,
        report_to="none"
    )
    
    trainer = Seq2SeqTrainer(
        model=model.model,  # use inner model
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=model.tokenizer,
    )
    
    trainer.train()
    return trainer

def train_t5_flan(device, model, optimiser, train_loader, val_loader):
    print(f"Starting training using {NUM_EPOCHS} epochs")

    scaler = amp.GradScaler(device_type=device)

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

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()

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