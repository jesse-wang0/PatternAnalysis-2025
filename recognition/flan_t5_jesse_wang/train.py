import torch
from torch import amp

from dataset import load_bio_lay_summ_data
from modules import BioLaySummT5Flan

MODEL_NAME = "google/flan-t5-small"

# --------------------------
# HYPERPARAMETERS
# --------------------------
L_RATE = 3e-4
BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH = 4
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 3
NUM_EPOCHS = 2

def main():
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(device)

   train_loader, val_loader, _ = load_bio_lay_summ_data(
      MODEL_NAME,
      batch_size=BATCH_SIZE,
      eval_batch_size=PER_DEVICE_EVAL_BATCH
   )

   model = BioLaySummT5Flan().to(device)

   # No manual criterion - handled by huggingface T5ForConditionalGeneration

   optimiser = torch.optim.AdamW(
      model.parameters(),
      lr=L_RATE,
      weight_decay=WEIGHT_DECAY
   )

   train_t5_flan(device, model, optimiser, train_loader, val_loader )

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
               outputs = model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              labels=labels)
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            validation_loss += loss.item()

      avg_val_loss = validation_loss / len(val_loader)
      print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}')

   print("Training and Validation Done!")

if __name__ == "__main__":
   main()