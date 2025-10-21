from pathlib import Path
import yaml
import torch
from torch import amp
import pandas as pd
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import load_bio_lay_summ_data
from modules import PretrainedT5
from training_history import plot_training_history

CONFIG_PATH = Path("configs")

def load_config(config_name):
    config_path = CONFIG_PATH / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as file:
        config = yaml.safe_load(file)
    return config

def main():
    config = load_config("t5_small.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    results_folder = Path("./results")
    results_folder.mkdir(parents=True, exist_ok=True)

    data = load_bio_lay_summ_data(
        AutoTokenizer.from_pretrained(config["model"]["name"]),
        batch_size=config["data"]["batch_size"],
        eval_batch_size=config["data"]["eval_batch_size"],
        train_split_ratio=config["data"]["train_split_ratio"],
        max_input_length=config["data"]["max_input_length"],
        max_output_length=config["data"]["max_output_length"],
        seed=config["data"]["seed"]
    )
    train_loader = data["loaders"]["train"]
    val_loader = data["loaders"]["val"]

    # No manual criterion - handled by huggingface T5ForConditionalGeneration
    model = PretrainedT5(config["model"]["name"]).to(device)
    print(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"])
    )
    
    train_t5_flan(device, config, model, optimizer, train_loader, val_loader)

def train_t5_flan(device, config, model, optimizer, train_loader, val_loader):
    OUTPUT_PATH = Path(config["saving_logging"]["output_dir"])
    NUM_EPOCHS = config["training"]["num_epochs"]
    print(f"Starting training using {NUM_EPOCHS} epochs")

    # Tracking for best model
    best_eval_loss = float('inf')
    saved_checkpoints = []
    training_history = []
    total_steps = 0

    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        
        # Wrap train_loader with tqdm
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for step, batch in enumerate(train_progress):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training with bf16
            with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["data"]["max_grad_norm"])
            optimizer.step()
            
            running_loss += loss.item()
            total_steps += 1
            
            # Update tqdm with current loss
            train_progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{running_loss/(step+1):.4f}'
            })
            
            # Log to training history
            if total_steps % config["saving_logging"]["logging_steps"] == 0 or (epoch == 0 and step == 0):
                training_history.append({
                    'epoch': epoch + (step + 1) / len(train_loader),
                    'step': total_steps,
                    'train_loss': loss.item(),
                    'eval_loss': None,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
        
        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {avg_train_loss:.4f}')

        # --- Validation Phase ---
        model.eval()
        validation_loss = 0.0
        
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")

        with torch.no_grad():
            for batch in val_progress:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                
                validation_loss += loss.item()
                
                # Update validation progress bar
                val_progress.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_val_loss = validation_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}')
        
        training_history.append({
            'epoch': epoch + 1,
            'step': total_steps,
            'train_loss': None,
            'eval_loss': avg_val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        # --- Save Checkpoint ---
        checkpoint_path = OUTPUT_PATH / f"checkpoint-epoch-{epoch+1}"
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        
        # Save model and optimizer state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'eval_loss': avg_val_loss,
        }, checkpoint_path / "pytorch_model.bin")
        
        # Also save tokenizer if available
        if hasattr(model, 'tokenizer'):
            model.tokenizer.save_pretrained(checkpoint_path)
        
        saved_checkpoints.append((checkpoint_path, avg_val_loss))

        if avg_val_loss < best_eval_loss:
            best_eval_loss = avg_val_loss
            best_checkpoint_path = OUTPUT_PATH / "best_model"
            Path(best_checkpoint_path).mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'eval_loss': avg_val_loss,
            }, best_checkpoint_path / "pytorch_model.bin")
            
            if hasattr(model, 'tokenizer'):
                model.tokenizer.save_pretrained(best_checkpoint_path)
            
            print(f"New best model saved with eval_loss: {best_eval_loss:.4f}")

    print("Training and Validation Done!")
    
    # Save training history to CSV
    df = pd.DataFrame(training_history)
    df.to_csv(OUTPUT_PATH / "training_history.csv", index=False)
    print("Training history saved to training_history.csv")
    
    # Plot training curves
    plot_training_history(OUTPUT_PATH / "training_history.csv", OUTPUT_PATH)
    
    return model

if __name__ == "__main__":
    main()