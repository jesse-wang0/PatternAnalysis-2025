import torch
from dataset import load_bio_lay_summ_data
from modules import BioLaySummT5Flan
import random
import evaluate

from dataset import load_bio_lay_summ_data
from modules import BioLaySummT5Flan

# MODEL_NAME = "google/flan-t5-base"
MODEL_NAME = "google/flan-t5-small"
NUM_SAMPLES = 10

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    data = load_bio_lay_summ_data(
        MODEL_NAME,
        batch_size=8,
        eval_batch_size=4
    )
    test_loader = data["loaders"]["test"]

    model = BioLaySummT5Flan()
    model = model.to(device)

    test_t5_flan(device, model, test_loader)

def test_t5_flan(device, model, test_loader, num_samples=10):
    model.eval()
    test_loss = 0.0
    all_input_texts = []
    all_labels_texts = []

    # --- Compute average loss ---
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            test_loss += outputs.loss.item()

            # For generating samples later
            all_input_texts.extend(model.tokenizer.batch_decode(input_ids, skip_special_tokens=True))
            all_labels_texts.extend(model.tokenizer.batch_decode(labels, skip_special_tokens=True))

    avg_test_loss = test_loss / len(test_loader)
    print(f"\nAverage Test Loss: {avg_test_loss:.4f}\n")

    # --- Generate sample summaries ---
    sample_indices = random.sample(range(len(all_input_texts)), min(num_samples, len(all_input_texts)))
    sample_texts = [all_input_texts[i] for i in sample_indices]
    reference_texts = [all_labels_texts[i] for i in sample_indices]

    inputs = model.tokenizer(
        sample_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    generated_ids = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=128)
    generated_texts = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print("=== Sample Summaries ===")
    for i, (inp, ref, pred) in enumerate(zip(sample_texts, reference_texts, generated_texts), 1):
        print(f"\nSample {i}:")
        print(f"Original Report: {inp}")
        print(f"Reference Summary: {ref}")
        print(f"Predicted Summary: {pred}")
    
    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=generated_texts,
        references=reference_texts,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"]
    )

    print("\n=== ROUGE Scores ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
        
if __name__ == "__main__":
    main()