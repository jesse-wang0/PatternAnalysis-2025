import torch
from dataset import load_bio_lay_summ_data
import random
import evaluate

from dataset import load_bio_lay_summ_data
from modules import BioLaySummT5Flan, PretrainedT5, FineTunedT5

NUM_SAMPLES = 10

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    pre_trained_small = PretrainedT5(model_name="google/flan-t5-small")
    pre_trained_base = PretrainedT5(model_name="google/flan-t5-base")
    models = [pre_trained_small, pre_trained_base]

    data = load_bio_lay_summ_data(
        models[0].tokenizer,  # all use same tokenizer
        batch_size=8,
        eval_batch_size=4
    )
    test_loader = data["loaders"]["test"]

    test_t5_flan(device, models, test_loader)

def test_t5_flan(
    device: torch.device, 
    models,
    test_loader,
):
    for model in models:
        model = model.to(device)

        model.eval()
        test_loss = 0.0
        all_inputs = []
        all_ground_truths = []
        all_predictions = []

        # --- Compute average loss ---
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                test_loss += outputs.loss.item()

                # For generating samples later
                all_inputs.extend(model.tokenizer.batch_decode(input_ids, skip_special_tokens=True))
                all_ground_truths.extend(model.tokenizer.batch_decode(labels, skip_special_tokens=True))

                 # Generate predictions
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128
                )
                all_predictions.extend(model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

        avg_test_loss = test_loss / len(test_loader)
        print(f"\nAverage Test Loss: {avg_test_loss:.4f}\n")

        # --- Generate sample summaries ---
        random.seed(0)
        sample_indices = random.sample(range(len(all_inputs)), min(NUM_SAMPLES, len(all_inputs)))
        
        sample_texts = [all_inputs[i] for i in sample_indices]
        reference_texts = [all_ground_truths[i] for i in sample_indices]
        generated_texts = [all_predictions[i] for i in sample_indices]

        print("=== Sample Summaries ===")
        for i, (inp, ref, pred) in enumerate(zip(sample_texts, reference_texts, generated_texts), 1):
            print(f"\nSample {i}:")
            print(f"Original Report: {inp}")
            print(f"Reference Summary: {ref}")
            print(f"Predicted Summary: {pred}")
        
        rouge = evaluate.load("rouge")
        results = rouge.compute(
            predictions=all_predictions,
            references=all_ground_truths,
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"]
        )

        print(f"\n=== ROUGE Scores for {model.model_name} ===")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()