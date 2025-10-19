from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import random
import evaluate

from dataset import load_bio_lay_summ_data
from modules import BioLaySummT5Flan, PretrainedT5, FineTunedT5

NUM_SAMPLES = 10
RESULT_FOLDER = Path("./results")
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    pre_trained_small = PretrainedT5(model_name="google/flan-t5-small")
    pre_trained_base = PretrainedT5(model_name="google/flan-t5-base")
    tuned_small = FineTunedT5(model_name="finetuned_flan-t5-small", model_path="./results/final_model")
    models = [pre_trained_small, pre_trained_base, tuned_small]

    data = load_bio_lay_summ_data(
        models[0].tokenizer,  # all t5_flan models use same tokenizer
        batch_size=8,
        eval_batch_size=4
    )
    test_loader = data["loaders"]["test"]

    all_result_metrics = {}
    for model in models:
        model_metrics = test_t5_flan(device, model, test_loader)
        all_result_metrics.update(model_metrics)  # merge dict keyed by model_name

    graph_rouges(all_result_metrics)

def test_t5_flan(
    device: torch.device, 
    model,
    test_loader,
):
    print("=" * 80)
    print(f"Evaluating Model: {model.model_name}")
    print("=" * 80)

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
            
            # Replace -100 with pad_token_id before decoding
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = model.tokenizer.pad_token_id
            all_ground_truths.extend(
                model.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
            )

            # Generate predictions
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128
            )
            all_predictions.extend(
                model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            )

    avg_test_loss = test_loss / len(test_loader)
    print(f"\nAverage Test Loss: {avg_test_loss:.4f}\n")

    # --- Generate sample summaries ---
    random.seed(0)
    sample_indices = random.sample(range(len(all_inputs)), min(NUM_SAMPLES, len(all_inputs)))
    
    sample_texts = [all_inputs[i] for i in sample_indices]
    reference_texts = [all_ground_truths[i] for i in sample_indices]
    generated_texts = [all_predictions[i] for i in sample_indices]

    # Calculate rogue scores for the current model
    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=all_predictions,
        references=all_ground_truths,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"]
    )

    rouge_scores = {}
    print(f"\n=== ROUGE Scores for {model.model_name} ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
        rouge_scores[metric] = round(value, 4)

    generated_examples = []
    print("=== Sample Summaries ===")
    for i, (text, ground_truth, prediction) in enumerate(zip(sample_texts, reference_texts, generated_texts), 1):
        generated_examples.append({
            "Sample": i,
            "Original Report": text,
            "Reference Summary": ground_truth,
            "Predicted Summary": prediction
        })

    result_metrics = {
        model.model_name: {
            "average_test_loss": round(avg_test_loss, 4),
            "rouge_scores": rouge_scores,
            "examples": generated_examples
        }
    }

    # Save as example results as JSON file
    output_path = RESULT_FOLDER / f"{model.model_name.replace('/', '_')}_generations.json"
    with open(output_path, "w", encoding="utf-8") as f:
        print(f"Saving result metrics to {output_path}")
        json.dump(result_metrics, f, indent=4, ensure_ascii=False)
    
    return result_metrics

def graph_rouges(result_metrics):
    # Extract model names and their ROUGE scores
    model_names = list(result_metrics.keys())
    rouge_types = list(next(iter(result_metrics.values()))["rouge_scores"].keys())

    # Prepare data matrix: each row = model, each column = ROUGE type
    scores = np.array([
        [result_metrics[m]["rouge_scores"][r] for r in rouge_types]
        for m in model_names
    ])

    # Set up bar positions
    x = np.arange(len(rouge_types))  # ROUGE types on x-axis
    width = 0.8 / len(model_names)   # width of each bar

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each model’s scores side-by-side
    for i, model_name in enumerate(model_names):
        ax.bar(x + i * width, scores[i], width, label=model_name)

    # Styling
    ax.set_xlabel("ROUGE Metric")
    ax.set_ylabel("Score")
    ax.set_title("ROUGE Score Comparison Across Models")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(rouge_types)
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    output_path = RESULT_FOLDER / "rouge_comparison.png"
    plt.savefig(output_path)

if __name__ == "__main__":
    main()