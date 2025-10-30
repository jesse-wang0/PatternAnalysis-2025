"""
Evaluates pretrained, fine-tuned, and LoRA-adapted BioLay-Summ T5 models on the test set.
Visualizes ROUGE scores and sample summaries.

Results and visualizations are saved to the `results/` directory.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import random
import evaluate
from tqdm import tqdm

from dataset import load_bio_lay_summ_data
from modules import PretrainedT5, FineTunedT5, FineTunedT5LoRA

# --- Configs --- 
NUM_SAMPLES = 10
RESULT_FOLDER = Path("./results")
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

def main():
    """
    Runs evaluation for multiple BioLay-Summ T5 models.

    Loads pretrained and fine-tuned models, prepares the test dataset, computes ROUGE metrics,
    generates sample summaries, and visualizes comparative ROUGE scores.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    pre_trained_small = PretrainedT5(model_name="google/flan-t5-small").to(device)
    pre_trained_base = PretrainedT5(model_name="google/flan-t5-base").to(device)
    tuned_small = FineTunedT5(model_path="./t5flan-small-results/best_model/pytorch_model.bin").to(device)
    tuned_base = FineTunedT5(model_path="./t5flan-base-results/best_model/pytorch_model.bin").to(device)
    tuned_lora = FineTunedT5LoRA(model_path="./t5flan-base-lora-results/best_model/pytorch_model.bin").to(device)
    models = [pre_trained_small, pre_trained_base, tuned_small, tuned_base, tuned_lora]

    _, _, test_loader = load_bio_lay_summ_data(
        models[0].tokenizer,  # all t5_flan models use same tokenizer
        batch_size=8,
        eval_batch_size=16
    )

    rouge = evaluate.load("rouge")

    all_result_metrics = {}
    for model in models:
        model_metrics = test_t5_flan(device, model, rouge, test_loader)
        all_result_metrics.update(model_metrics)  # merge dict keyed by model_name

    graph_rouges(all_result_metrics)

def test_t5_flan(
    device: torch.device, 
    model,
    rouge,
    test_loader,
):
    """
    Evaluates a single T5 model on the test dataset.

    Args:
        device (torch.device): Device to run inference on.
        model: A PretrainedT5, FineTunedT5, or FineTunedT5LoRA instance.
        rouge: Hugging Face ROUGE metric instance.
        test_loader: DataLoader for the test dataset.

    Returns:
        dict: Dictionary containing ROUGE scores and sample generations for the model.
    """

    model_name = model.model_name
    if model.fine_tuned:
        if model.lora:
            model_name += "_lora"
        else:
            model_name += "_tuned"
            
    print("=" * 80)
    print(f"Evaluating Model: {model_name}")
    print("=" * 80)

    model.eval()
    all_inputs = []
    all_ground_truths = []
    all_predictions = []

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            all_inputs.extend(model.tokenizer.batch_decode(input_ids, skip_special_tokens=True))

            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = model.tokenizer.pad_token_id
            all_ground_truths.extend(
                model.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
            )

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                do_sample=False,
                num_beams=1,
                use_cache=True
            )
            all_predictions.extend(
                model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            )

    # Generate sample summaries
    random.seed(0)
    sample_indices = random.sample(range(len(all_inputs)), min(NUM_SAMPLES, len(all_inputs)))
    
    sample_texts = [all_inputs[i] for i in sample_indices]
    reference_texts = [all_ground_truths[i] for i in sample_indices]
    generated_texts = [all_predictions[i] for i in sample_indices]

    # Calculate rogue scores for the current model
    results = rouge.compute(
        predictions=all_predictions,
        references=all_ground_truths,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"]
    )

    rouge_scores = {}
    print(f"\n=== ROUGE Scores for {model_name} ===")
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
        model_name: {
            "rouge_scores": rouge_scores,
            "examples": generated_examples
        }
    }

    # Save as example results as JSON file
    output_path = RESULT_FOLDER / f"{model_name.replace('/', '_')}_generations.json"
    with open(output_path, "w", encoding="utf-8") as f:
        print(f"Saving result metrics to {output_path}")
        json.dump(result_metrics, f, indent=4, ensure_ascii=False)
    
    return result_metrics

def graph_rouges(result_metrics):
    """
    Plots and saves a bar chart comparing ROUGE scores across models.

    Args:
        result_metrics (dict): Dictionary of models with their ROUGE scores.
    """

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