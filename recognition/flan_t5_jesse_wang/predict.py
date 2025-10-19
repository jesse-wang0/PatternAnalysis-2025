from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
from dataset import load_bio_lay_summ_data

# Load the tokenizer, model, and data collator
MODEL_NAME = "google/flan-t5-base"
OUTPUT_DIR = "results"
pre_trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenized_train, tokenized_val, tokenized_test = load_bio_lay_summ_data(MODEL_NAME)

dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
test_index = 200
report = dataset["train"][test_index]["radiology_report"]
summary = dataset["train"][test_index]["layman_report"]

# Build the same prompt format used in your preprocessing
prompt = f"Provide a layman's interpretation of this medical report:\n{report}"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

# Generate
outputs = pre_trained_model.generate(
    **inputs,
    max_new_tokens=200,
    num_beams=4
)

# Decode
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Display
dash_line = "-" * 80
print(dash_line)
print("INPUT PROMPT:\n", prompt)
print(dash_line)
print("BASELINE HUMAN SUMMARY:\n", summary)
print(dash_line)
print("MODEL GENERATION - ZERO SHOT:\n", decoded)
