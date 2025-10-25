# Layperson Summary Translation with FLAN-T5

Jesse Wang (s4807630)

## Introduction

Fine-tuned a pretrained encoder–decoder **FLAN-T5** model to translate expert radiology reports into layperson summaries. This project specifically fine tunes the small and base models of the T5 Flan, as well as applying LoRA (PEFT) to the base T5 Flan.

## Dataset

**Source:** [BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track)

**Total samples:** 160,454

- **Training split:** 120,363 samples
- **Validation split:** 10,000 samples
- **Test split:** 30,091 samples

An **80/20 split** was applied to the _original training set_ to create an internal **train/holdout validation** set.

- A 20% holdout was chosen to ensure a sufficiently large validation sample while retaining most data for training.

The **validation set from the Hugging Face dataset** was used directly as the **final test set**.

## Pre-processing

1. Loaded the dataset using the Hugging Face `datasets` library

2. **Tokenize** both the radiology report and layperson summary using a provided tokenizer

3. **Prepend an instruction prompt** to each input:
   ```text
   Provide a layman's interpretation of this medical report:
   {radiology_report}
   ```
4. **Truncate and pad** to fixed maximum lengths (`max_input_length=512`, `max_output_length=128`)

5. Replace padding tokens in labels with `-100` to ignore them during loss computation

## Models

Two pretrained models were used to establish baselines:

- **FLAN-T5 Small**
- **FLAN-T5 Base**

These served as reference points to evaluate the effect of fine-tuning.

The following models were fine-tuned on the BioLaySumm dataset:

- **FLAN-T5 Small (Tuned)** – full fine-tuning to assess smaller model performance
- **FLAN-T5 Base (Tuned)** – full fine-tuning for improved summary quality
- **FLAN-T5 Base (LoRA)** – parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA)

The LoRA variant was included to compare performance and training efficiency against full fine-tuning approaches.

## Evaluation Metrics

Primary evaluation metrics were based on the ROUGE family, widely used in text summarization tasks:

- **ROUGE-1**: Measures unigram (word-level) overlap between generated and reference summaries — evaluates overall content coverage.
- **ROUGE-2**: Measures bigram (two-word sequence) overlap — reflects fluency and phrase-level accuracy.
- **ROUGE-L**: Based on the longest common subsequence — assesses coherence and logical flow of the generated summary.
- **ROUGE-Lsum**: Sentence-level variant of ROUGE-L — captures structural alignment and readability across sentences.

All metrics were computed using the **Hugging Face `evaluate`** library, following standard summarization evaluation protocols.

## Tuning

| Attribute                | **FLAN-T5 Small (Tuned)** | **FLAN-T5 Base (Tuned)** | **FLAN-T5 Base (LoRA)**             |
| ------------------------ | ------------------------- | ------------------------ | ----------------------------------- |
| **Total Parameters**     | 77,000,000                | 248,000,000              | 248,000,000                         |
| **Trainable Parameters** | 77,000,000                | 248,000,000              | 884,736                             |
| **Trainable Fraction**   | 100%                      | 100%                     | 0.36%                               |
| **Frozen Parameters**    | 0                         | 0                        | 247,115,264                         |
| **Memory Usage**         | ~6 GB                     | ~12 GB                   | ~6 GB (with gradient checkpointing) |
| **Training Speed**       | 1.4× baseline             | 1.0× baseline            | 2.2× baseline                       |
| **Batch Size**           | 8                         | 4                        | 8                                   |
| **Learning Rate**        | 5e-5                      | 5e-5                     | 1e-4                                |
| **Epochs**               | 3                         | 2                        | 3                                   |

> **Summary:**
>
> - The **FLAN-T5 Small (Tuned)** model provided a lightweight baseline for assessing scalability.
> - The **FLAN-T5 Base (Tuned)** model achieved higher-quality summaries but required more compute.
> - The **FLAN-T5 Base (LoRA)** configuration delivered a strong trade-off between performance, speed, and efficiency — fine-tuning only **0.36%** of parameters while maintaining comparable ROUGE scores.

## Hardware & Environment

| Component              | Specification                                       |
| ---------------------- | --------------------------------------------------- |
| **GPU**                | NVIDIA A100 (40 GB VRAM)                            |
| **Runtime**            | PyTorch 2.3.1, CUDA 12.1                            |
| **Training Framework** | Hugging Face Transformers + PEFT (LoRA)             |
| **Training Duration**  | ~2.2 hours (LoRA), ~4.8 hours (Base full fine-tune) |

> All models were trained using **AdamW optimizer**, early stopping based on validation loss, and evaluated with **ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum** metrics.

## Results

Flan T5 SMALL MODEL (PRE TRAINED)

```json
"rouge_scores": {
    "rouge1": 0.2279,
    "rouge2": 0.0722,
    "rougeL": 0.203,
    "rougeLsum": 0.2031
}
```

Flan T5 BASE MODEL (PRE TRAINED)

```json
"rouge_scores": {
    "rouge1": 0.2091,
    "rouge2": 0.0594,
    "rougeL": 0.1788,
    "rougeLsum": 0.1789
}
```

Flan T5 SMALL MODEL (FINE TUNED)

- Epochs: 3 (Best model selected at `epoch 3`)
- GPU: A100 (40GB VRAM)
- Time taken: 1.2 hours

```json
"rouge_scores": {
    "rouge1": 0.7247,
    "rouge2": 0.5506,
    "rougeL": 0.6789,
    "rougeLsum": 0.6789
}
```

in ~/t5flan-small-results on rangpur

Flan T5 BASE MODEL (FINE TUNED)

- Epochs: 4 (Best model selected at `epoch 4`)
- GPU: A100 (40GB VRAM)
- Time taken: 2.2 hours

```json
"rouge_scores": {
    "rouge1": 0.7406,
    "rouge2": 0.5742,
    "rougeL": 0.6980,
    "rougeLsum": 0.6980
}
```

In 3710 folder

- Epochs: 4 (Best model selected at `epoch 4`)
- GPU: A100 (40GB VRAM)
- 1.8 hours (108 minutes) (5 minutes faster per training epoch)
- inference slightly slower

inference: 17 minutes

```json
"rouge_scores": {
    "rouge1": 0.7233,
    "rouge2": 0.5480,
    "rougeL": 0.6771,
    "rougeLsum": 0.6771
}
```

In rangpur

## References

https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track

Chung, H. W., et al. (2022). Scaling Instruction-Finetuned Language Models. arXiv preprint arXiv:2210.11416. https://arxiv.org/abs/2210.11416

https://www.kaggle.com/code/paultimothymooney/fine-tune-flan-t5-with-peft-lora-deeplearning-ai#2---Perform-Full-Fine-Tuning

https://www.datacamp.com/tutorial/flan-t5-tutorial

https://medium.com/@qmsoqm2/auto-regressive-vs-sequence-to-sequence-d7362eda001e

https://huggingface.co/spaces/evaluate-metric/rouge
