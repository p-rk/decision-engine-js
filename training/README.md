# Training data and model development

This directory contains the versioned dataset format, validator, and supervised PyTorch
fine-tuning pipeline. The trainer starts from a compatible Laya checkpoint and writes both a
reloadable PyTorch checkpoint and a playground-compatible ONNX model.

## What gets trained

Training data does not get "trained" by itself. A pretrained bidirectional encoder is fine-tuned
with:

1. The state, question instructions, and dynamic options encoded as one sequence.
2. A classification loss on the correct option marker.
3. An action loss teaching the model when to answer or escalate.
4. Optional RLCD fine-tuning using a strictly proper scoring rule.
5. Temperature calibration on held-out validation predictions.

Start from a pretrained Laya-compatible checkpoint. Training the 300M–400M parameter encoder from
scratch requires far more data and compute than workflow fine-tuning.

## Data format

Use one JSON object per line. Each record contains one state/question pair because every question
has its own dynamically defined answer space.

```json
{
  "id": "ticket-001-department",
  "state": "I was billed twice.",
  "question": {
    "type": "choice",
    "instructions": "Which department should handle this request?",
    "criteria": {
      "billing": "invoices, payments, refunds",
      "technical": "bugs and outages",
      "sales": "pricing and contracts"
    }
  },
  "target": {
    "choice": "billing",
    "action": "answer"
  },
  "split": "train",
  "metadata": {
    "language": "en",
    "source": "human-reviewed"
  }
}
```

Supported targets:

- `choice`: the exact label from `question.criteria`
- `score`: a zero-based integer index into the ordered criteria
- `noul`: a boolean
- `action`: `answer` or `escalate`

The authoritative structural definition is in `schema.json`.

Validate a dataset:

```bash
bun run validate:data
python3 training/validate.py path/to/your-data.jsonl
```

`department.jsonl` contains a balanced, realistic synthetic starter dataset for the playground:
54 examples, 18 per department, with 42 training, 6 validation, and 6 test records. Treat it as a
development fixture and replace or supplement it with reviewed examples from your own workflow.

## Collecting useful data

For a first workflow-specific model:

1. Export real historical examples only when you have permission to use them.
2. Remove secrets and personally identifiable information.
3. Write one stable rubric for each question.
4. Have trained reviewers label examples using that rubric.
5. Include hard negatives and ambiguous examples, not only obvious cases.
6. Mark cases that require human judgment with `action: "escalate"`.
7. Keep label frequencies reasonably balanced or use loss weighting.
8. Deduplicate before splitting.
9. Split by customer, conversation, or source—not random message rows—to prevent leakage.

A useful pilot commonly starts around 500–2,000 reviewed examples per workflow. Rare labels and
multilingual coverage may need substantially more. Dataset quality and representative edge cases
matter more than generating a large number of repetitive synthetic examples.

Suggested split:

- 70–80% training
- 10–15% validation
- 10–15% test

Never use the test split for model selection or temperature fitting.

## Train and export

Install the existing conversion environment, validate the dataset, and train:

```bash
uv sync --project conversion
bun run validate:data
bun run train:model
```

The default command writes a checkpoint to `models/laya-finetuned`. It freezes the approximately
400M-parameter encoder and trains the decision transformer, option scorer, type embeddings, and
answer/escalate head. This is the practical default for Apple Silicon and small workflow datasets.

Export the best validation checkpoint to ONNX:

```bash
bun run export:trained
```

Run the playground with the trained model:

```bash
MODEL_DIR=models/laya-finetuned-onnx bun run demo
```

The complete command is configurable:

```bash
uv run --project conversion training/train.py path/to/data.jsonl \
  --output models/my-checkpoint \
  --epochs 20 \
  --batch-size 4 \
  --learning-rate 5e-4

uv run --project conversion conversion/export_onnx.py \
  --checkpoint models/my-checkpoint \
  --output models/my-checkpoint-onnx
```

Use `--train-encoder` only with enough representative data and memory. It enables full-model
fine-tuning and uses substantially more memory than the default head-only training.

Training optimizes option cross-entropy and answer/escalate cross-entropy, clips their gradients
separately so saturated action logits cannot overwhelm option learning, keeps the checkpoint with
the lowest validation option loss, and records validation/test metrics in `rl_agent_config.json`.
Temperature calibration is fitted only for question types or option-count buckets having at least
10 validation examples; smaller sets use a neutral temperature of `1.0`.

The JSONL examples must match how inference questions are presented. For the playground, train
choice records with the same question and labels users will enter, and include multiple varied
examples for each label. A single hacked-account row is useful as a smoke test but is not enough
to reliably change general behavior.

## Trainer stages

The supervised pipeline implements these stages:

1. Load a compatible ModernBERT/mmBERT encoder and decision-head checkpoint.
2. Reproduce the exact marker-based sequence construction in `@decision-engine/core`.
3. Fine-tune with option cross-entropy plus action-head cross-entropy.
4. Select checkpoints using validation negative log-likelihood and task accuracy.
5. Fit temperatures by question type and option-count bucket when enough validation data exists.
6. Evaluate option and action loss/accuracy on validation and test data.
7. Export the complete encoder and both heads to ONNX.
8. Run numerical parity tests against PyTorch before publishing.

RLCD, macro F1/ECE reporting, and quantization-aware training remain future enhancements. Establish
a reproducible supervised baseline before attempting them.
