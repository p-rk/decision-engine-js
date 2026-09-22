# Decision Engine JS

Fast local structured decisions for Node.js, Bun, and the web.

This project is a cross-platform TypeScript implementation of Laya-compatible marker-based
decision inference. It targets ONNX Runtime rather than MLX, so deployment is not restricted to
Apple Silicon. Use one local model for choices, scoring, Noul conditions, routing, and batch
classification without sending application data to a hosted inference API.

## Current status

Implemented:

- Shared TypeScript question and response types
- Choice, score, and Noul option rendering
- Laya-compatible softmax, entropy confidence, and temperature calibration
- Prediction response construction
- Versioned model manifest parsing
- ONNX Runtime adapters for Node.js and browsers
- WebGPU-to-WebAssembly browser fallback
- Versioned JSONL training-data format, validator, and supervised fine-tuning pipeline
- Core unit tests
- Exact Hugging Face tokenizer loading and Laya sequence preparation
- `DecisionEngine.load()` and `predict()` for Node.js and Bun
- A Bun classification CLI

Not implemented yet:

- Production-grade model compression
- Production-grade quantization-aware training

These gaps fail explicitly; the project does not return mock predictions.

## Workspace

```text
packages/core  Shared contracts, question handling, calibration, postprocessing
packages/node  Native ONNX Runtime adapter for Windows, Linux, and macOS
packages/web   Browser ONNX Runtime adapter with WebGPU/WASM fallback
training       Dataset schema, examples, validator, and training design
models         Manifest examples; model binaries are intentionally ignored
conversion     Reproducible PyTorch-to-ONNX exporter and parity validation
```

## Develop

Requirements:

- Bun 1.3+
- Node.js 20+ only when consuming `@decision-engine/node` from Node instead of Bun
- Python 3.12 for model conversion and training tooling

```bash
bun install
bun test
bun run typecheck
bun run build
bun run validate:data
```

## Run with Bun

Download and verify the published FP32 model:

```bash
bun run setup
```

This streams the model from
[`p-rk/decision-engine-laya-onnx`](https://huggingface.co/p-rk/decision-engine-laya-onnx),
pins the published revision, and verifies the ONNX file against the SHA-256 in its manifest.
Use `bun run setup --force` to replace an existing download. Set `MODEL_DIR` to choose another
destination or `DECISION_ENGINE_MODEL_REVISION` to deliberately use another repository revision.

Alternatively, generate the model locally with `bun run convert:model`.

Classify a customer message:

```bash
bun run classify -- "I was billed twice. Please refund the duplicate."
```

Run the compiled package with Node.js:

```bash
bun run build
bun run classify:node -- "I was billed twice. Please refund the duplicate."
```

Use the library API:

```ts
import { DecisionEngine } from "@decision-engine/node";

const agent = await DecisionEngine.load("./models/laya-english-fp32");
try {
  const result = await agent.predict("I was billed twice.", {
    department: {
      type: "choice",
      instructions: "Who should handle this?",
      criteria: ["billing", "technical", "sales"],
    },
  });
  console.log(result);
} finally {
  await agent.dispose();
}
```

The model is approximately 1.6 GB and can take several seconds to load. Reuse one agent instance
for all predictions instead of loading it per request.

### Performance

The Node/Bun runtime uses optimized native ONNX Runtime execution. On a 12-core Apple Silicon
machine, the measured settings are:

- Input preparation: about 0.06 ms
- Model loading: about 1.2 seconds, once at startup
- Warm inference: about 746 ms with 8 threads
- Previous default warm inference: about 895 ms

The runtime automatically uses up to 8 intra-op threads while leaving two logical cores available
for the application. `bun run benchmark` reruns the local comparison. Bun already makes the HTTP
and application layer negligible; larger speedups require a smaller/distilled model, a shorter
fixed-context export, or a GPU execution provider.

## Run the web UI

```bash
bun run demo
```

Open <http://localhost:3000>. The playground includes Choose, Score, Noul, Route, and Batch
workflows. Results include probabilities, confidence, model-loading wait, inference time, and
total request time. The server starts loading the model immediately and reuses it for subsequent
requests.

## Publish on GitHub

The repository is prepared with:

- Apache-2.0 `LICENSE`
- Upstream attribution in `NOTICE`
- Generated model and virtual-environment exclusions
- Reproducible Bun and Python lockfiles
- GitHub Actions validation in `.github/workflows/ci.yml`
- No model binaries committed to Git

Create and publish the repository:

```bash
git init
git add .
git commit -m "Initial Decision Engine JS release"
git branch -M main
git remote add origin git@github.com:p-rk/decision-engine-js.git
git push -u origin main
```

Create the empty `decision-engine-js` repository in GitHub before adding the remote. Do not commit the
1.6 GB generated model; users can generate it with `bun run convert:model`, or a validated model
can be published separately through Hugging Face or GitHub Releases.

## Export the English model

The exporter downloads the exact PyTorch revision used to create `aac6fef/laya-mlx`, converts it
to an FP32 ONNX reference graph, and rejects the output if ONNX Runtime does not match PyTorch
within the configured tolerance. The validated FP32 graph is the input for later quantization.

```bash
uv run --project conversion conversion/export_onnx.py
```

Generated artifacts are written to `models/laya-english-fp32/` and are ignored by Git because the
model is hundreds of megabytes. The manifest, tokenizer, encoder configuration, and model hash are
generated beside the ONNX file.

Naive INT8 post-training quantization was evaluated but rejected because it changed sample
decision probabilities by more than 0.24. Smaller browser artifacts require quantization-aware
fine-tuning or distillation and are intentionally deferred.

## ONNX model contract

The runtime expects five inputs:

| Input | Type | Shape |
|---|---|---|
| `input_ids` | int64 | `[batch, sequence]` |
| `attention_mask` | bool | `[batch, sequence]` |
| `marker_pos` | int64 | `[batch, options]` |
| `marker_mask` | bool | `[batch, options]` |
| `qtype` | int64 | `[batch]` |

It expects two outputs:

| Output | Type | Shape |
|---|---|---|
| `logits` | float32 | `[batch, options]` |
| `action` | float32 | `[batch, actions]` |

See `models/example.manifest.json` for the complete artifact contract.

## Training

Read `training/README.md`, then copy `training/example.jsonl` and replace the synthetic examples
with reviewed data from your workflow. Validate the dataset, fine-tune a checkpoint, export it to
ONNX, and start the demo with `MODEL_DIR` pointing to the exported model directory.
