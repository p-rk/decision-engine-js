#!/usr/bin/env python3

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from laya import Agent
from laya.common import QTYPES, build_sequence, collate_items

MODEL_ID = "convaiinnovations/laya"
SOURCE_REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"
SAMPLE_STATE = "I was billed twice. Please refund the duplicate."
SAMPLE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Who should handle this?",
        "criteria": ["billing", "technical", "sales"],
    },
    "refund": {
        "type": "noul",
        "instructions": "Does the customer explicitly ask for money back?",
    },
}


class ExportModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, action = self.model(
            input_ids,
            attention_mask,
            marker_pos,
            marker_mask,
            qtype,
        )
        return logits.float(), action.float()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_sample(agent: Agent) -> dict[str, torch.Tensor]:
    items = []
    for definition in SAMPLE_QUESTIONS.values():
        question = agent._to_internal(definition)
        token_ids, markers = build_sequence(
            agent.tok,
            SAMPLE_STATE,
            question,
            agent.cfg.get("max_len", 512),
            agent.cfg.get("head_max_len", 192),
        )
        items.append(
            {
                "ids": token_ids,
                "markers": markers,
                "qtype": QTYPES[question["t"]],
            }
        )
    batch = collate_items([items], agent.tok.pad_token_id)
    if batch is None:
        raise RuntimeError("Failed to prepare the ONNX validation sample")
    max_length = agent.cfg.get("max_len", 512)
    padding = max_length - batch["input_ids"].shape[1]
    if padding < 0:
        raise RuntimeError("Validation batch exceeds the configured model context")
    return {
        "input_ids": F.pad(
            batch["input_ids"],
            (0, padding),
            value=agent.tok.pad_token_id,
        ),
        "attention_mask": F.pad(
            batch["attention_mask"],
            (0, padding),
            value=0,
        ).bool(),
        "marker_pos": batch["marker_pos"],
        "marker_mask": batch["marker_mask"],
        "qtype": batch["qtype"],
    }


def export_model(
    model: ExportModel,
    sample: dict[str, torch.Tensor],
    output_path: Path,
) -> None:
    input_names = list(sample)
    torch.onnx.export(
        model,
        tuple(sample[name] for name in input_names),
        output_path,
        input_names=input_names,
        output_names=["logits", "action"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "marker_pos": {0: "batch", 1: "options"},
            "marker_mask": {0: "batch", 1: "options"},
            "qtype": {0: "batch"},
            "logits": {0: "batch", 1: "options"},
            "action": {0: "batch"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )


def validate_model(
    model: ExportModel,
    sample: dict[str, torch.Tensor],
    output_path: Path,
    tolerance: float,
) -> dict[str, float]:
    def softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values, axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=-1, keepdims=True)

    onnx.checker.check_model(output_path)
    with torch.inference_mode():
        expected_logits, expected_action = model(**sample)

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    ort_inputs = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in sample.items()
    }
    actual_logits, actual_action = session.run(["logits", "action"], ort_inputs)
    expected = [
        expected_logits.detach().cpu().numpy(),
        expected_action.detach().cpu().numpy(),
    ]
    expected_probabilities = softmax(expected[0])
    actual_probabilities = softmax(actual_logits)
    expected_action_probabilities = softmax(expected[1])
    actual_action_probabilities = softmax(actual_action)
    differences = {
        "logits_max_abs": float(np.max(np.abs(expected[0] - actual_logits))),
        "action_max_abs": float(np.max(np.abs(expected[1] - actual_action))),
        "probabilities_max_abs": float(
            np.max(np.abs(expected_probabilities - actual_probabilities))
        ),
        "action_probabilities_max_abs": float(
            np.max(
                np.abs(
                    expected_action_probabilities - actual_action_probabilities
                )
            )
        ),
    }
    option_argmax_matches = np.array_equal(
        np.argmax(expected_probabilities, axis=-1),
        np.argmax(actual_probabilities, axis=-1),
    )
    action_argmax_matches = np.array_equal(
        np.argmax(expected_action_probabilities, axis=-1),
        np.argmax(actual_action_probabilities, axis=-1),
    )
    probability_error = max(
        differences["probabilities_max_abs"],
        differences["action_probabilities_max_abs"],
    )
    if (
        probability_error > tolerance
        or not option_argmax_matches
        or not action_argmax_matches
    ):
        raise RuntimeError(
            "ONNX parity failed: "
            f"tolerance={tolerance}, differences={differences}, "
            f"option_argmax_matches={option_argmax_matches}, "
            f"action_argmax_matches={action_argmax_matches}"
        )
    return differences


def write_artifacts(
    source_dir: Path,
    output_dir: Path,
    model_path: Path,
    parity: dict[str, float],
    precision: str,
    tolerance: float,
    model_id: str,
    revision: str | None,
) -> None:
    shutil.copytree(source_dir / "tokenizer", output_dir / "tokenizer")
    shutil.copytree(source_dir / "encoder", output_dir / "encoder")
    shutil.copy2(source_dir / "rl_agent_config.json", output_dir)
    config = json.loads((source_dir / "rl_agent_config.json").read_text())
    manifest = {
        "schemaVersion": 1,
        "model": {
            "id": model_id,
            "responseName": "laya-rl-agent",
            "format": "onnx",
            "precision": precision,
        },
        "files": {
            "model": model_path.name,
            "tokenizer": "tokenizer/tokenizer.json",
            "config": "rl_agent_config.json",
            "sha256": {model_path.name: sha256(model_path)},
        },
        "inference": {
            "maxLength": config.get("max_len", 512),
            "headMaxLength": config.get("head_max_len", 192),
            "inputs": {
                "inputIds": "input_ids",
                "attentionMask": "attention_mask",
                "markerPositions": "marker_pos",
                "markerMask": "marker_mask",
                "questionTypes": "qtype",
            },
            "outputs": {
                "logits": "logits",
                "actionLogits": "action",
            },
        },
        "calibration": {
            "temperatures": config.get("temperature", [1, 1, 1]),
            "temperaturesByOptions": config.get("temperature_by_options", {}),
        },
        "validation": {
            "sampleQuestions": len(SAMPLE_QUESTIONS),
            "tolerance": tolerance,
            **parity,
        },
    }
    if revision is not None:
        manifest["model"]["revision"] = revision
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the English Laya decision model to portable ONNX"
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument(
        "--checkpoint",
        default=MODEL_ID,
        help="Local fine-tuned checkpoint directory or Hugging Face model ID.",
    )
    parser.add_argument(
        "--revision",
        default=SOURCE_REVISION,
        help="Hugging Face revision. Ignored for local checkpoints.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="FP32 is the portable reference. FP16 export is experimental.",
    )
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument(
        "--reuse-export",
        action="store_true",
        help="Validate and package an existing ONNX file without exporting it again.",
    )
    args = parser.parse_args()
    output_dir = (
        args.output or Path(f"models/laya-english-{args.precision}")
    ).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.reuse_export:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new directory or remove the generated artifacts explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint).expanduser()
    if checkpoint_path.exists():
        source_dir = checkpoint_path.resolve()
        model_id = source_dir.name
        revision = None
        print(f"Using local checkpoint {source_dir}...")
    else:
        model_id = args.checkpoint
        revision = args.revision
        print(f"Downloading {model_id}@{revision}...")
        source_dir = Path(
            snapshot_download(
                model_id,
                revision=revision,
                allow_patterns=[
                    "rl_agent_config.json",
                    "model.safetensors",
                    "tokenizer/*",
                    "encoder/*",
                ],
            )
        )
    print("Loading the PyTorch reference model on CPU...")
    agent = Agent(str(source_dir), device="cpu")
    sample = prepare_sample(agent)
    if args.precision == "fp16":
        agent.model.half()
        # Upstream explicitly promotes pooled/action features and normally relies on autocast.
        agent.model.act_head.float()
    model = ExportModel(agent.model.eval())
    model_path = output_dir / f"model.{args.precision}.onnx"

    if args.reuse_export:
        if not model_path.is_file():
            raise FileNotFoundError(f"Existing ONNX model not found: {model_path}")
        print(f"Reusing existing export {model_path}...")
    else:
        print(f"Exporting {model_path}...")
        export_model(model, sample, model_path)
    print("Checking ONNX structure and numerical parity...")
    parity = validate_model(model, sample, model_path, args.tolerance)
    write_artifacts(
        source_dir,
        output_dir,
        model_path,
        parity,
        args.precision,
        args.tolerance,
        model_id,
        revision,
    )
    size_mib = model_path.stat().st_size / (1024 * 1024)
    print(f"Export complete: {model_path} ({size_mib:.1f} MiB)")
    print(f"Parity: {parity}")


if __name__ == "__main__":
    main()
