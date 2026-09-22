#!/usr/bin/env python3

import argparse
import json
import math
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from laya import Agent
from laya.common import QTYPES, build_sequence, collate_items, temp_bucket
from safetensors.torch import save_file

from validate import validate_file


DEFAULT_MODEL = "convaiinnovations/laya"
DEFAULT_REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"
ACTION_LABELS = {"answer": 0, "escalate": 1}
MIN_CALIBRATION_EXAMPLES = 10


def load_records(path: Path) -> dict[str, list[dict[str, Any]]]:
    validate_file(path)
    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    with path.open(encoding="utf-8") as source:
        for raw_line in source:
            if raw_line.strip():
                record = json.loads(raw_line)
                splits[record["split"]].append(record)
    if not splits["train"]:
        raise ValueError("dataset must contain at least one training record")
    if not splits["validation"]:
        raise ValueError("dataset must contain at least one validation record")
    return splits


def target_indices(record: dict[str, Any]) -> tuple[int, int]:
    question = record["question"]
    target = record["target"]
    kind = question["type"]
    if kind == "choice":
        criteria = question["criteria"]
        labels = list(criteria) if isinstance(criteria, dict) else criteria
        option_target = labels.index(target["choice"])
    elif kind == "score":
        option_target = target["score"]
    else:
        option_target = int(target["noul"])
    return option_target, ACTION_LABELS[target.get("action", "answer")]


def prepare_item(agent: Agent, record: dict[str, Any]) -> dict[str, Any]:
    question = agent._to_internal(record["question"])
    ids, markers = build_sequence(
        agent.tok,
        record["state"],
        question,
        agent.cfg.get("max_len", 512),
        agent.cfg.get("head_max_len", 192),
    )
    option_target, action_target = target_indices(record)
    if option_target >= len(markers):
        raise ValueError(
            f"record {record['id']!r} target option was truncated; "
            "shorten its instructions or criteria"
        )
    return {
        "ids": ids,
        "markers": markers,
        "qtype": QTYPES[question["t"]],
        "label": option_target,
        "action_label": action_target,
        "record_id": record["id"],
    }


def batches(
    items: list[dict[str, Any]],
    batch_size: int,
    pad_id: int,
    *,
    shuffle: bool,
    rng: random.Random,
):
    indices = list(range(len(items)))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        selected = [items[index] for index in indices[start : start + batch_size]]
        batch = collate_items([selected], pad_id)
        if batch is None:
            continue
        batch["action_label"] = torch.tensor(
            [item["action_label"] for item in selected],
            dtype=torch.long,
        )
        yield batch


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "input_ids",
        "attention_mask",
        "marker_pos",
        "marker_mask",
        "qtype",
        "label",
        "action_label",
    )
    return {key: batch[key].to(device) for key in keys}


def compute_loss(
    logits: torch.Tensor,
    action_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    action_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    option_loss = F.cross_entropy(logits, batch["label"])
    action_loss = F.cross_entropy(action_logits, batch["action_label"])
    return (
        option_loss + action_loss_weight * action_loss,
        option_loss,
        action_loss,
    )


@torch.no_grad()
def evaluate(
    agent: Agent,
    items: list[dict[str, Any]],
    batch_size: int,
    action_loss_weight: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not items:
        return {}, []
    agent.model.eval()
    totals = Counter()
    predictions: list[dict[str, Any]] = []
    rng = random.Random(0)
    for raw_batch in batches(
        items,
        batch_size,
        agent.tok.pad_token_id,
        shuffle=False,
        rng=rng,
    ):
        batch = move_batch(raw_batch, agent.device)
        logits, action_logits = agent.model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["marker_pos"],
            batch["marker_mask"],
            batch["qtype"],
        )
        loss, option_loss, action_loss = compute_loss(
            logits,
            action_logits,
            batch,
            action_loss_weight,
        )
        count = batch["label"].numel()
        totals["count"] += count
        totals["loss"] += float(loss) * count
        totals["option_loss"] += float(option_loss) * count
        totals["action_loss"] += float(action_loss) * count
        totals["option_correct"] += int(
            (logits.argmax(dim=-1) == batch["label"]).sum()
        )
        totals["action_correct"] += int(
            (action_logits.argmax(dim=-1) == batch["action_label"]).sum()
        )
        for index in range(count):
            option_count = int(batch["marker_mask"][index].sum())
            predictions.append(
                {
                    "logits": logits[index, :option_count].float().cpu(),
                    "label": int(batch["label"][index]),
                    "qtype": int(batch["qtype"][index]),
                    "option_count": option_count,
                }
            )
    count = totals["count"]
    return {
        "loss": totals["loss"] / count,
        "option_loss": totals["option_loss"] / count,
        "action_loss": totals["action_loss"] / count,
        "option_accuracy": totals["option_correct"] / count,
        "action_accuracy": totals["action_correct"] / count,
    }, predictions


def best_temperature(predictions: list[dict[str, Any]]) -> float:
    log_min = math.log(0.25)
    log_max = math.log(4.0)
    candidates = [
        math.exp(log_min + (log_max - log_min) * index / 199)
        for index in range(200)
    ]
    return min(
        candidates,
        key=lambda temperature: sum(
            float(
                F.cross_entropy(
                    prediction["logits"][None, :] / temperature,
                    torch.tensor([prediction["label"]]),
                )
            )
            for prediction in predictions
        ),
    )


def fit_temperatures(
    predictions: list[dict[str, Any]],
) -> tuple[list[float], dict[str, float]]:
    temperatures = [1.0, 1.0, 1.0]
    by_options: dict[str, float] = {}
    for qtype in range(3):
        selected = [item for item in predictions if item["qtype"] == qtype]
        if len(selected) >= MIN_CALIBRATION_EXAMPLES:
            temperatures[qtype] = best_temperature(selected)

    buckets: dict[str, list[dict[str, Any]]] = {}
    for prediction in predictions:
        bucket = temp_bucket(prediction["qtype"], prediction["option_count"])
        buckets.setdefault(bucket, []).append(prediction)
    for bucket, selected in buckets.items():
        if len(selected) >= MIN_CALIBRATION_EXAMPLES:
            by_options[bucket] = best_temperature(selected)
    return temperatures, by_options


def resolve_checkpoint(model: str, revision: str | None) -> Path:
    local_path = Path(model).expanduser()
    if local_path.exists():
        return local_path.resolve()
    return Path(
        snapshot_download(
            model,
            revision=revision,
            allow_patterns=[
                "rl_agent_config.json",
                "model.safetensors",
                "tokenizer/*",
                "encoder/*",
            ],
        )
    )


def prepare_output(output_dir: Path, source_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose a new directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer", "encoder"):
        source = source_dir / name
        if not source.is_dir():
            raise FileNotFoundError(f"checkpoint is missing required directory: {source}")
        shutil.copytree(source, output_dir / name)


def save_checkpoint(
    agent: Agent,
    output_dir: Path,
    source_dir: Path,
    args: argparse.Namespace,
    metrics: dict[str, dict[str, float]],
    temperatures: list[float],
    temperatures_by_options: dict[str, float],
    elapsed_seconds: float,
) -> None:
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in agent.model.state_dict().items()
    }
    save_file(state, output_dir / "model.safetensors")
    config = json.loads((source_dir / "rl_agent_config.json").read_text())
    config["temperature"] = temperatures
    config["temperature_by_options"] = temperatures_by_options
    config["training"] = {
        "dataset": str(args.dataset.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "action_loss_weight": args.action_loss_weight,
        "encoder_frozen": not args.train_encoder,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "metrics": metrics,
        "base_model": args.model,
        "base_revision": args.revision,
    }
    (output_dir / "rl_agent_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning for a Laya typed-decision checkpoint"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/laya-finetuned"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--action-loss-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-encoder",
        action="store_true",
        help="Fine-tune the full encoder; requires substantially more memory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if args.action_loss_weight < 0:
        raise ValueError("action loss weight must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits = load_records(args.dataset)
    source_dir = resolve_checkpoint(args.model, args.revision)
    output_dir = args.output.resolve()
    prepare_output(output_dir, source_dir)

    print(f"Loading checkpoint from {source_dir}...")
    agent = Agent(str(source_dir))
    if not args.train_encoder:
        for parameter in agent.model.encoder.parameters():
            parameter.requires_grad = False
    named_trainable = [
        (name, parameter)
        for name, parameter in agent.model.named_parameters()
        if parameter.requires_grad
    ]
    trainable = [parameter for _, parameter in named_trainable]
    action_parameters = [
        parameter
        for name, parameter in named_trainable
        if name.startswith("act_head.")
    ]
    option_parameters = [
        parameter
        for name, parameter in named_trainable
        if not name.startswith("act_head.")
    ]
    print(
        f"Device: {agent.device}; trainable parameters: "
        f"{sum(parameter.numel() for parameter in trainable):,}"
    )

    prepared = {
        split: [prepare_item(agent, record) for record in records]
        for split, records in splits.items()
    }
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = random.Random(args.seed)
    best_validation_option_loss = math.inf
    best_trainable: dict[str, torch.Tensor] | None = None
    started_at = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        agent.model.train()
        running_loss = 0.0
        examples = 0
        for raw_batch in batches(
            prepared["train"],
            args.batch_size,
            agent.tok.pad_token_id,
            shuffle=True,
            rng=rng,
        ):
            batch = move_batch(raw_batch, agent.device)
            optimizer.zero_grad(set_to_none=True)
            logits, action_logits = agent.model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["marker_pos"],
                batch["marker_mask"],
                batch["qtype"],
            )
            loss, option_loss, action_loss = compute_loss(
                logits,
                action_logits,
                batch,
                args.action_loss_weight,
            )
            action_gradients = (
                torch.autograd.grad(
                    args.action_loss_weight * action_loss,
                    action_parameters,
                    retain_graph=True,
                )
                if args.action_loss_weight > 0
                else ()
            )
            option_loss.backward()
            for parameter, gradient in zip(action_parameters, action_gradients):
                parameter.grad = gradient
            torch.nn.utils.clip_grad_norm_(option_parameters, 1.0)
            if action_gradients:
                torch.nn.utils.clip_grad_norm_(action_parameters, 1.0)
            optimizer.step()
            count = batch["label"].numel()
            running_loss += float(loss.detach()) * count
            examples += count

        validation_metrics, _ = evaluate(
            agent,
            prepared["validation"],
            args.batch_size,
            args.action_loss_weight,
        )
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"train_combined_loss={running_loss / examples:.4f} "
            f"validation_option_loss={validation_metrics['option_loss']:.4f} "
            f"validation_action_loss={validation_metrics['action_loss']:.4f} "
            f"option_accuracy={validation_metrics['option_accuracy']:.3f} "
            f"action_accuracy={validation_metrics['action_accuracy']:.3f}"
        )
        if validation_metrics["option_loss"] < best_validation_option_loss:
            best_validation_option_loss = validation_metrics["option_loss"]
            best_trainable = {
                name: parameter.detach().cpu().clone()
                for name, parameter in agent.model.named_parameters()
                if parameter.requires_grad
            }

    if best_trainable is None:
        raise RuntimeError("training did not produce a checkpoint")
    with torch.no_grad():
        for name, parameter in agent.model.named_parameters():
            if name in best_trainable:
                parameter.copy_(best_trainable[name].to(parameter.device))

    metrics: dict[str, dict[str, float]] = {}
    validation_metrics, validation_predictions = evaluate(
        agent,
        prepared["validation"],
        args.batch_size,
        args.action_loss_weight,
    )
    metrics["validation"] = validation_metrics
    if prepared["test"]:
        metrics["test"], _ = evaluate(
            agent,
            prepared["test"],
            args.batch_size,
            args.action_loss_weight,
        )
    temperatures, temperatures_by_options = fit_temperatures(
        validation_predictions
    )
    save_checkpoint(
        agent,
        output_dir,
        source_dir,
        args,
        metrics,
        temperatures,
        temperatures_by_options,
        time.monotonic() - started_at,
    )
    if len(validation_predictions) < MIN_CALIBRATION_EXAMPLES:
        print(
            "Warning: fewer than "
            f"{MIN_CALIBRATION_EXAMPLES} validation examples; "
            "temperatures remain neutral (1.0)."
        )
    print(f"Saved fine-tuned checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
