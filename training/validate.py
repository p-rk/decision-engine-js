#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_question(record: dict, line_number: int) -> tuple[str, list[str]]:
    prefix = f"line {line_number}"
    question = record.get("question")
    require(isinstance(question, dict), f"{prefix}: question must be an object")
    kind = question.get("type")
    require(kind in {"choice", "score", "noul"}, f"{prefix}: unsupported question type")
    instructions = question.get("instructions")
    require(
        isinstance(instructions, str) and instructions.strip(),
        f"{prefix}: instructions must not be blank",
    )

    criteria = question.get("criteria")
    if kind == "choice":
        require(
            isinstance(criteria, (list, dict)) and len(criteria) >= 2,
            f"{prefix}: choice criteria requires at least two labels",
        )
        labels = list(criteria) if isinstance(criteria, dict) else criteria
        require(
            all(isinstance(label, str) and label for label in labels),
            f"{prefix}: choice labels must be non-empty strings",
        )
        require(len(labels) == len(set(labels)), f"{prefix}: choice labels must be unique")
        return kind, labels

    if kind == "score":
        require(
            isinstance(criteria, list) and len(criteria) >= 2,
            f"{prefix}: score criteria requires at least two levels",
        )
        return kind, [str(index) for index in range(len(criteria))]

    require(
        criteria is None or isinstance(criteria, dict),
        f"{prefix}: noul criteria must be an object when provided",
    )
    return kind, ["false", "true"]


def validate_target(
    record: dict,
    kind: str,
    labels: list[str],
    line_number: int,
) -> None:
    prefix = f"line {line_number}"
    target = record.get("target")
    require(isinstance(target, dict), f"{prefix}: target must be an object")
    expected_key = kind
    require(expected_key in target, f"{prefix}: target.{expected_key} is required")
    allowed_keys = {expected_key, "action"}
    require(
        set(target).issubset(allowed_keys),
        f"{prefix}: target contains fields incompatible with {kind}",
    )

    if kind == "choice":
        require(
            target["choice"] in labels,
            f"{prefix}: target choice must be one of {labels}",
        )
    elif kind == "score":
        score = target["score"]
        require(
            isinstance(score, int) and not isinstance(score, bool),
            f"{prefix}: target score must be an integer",
        )
        require(0 <= score < len(labels), f"{prefix}: target score is outside criteria")
    else:
        require(
            isinstance(target["noul"], bool),
            f"{prefix}: target noul must be a boolean",
        )

    action = target.get("action", "answer")
    require(
        action in {"answer", "escalate"},
        f"{prefix}: action must be answer or escalate",
    )


def validate_file(path: Path) -> None:
    seen_ids: set[str] = set()
    splits: Counter[str] = Counter()
    question_types: Counter[str] = Counter()
    records = 0

    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error.msg}") from error

            require(isinstance(record, dict), f"line {line_number}: record must be an object")
            record_id = record.get("id")
            require(
                isinstance(record_id, str) and record_id,
                f"line {line_number}: id must be a non-empty string",
            )
            require(record_id not in seen_ids, f"line {line_number}: duplicate id {record_id!r}")
            seen_ids.add(record_id)

            state = record.get("state")
            require(
                isinstance(state, (str, dict, list)) and not (
                    isinstance(state, str) and not state.strip()
                ),
                f"line {line_number}: state must be non-empty text, an object, or an array",
            )
            split = record.get("split")
            require(
                split in {"train", "validation", "test"},
                f"line {line_number}: invalid split",
            )

            kind, labels = validate_question(record, line_number)
            validate_target(record, kind, labels, line_number)
            splits[split] += 1
            question_types[kind] += 1
            records += 1

    require(records > 0, "dataset contains no records")
    print(f"Validated {records} records from {path}")
    print("Splits:", ", ".join(f"{key}={value}" for key, value in sorted(splits.items())))
    print(
        "Question types:",
        ", ".join(f"{key}={value}" for key, value in sorted(question_types.items())),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Decision Engine JS JSONL training data")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    validate_file(args.dataset)


if __name__ == "__main__":
    main()
