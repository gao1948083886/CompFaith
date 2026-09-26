"""Run the reproducible CompFaith performance and speed evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Code.compression import DualRawEvaluator
from Code.datasets import load_jsonl

EXPECTED_COUNTS = {"DIVERSUMM": 293, "LONGEVAL": 80, "RAMPRASAD'24": 200}


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[start][1]:
            end += 1
        average = (start + end + 2) / 2.0
        for position in range(start, end + 1):
            ranks[indexed[position][0]] = average
        start = end + 1
    return ranks


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def roc_auc(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return 0.0
    ordered = sorted(zip(scores, labels, strict=True), key=lambda pair: pair[0])
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[start][0]:
            end += 1
        average_rank = (start + end + 2) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[start : end + 1])
        start = end + 1
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def percentile_nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_formal_records(path: Path) -> list[dict]:
    records = load_jsonl(path)
    counts: dict[str, int] = {}
    for record in records:
        dataset = str(record["dataset"])
        counts[dataset] = counts.get(dataset, 0) + 1
        for key in ("faithfulness", "faithful_label", "hallucination_label"):
            if key not in record:
                raise ValueError(f"Missing {key!r} in {record['id']}")
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Expected {EXPECTED_COUNTS}, found {counts}")
    return records


def score_records(records: list[dict], evaluator: DualRawEvaluator) -> tuple[dict[str, dict], list[dict]]:
    scores: dict[str, dict] = {}
    rows: list[dict] = []
    for record in records:
        result = evaluator.evaluate(record["context"], record["candidate"])
        score = float(result["score"])
        scores[record["id"]] = {"score": score, "result": result}
        rows.append(
            {
                "id": record["id"],
                "dataset": record["dataset"],
                "H": repr(score),
                "faithfulness_score": repr(1.0 - score),
                "faithful_label": record["faithful_label"],
                "hallucination_label": record["hallucination_label"],
                "lz4_score": repr(float(result["component_scores"][0])),
                "zstd_score": repr(float(result["component_scores"][1])),
            }
        )
    return scores, rows


def performance_summary(records: list[dict], scores: dict[str, dict]) -> list[dict]:
    output: list[dict] = []
    for dataset in EXPECTED_COUNTS:
        selected = [record for record in records if record["dataset"] == dataset]
        risks = [scores[record["id"]]["score"] for record in selected]
        faithfulness_scores = [1.0 - value for value in risks]
        references = [float(record["faithfulness"]) for record in selected]
        labels = [int(record["hallucination_label"]) for record in selected]
        output.append(
            {
                "dataset": dataset,
                "N": len(selected),
                "ROC_AUC": roc_auc(risks, labels),
                "Pearson": correlation(faithfulness_scores, references),
                "Spearman": correlation(rank(faithfulness_scores), rank(references)),
            }
        )
    return output


def speed_summary(
    records: list[dict],
    evaluator: DualRawEvaluator,
    repeats: int,
    warmup: bool,
) -> list[dict]:
    if warmup:
        for record in records:
            evaluator.evaluate(record["context"], record["candidate"])

    rows: list[dict] = []
    for repeat in range(1, repeats + 1):
        for dataset in EXPECTED_COUNTS:
            selected = [record for record in records if record["dataset"] == dataset]
            latencies: list[float] = []
            started = time.perf_counter()
            for record in selected:
                sample_started = time.perf_counter()
                evaluator.evaluate(record["context"], record["candidate"])
                latencies.append((time.perf_counter() - sample_started) * 1000.0)
            wall_seconds = time.perf_counter() - started
            rows.append(
                {
                    "dataset": dataset,
                    "repeat": repeat,
                    "N": len(selected),
                    "mean_latency_ms": fmean(latencies),
                    "p95_latency_ms": percentile_nearest_rank(latencies, 0.95),
                    "throughput_samples_per_s": len(selected) / wall_seconds,
                    "wall_seconds": wall_seconds,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CompFaith on the bundled formal dataset")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "Dataset" / "processed" / "formal_records.jsonl",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    records = load_formal_records(args.data)
    evaluator = DualRawEvaluator()
    started = datetime.now(timezone.utc)
    run_id = started.strftime("compfaith_%Y%m%d_%H%M%S")
    output = args.output or ROOT / "Result" / run_id
    output.mkdir(parents=True, exist_ok=False)

    scores, per_sample = score_records(records, evaluator)
    performance = performance_summary(records, scores)
    speed = speed_summary(records, evaluator, args.repeats, not args.no_warmup)
    speed_by_dataset: list[dict] = []
    for dataset in EXPECTED_COUNTS:
        selected = [row for row in speed if row["dataset"] == dataset]
        speed_by_dataset.append(
            {
                "dataset": dataset,
                "N": EXPECTED_COUNTS[dataset],
                "mean_latency_ms": fmean(row["mean_latency_ms"] for row in selected),
                "p95_latency_ms": fmean(row["p95_latency_ms"] for row in selected),
                "throughput_samples_per_s": fmean(
                    row["throughput_samples_per_s"] for row in selected
                ),
            }
        )

    write_csv(output / "per_sample_scores.csv", per_sample)
    write_csv(output / "performance.csv", performance)
    write_csv(output / "speed_by_repeat.csv", speed)
    write_csv(output / "speed_summary.csv", speed_by_dataset)
    (output / "run_metadata.json").write_text(
        json.dumps(
            {
                "method": evaluator.config,
                "dataset": str(args.data),
                "sample_counts": EXPECTED_COUNTS,
                "repeats": args.repeats,
                "warmup": not args.no_warmup,
                "started_utc": started.isoformat(),
                "python": sys.version,
                "platform": platform.platform(),
                "output": str(output),
                "metrics": {
                    "ROC_AUC": "H versus hallucination_label",
                    "Pearson": "1-H versus faithfulness",
                    "Spearman": "rank(1-H) versus rank(faithfulness)",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Results written to {output}")
    for row in performance:
        print(json.dumps(row, ensure_ascii=False))
    for row in speed_by_dataset:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
