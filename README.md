<div align="center">

# CompFaith

### Training-Free Source-Conditioned Compression Scoring
### for Faithfulness Hallucination Detection

<p>
  <img src="https://img.shields.io/badge/ICASSP%202027-Under%20Review-f59e0b?style=for-the-badge" alt="ICASSP 2027 under review">
  <img src="https://img.shields.io/badge/Training--Free-Yes-7c3aed?style=for-the-badge" alt="Training-Free">
  <img src="https://img.shields.io/badge/GPU-Not%20Required-22c55e?style=for-the-badge" alt="No GPU required">
</p>

**This repository accompanies our manuscript submitted to IEEE ICASSP 2027,
which is currently under review.**

</div>

---

CompFaith is a training-free method for scoring faithfulness hallucination risk
from source-conditioned lossless compression. It uses LZ4 and Zstandard and
does not require model training, neural inference, an API key, or a GPU.

The repository contains the complete three-dataset input package used for the
CompFaith experiments and the code needed to reproduce the method's
performance and speed measurements. Baseline model checkpoints, predictions,
and baseline results are intentionally not included.

## Method

For a source document `x`, response `y`, and compressor `k`, CompFaith computes

```text
Delta_k(x, y) = C_k(x || y) - C_k(x)
H_k(x, y) = min(C_k(y), max(0, Delta_k(x, y))) / C_k(y)
H(x, y) = min(H_LZ4(x, y), H_Zstd(x, y))
```

Higher `H(x, y)` means higher faithfulness hallucination risk.

## Contents

- `Code/compression/`: the frozen Dual Raw implementation;
- `Dataset/processed/formal_records.jsonl`: the complete 573-record evaluation
  input, with 293 DIVERSUMM, 80 LONGEVAL, and 200 RAMPRASAD'24 records;
- `Dataset/docs/formal_schema.md`: input fields and label conversion rules;
- `scripts/run_experiment.py`: the default formal evaluation entry point;
- `scripts/run_formal_experiment.py`: the implementation behind the entry
  point.

## Run

From the repository root, install the two required compression libraries:

```bash
python -m pip install -r requirements.txt
```

Run the performance and speed evaluation:

```bash
python scripts/run_experiment.py
```

The default command uses the complete 573-record dataset, performs one warmup
pass, computes ROC-AUC, Pearson, and Spearman for each dataset, and measures
per-sample mean latency, P95 latency, and throughput over five repeats. Results
are written to a timestamped directory under `Result/`:

- `performance.csv`
- `per_sample_scores.csv`
- `speed_by_repeat.csv`
- `speed_summary.csv`
- `run_metadata.json`

To use a different output directory or repeat count:

```bash
python scripts/run_experiment.py \
    --output Result/my_run \
    --repeats 5
```

Speed results depend on the user's hardware, operating system, Python version,
and system load. The performance evaluation uses the bundled records and the
frozen Dual Raw configuration: LZ4 frame level 16, Zstandard level 19, UTF-8
encoding, case preserved, whitespace stripped, newline separator, and minimum
aggregation.

## Data and labels

The bundled input contains only the fields required by CompFaith. The three
datasets use their supplied annotations in a common format:

- DIVERSUMM: native summary-level binary labels and the `NoE` sentence-label
  proportion as the continuous faithfulness reference;
- LONGEVAL: `fine_rating / 100`, with exactly `1.0` treated as faithful;
- RAMPRASAD'24: native binary and continuous faithfulness labels.

See `Dataset/docs/formal_schema.md` for the complete schema.

## Scope

This release is intentionally limited to the CompFaith method. It does not
ship baseline model weights, baseline prediction files, baseline speed
measurements, or secret credentials.

## Citation

If you use this code or data, please cite the accompanying CompFaith manuscript
after publication.
