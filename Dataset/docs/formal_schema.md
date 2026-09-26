# Formal Dataset Schema

`Dataset/processed/formal_records.jsonl` contains the 573 records used by the
CompFaith evaluation: 293 DIVERSUMM records, 80 LONGEVAL records, and 200
RAMPRASAD'24 records.

Each JSONL record contains:

- `id`: unique row identifier;
- `dataset`: one of the three dataset names;
- `context`: source document text;
- `candidate`: generated response text;
- `faithfulness`: continuous reference faithfulness value in `[0, 1]`;
- `faithful_label`: binary faithful label (`1` means faithful);
- `hallucination_label`: binary hallucination label (`1` means hallucinated).

The file contains only fields required by the CompFaith performance and speed
evaluation. No baseline predictions or baseline model outputs are included.

Label conversion follows the dataset annotations used for the paper:

- DIVERSUMM uses the native summary-level binary label and the proportion of
  `NoE` sentence annotations as the continuous faithfulness reference;
- LONGEVAL uses `fine_rating / 100`, with exactly `1.0` treated as faithful;
- RAMPRASAD'24 uses its native binary label and continuous faithfulness label.
