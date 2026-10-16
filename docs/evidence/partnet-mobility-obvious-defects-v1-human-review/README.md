# PartNet-Mobility obvious-defect human review

This directory is a human-review overlay for `../partnet-mobility-obvious-defects-v1/`.
The parent candidate bundle remains unchanged and reproducible.

- `submissions.raw.jsonl` preserves all ten submitted records. The duplicate case
  `odr-902e87011e30` has matching structured judgments and two compatible notes.
- `human_review.jsonl` contains one canonical record for each of the nine cases;
  the duplicate notes are merged without changing the judgment.
- `private/adjudication.jsonl` is derived by comparing the canonical blind review
  against private Gold and checker output.
- `summary.json` records counts, hashes, duplicate handling, and scope boundaries.

Reproduce the private adjudication from the repository root:

```bash
python -m scripts.adjudicate_obvious_defect_review
python -m pytest tests/unit/test_obvious_defect_adjudication.py -q
```

`review_approved_candidate` means that this review gate passed. It does not mean
the case has been exported to an SFT dataset or used for training.
