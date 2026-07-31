"""
Rigorous baseline-vs-fine-tuned evaluation.

METRICS (and why each one exists, tied to a named failure mode above):

1. Field-level Precision / Recall / F1, per top-level field (skills,
   education, certifications, career_gaps). A predicted item counts as a
   match only if its key identifying attribute matches the ground truth
   (e.g. skill name, degree+institution, cert name, gap start/end year) --
   this is stricter than just checking the model produced *a* JSON object,
   and is what "clear quantitative metrics" in the brief requires.

2. Hallucination rate: fraction of NON-NULL predicted values that have no
   ground-truth counterpart at all, OR whose evidence_span does not actually
   appear as a substring of the source resume text. This is the direct,
   automatic check for "Schema Drift Under Fine-Tuning" -- a model that
   invents a graduation year, or cites evidence text that isn't really in
   the document, gets caught here even if it otherwise "looks" plausible.

3. Null-precision: of the fields where the ground truth is null (no
   evidence exists), what fraction did the model correctly leave null rather
   than fabricate? This is the metric that would have caught the earlier
   failed attempt's core problem before it ever reached recruiters.

4. Same suite computed separately on data/test.jsonl (in-distribution
   industries) and data/test_ood.jsonl (industries excluded from
   train/val) -- this is what "avoids overfitting to a narrow slice" is
   checked against, and is exactly the comparison the earlier failed
   attempt skipped.

Usage:
    python src/evaluate.py --baseline_preds outputs/baseline_test.jsonl \
        --ft_preds outputs/finetuned_test.jsonl --split_name test
"""
import argparse
import json
from collections import defaultdict


def skill_key(s):
    return (s.get("name") or "").strip().lower()


def edu_key(e):
    return ((e.get("degree") or "").strip().lower(), (e.get("institution") or "").strip().lower())


def cert_key(c):
    return (c.get("name") or "").strip().lower()


def gap_key(g):
    return (g.get("start_year"), g.get("end_year"))


FIELD_KEY_FN = {
    "skills": skill_key,
    "education": edu_key,
    "certifications": cert_key,
    "career_gaps": gap_key,
}


def prf1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def score_field(gt_list, pred_list, key_fn):
    gt_keys = {key_fn(x) for x in gt_list}
    pred_keys = {key_fn(x) for x in (pred_list or [])}
    tp = len(gt_keys & pred_keys)
    fp = len(pred_keys - gt_keys)
    fn = len(gt_keys - pred_keys)
    return tp, fp, fn


def hallucination_and_null_precision(example):
    """Returns (n_hallucinated, n_nonnull_predicted, n_correct_nulls, n_gt_nulls)
    for the sub-fields inside each item (years_experience, graduation_year, etc.)
    where the ground truth explicitly has a null and we check if the model
    respected that, plus a raw evidence-grounding check."""
    text = example["text"]
    pred = example["prediction"] or {}
    label = example["label"]

    n_halluc, n_nonnull_pred = 0, 0
    n_correct_null, n_gt_null = 0, 0

    for field_name, key_fn in FIELD_KEY_FN.items():
        gt_items = {key_fn(x): x for x in label.get(field_name, [])}
        pred_items = pred.get(field_name, []) if isinstance(pred.get(field_name), list) else []
        for p in pred_items:
            evidence = p.get("evidence_span")
            if evidence:
                n_nonnull_pred += 1
                if evidence not in text:
                    n_halluc += 1  # cited evidence that isn't actually in the source
            k = key_fn(p)
            if k not in gt_items:
                # predicted an entity with no ground-truth counterpart at all
                if evidence:
                    n_halluc += 1
                continue
            gt_item = gt_items[k]
            for subfield in ("years_experience", "graduation_year", "year", "duration_months"):
                if subfield in gt_item:
                    if gt_item[subfield] is None:
                        n_gt_null += 1
                        if p.get(subfield) is None:
                            n_correct_null += 1
                        else:
                            n_halluc += 1  # fabricated a value where none was supported

    return n_halluc, n_nonnull_pred, n_correct_null, n_gt_null


def evaluate_predictions(pred_path):
    totals = {f: [0, 0, 0] for f in FIELD_KEY_FN}  # tp, fp, fn
    total_halluc = total_nonnull = total_correct_null = total_gt_null = 0
    n_parse_failures = 0
    n = 0

    with open(pred_path) as f:
        for line in f:
            ex = json.loads(line)
            n += 1
            if ex.get("prediction") is None:
                n_parse_failures += 1
                continue
            for field_name, key_fn in FIELD_KEY_FN.items():
                tp, fp, fn = score_field(
                    ex["label"].get(field_name, []),
                    ex["prediction"].get(field_name, []),
                    key_fn,
                )
                totals[field_name][0] += tp
                totals[field_name][1] += fp
                totals[field_name][2] += fn
            h, np_, cn, gn = hallucination_and_null_precision(ex)
            total_halluc += h
            total_nonnull += np_
            total_correct_null += cn
            total_gt_null += gn

    report = {"n_examples": n, "json_parse_failure_rate": n_parse_failures / n if n else 0.0}
    macro_f1s = []
    for field_name, (tp, fp, fn) in totals.items():
        p, r, f1 = prf1(tp, fp, fn)
        report[field_name] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3), "support": tp + fn}
        macro_f1s.append(f1)
    report["macro_f1"] = round(sum(macro_f1s) / len(macro_f1s), 3) if macro_f1s else 0.0
    report["hallucination_rate"] = round(total_halluc / total_nonnull, 3) if total_nonnull else 0.0
    report["null_precision"] = round(total_correct_null / total_gt_null, 3) if total_gt_null else 0.0
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_preds", required=True)
    ap.add_argument("--ft_preds", required=True)
    ap.add_argument("--split_name", default="test")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    baseline_report = evaluate_predictions(args.baseline_preds)
    ft_report = evaluate_predictions(args.ft_preds)

    combined = {"split": args.split_name, "baseline": baseline_report, "fine_tuned": ft_report}
    print(json.dumps(combined, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(combined, f, indent=2)


if __name__ == "__main__":
    main()
