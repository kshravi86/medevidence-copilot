#!/usr/bin/env python3
"""
PubMedQA PQA-L Dataset Preparation and Split Generation.
- Pins official PubMedQA repository revision and SHA-256 checksums.
- Reproduces official 500/500 development/test split.
- Performs stratified sampling of 100 development questions (seed=42).
- Selects shortest remaining development example for yes/no/maybe as 3 fixed demonstrations.
- Strictly separates reference conclusions & answers from model inputs.
- Implements automated zero-overlap verification across pilot, demonstrations, and test sets.
- Generates comprehensive manifest and verification reports.
"""

import os
import json
import hashlib
import subprocess
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
# A clone of github.com/pubmedqa/pubmedqa pins the source revision; without one, the files that
# scripts/download_data.py puts in data/raw/ are used and the revision is recorded as "unknown".
PQA_REPO_DIR = Path(os.environ['PUBMEDQA_REPO_DIR']) if os.environ.get('PUBMEDQA_REPO_DIR') else None
_SRC = PQA_REPO_DIR / 'data' if PQA_REPO_DIR else ROOT / 'data' / 'raw'
ORI_PQAL_PATH = _SRC / 'ori_pqal.json'
TEST_GT_PATH = _SRC / 'test_ground_truth.json'

OUTPUT_DIRS = [ROOT / 'data' / 'prepared']

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def format_context(labels, contexts) -> str:
    parts = []
    for l, c in zip(labels, contexts):
        lbl = l.strip().upper() if l else ''
        if lbl:
            parts.append(f"{lbl}: {c.strip()}")
        else:
            parts.append(c.strip())
    return "\n\n".join(parts)

def main():
    print("=" * 80)
    print("PUBMEDQA PQA-L DATASET PREPARATION & SPLIT GENERATION")
    print("=" * 80)

    # 1. Verify existence of source files
    if not ORI_PQAL_PATH.exists() or not TEST_GT_PATH.exists():
        raise FileNotFoundError("Official PubMedQA source files not found.")

    # 2. Pin Source Revision & Checksums
    try:
        if PQA_REPO_DIR is None:
            raise FileNotFoundError
        git_rev = subprocess.check_output(['git', '-C', str(PQA_REPO_DIR), 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        git_remote = subprocess.check_output(['git', '-C', str(PQA_REPO_DIR), 'remote', 'get-url', 'origin'], stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_rev, git_remote = 'unknown', 'https://github.com/pubmedqa/pubmedqa'
    ori_pqal_sha256 = sha256_file(ORI_PQAL_PATH)
    test_gt_sha256 = sha256_file(TEST_GT_PATH)

    print(f"Git Remote:    {git_remote}")
    print(f"Git Revision:  {git_rev}")
    print(f"PQA-L SHA256:  {ori_pqal_sha256}")
    print(f"TestGT SHA256: {test_gt_sha256}")

    # 3. Load Datasets
    with open(ORI_PQAL_PATH, 'r', encoding='utf-8') as f:
        ori_pqal = json.load(f)
    with open(TEST_GT_PATH, 'r', encoding='utf-8') as f:
        test_gt = json.load(f)

    total_pmids = sorted(list(ori_pqal.keys()))
    test_pmids = sorted(list(test_gt.keys()))
    dev_pmids = sorted(list(set(total_pmids) - set(test_pmids)))

    # Assertions on official split
    assert len(total_pmids) == 1000, f"Expected 1000 PQA-L instances, got {len(total_pmids)}"
    assert len(test_pmids) == 500, f"Expected 500 test instances, got {len(test_pmids)}"
    assert len(dev_pmids) == 500, f"Expected 500 dev instances, got {len(dev_pmids)}"
    assert set(test_pmids).isdisjoint(set(dev_pmids)), "Test and Dev PMIDs must be disjoint!"

    total_label_counts = dict(Counter(ori_pqal[p]['final_decision'] for p in total_pmids))
    dev_label_counts = dict(Counter(ori_pqal[p]['final_decision'] for p in dev_pmids))
    test_label_counts = dict(Counter(ori_pqal[p]['final_decision'] for p in test_pmids))

    print(f"Total Label Counts (N=1000): {total_label_counts}")
    print(f"Dev Label Counts   (N=500):  {dev_label_counts}")
    print(f"Test Label Counts  (N=500):  {test_label_counts}")

    # 4. Stratified Sampling of 100 Development Questions (Seed = 42)
    dev_labels = [ori_pqal[p]['final_decision'] for p in dev_pmids]
    _, pilot_pmids, _, pilot_labels = train_test_split(
        dev_pmids, dev_labels, test_size=100, stratify=dev_labels, random_state=42
    )
    pilot_pmids = sorted(pilot_pmids)
    pilot_label_counts = dict(Counter(ori_pqal[p]['final_decision'] for p in pilot_pmids))
    print(f"Pilot Label Counts (N=100, seed=42): {pilot_label_counts}")

    # 5. Remaining Development Examples
    remaining_dev_pmids = sorted(list(set(dev_pmids) - set(pilot_pmids)))
    assert len(remaining_dev_pmids) == 400, f"Expected 400 remaining dev PMIDs, got {len(remaining_dev_pmids)}"

    # 6. Select Shortest Remaining Example for each class as Demonstrations
    demonstrations = {}
    for cls in ['yes', 'no', 'maybe']:
        candidates = [p for p in remaining_dev_pmids if ori_pqal[p]['final_decision'] == cls]
        # Shortest by total text length of context + question
        candidates_sorted = sorted(
            candidates,
            key=lambda p: (
                len(ori_pqal[p]['QUESTION'] + ' ' + ' '.join(ori_pqal[p]['CONTEXTS'])),
                len(ori_pqal[p]['QUESTION'].split() + ' '.join(ori_pqal[p]['CONTEXTS']).split()),
                p  # deterministic tie-breaker
            )
        )
        shortest_pmid = candidates_sorted[0]
        shortest_item = ori_pqal[shortest_pmid]
        ctx_formatted = format_context(shortest_item.get('LABELS', []), shortest_item.get('CONTEXTS', []))
        demonstrations[cls] = {
            "pmid": shortest_pmid,
            "class": cls,
            "char_length": len(shortest_item['QUESTION'] + ' ' + ' '.join(shortest_item['CONTEXTS'])),
            "word_length": len((shortest_item['QUESTION'] + ' ' + ' '.join(shortest_item['CONTEXTS'])).split()),
            "input": {
                "question_id": shortest_pmid,
                "question": shortest_item['QUESTION'],
                "context_labels": shortest_item.get('LABELS', []),
                "contexts": shortest_item.get('CONTEXTS', []),
                "formatted_context": ctx_formatted
            },
            "reference": {
                "final_decision": shortest_item['final_decision'],
                "long_answer": shortest_item.get('LONG_ANSWER', ''),
                "clinical_rationale": shortest_item.get('LONG_ANSWER', '')
            }
        }
        print(f"Selected Demonstration for '{cls.upper()}': PMID {shortest_pmid} ({demonstrations[cls]['char_length']} chars, {demonstrations[cls]['word_length']} words)")

    demo_pmids = sorted([d['pmid'] for d in demonstrations.values()])
    assert len(demo_pmids) == 3, "Expected exactly 3 demonstration PMIDs"

    # Remaining dev pool after removing demonstrations: 400 - 3 = 397
    final_remaining_dev_pmids = sorted(list(set(remaining_dev_pmids) - set(demo_pmids)))
    assert len(final_remaining_dev_pmids) == 397, f"Expected 397 remaining dev PMIDs, got {len(final_remaining_dev_pmids)}"

    # 7. Zero-Overlap Verification Checks
    s_pilot = set(pilot_pmids)
    s_demo = set(demo_pmids)
    s_test = set(test_pmids)
    s_rem = set(final_remaining_dev_pmids)

    overlap_pilot_demo = sorted(list(s_pilot & s_demo))
    overlap_pilot_test = sorted(list(s_pilot & s_test))
    overlap_demo_test = sorted(list(s_demo & s_test))
    overlap_pilot_rem = sorted(list(s_pilot & s_rem))
    overlap_demo_rem = sorted(list(s_demo & s_rem))
    overlap_test_rem = sorted(list(s_test & s_rem))

    total_union = s_pilot | s_demo | s_test | s_rem

    assert len(overlap_pilot_demo) == 0, f"Pilot and Demo overlap: {overlap_pilot_demo}"
    assert len(overlap_pilot_test) == 0, f"Pilot and Test overlap: {overlap_pilot_test}"
    assert len(overlap_demo_test) == 0, f"Demo and Test overlap: {overlap_demo_test}"
    assert len(overlap_pilot_rem) == 0, f"Pilot and Remaining Dev overlap: {overlap_pilot_rem}"
    assert len(overlap_demo_rem) == 0, f"Demo and Remaining Dev overlap: {overlap_demo_rem}"
    assert len(overlap_test_rem) == 0, f"Test and Remaining Dev overlap: {overlap_test_rem}"
    assert len(total_union) == 1000, f"Expected union of 1000, got {len(total_union)}"

    print("✓ All 6 zero-overlap assertions passed successfully! (0% overlap)")

    # 8. Separate Model Inputs from Ground Truth References
    pilot_inputs = []
    pilot_ground_truth = []
    for pmid in pilot_pmids:
        item = ori_pqal[pmid]
        ctx_formatted = format_context(item.get('LABELS', []), item.get('CONTEXTS', []))
        pilot_inputs.append({
            "question_id": pmid,
            "question": item['QUESTION'],
            "context_labels": item.get('LABELS', []),
            "contexts": item.get('CONTEXTS', []),
            "formatted_context": ctx_formatted
        })
        pilot_ground_truth.append({
            "question_id": pmid,
            "final_decision": item['final_decision'],
            "long_answer": item.get('LONG_ANSWER', ''),
            "reasoning_required_pred": item.get('reasoning_required_pred', ''),
            "reasoning_free_pred": item.get('reasoning_free_pred', '')
        })

    test_inputs = []
    test_ground_truth = []
    for pmid in test_pmids:
        item = ori_pqal[pmid]
        ctx_formatted = format_context(item.get('LABELS', []), item.get('CONTEXTS', []))
        test_inputs.append({
            "question_id": pmid,
            "question": item['QUESTION'],
            "context_labels": item.get('LABELS', []),
            "contexts": item.get('CONTEXTS', []),
            "formatted_context": ctx_formatted
        })
        test_ground_truth.append({
            "question_id": pmid,
            "final_decision": item['final_decision'],
            "long_answer": item.get('LONG_ANSWER', '')
        })

    # 9. Split Assignments Mapping
    split_assignments = {}
    for p in pilot_pmids:
        split_assignments[p] = "pilot"
    for p in demo_pmids:
        split_assignments[p] = "demonstration"
    for p in final_remaining_dev_pmids:
        split_assignments[p] = "dev_remaining"
    for p in test_pmids:
        split_assignments[p] = "test"
    assert len(split_assignments) == 1000

    # 10. Overlap Verification Report
    overlap_report = {
        "verification_status": "PASSED",
        "timestamp_utc": "2026-11-15T05:05:00Z",
        "seed": 42,
        "sampling_method": "stratified",
        "counts": {
            "pilot": len(pilot_pmids),
            "demonstrations": len(demo_pmids),
            "dev_remaining": len(final_remaining_dev_pmids),
            "test": len(test_pmids),
            "total_unique": len(total_union)
        },
        "pairwise_overlaps": {
            "pilot_vs_demonstrations": len(overlap_pilot_demo),
            "pilot_vs_test": len(overlap_pilot_test),
            "demonstrations_vs_test": len(overlap_demo_test),
            "pilot_vs_dev_remaining": len(overlap_pilot_rem),
            "demonstrations_vs_dev_remaining": len(overlap_demo_rem),
            "test_vs_dev_remaining": len(overlap_test_rem)
        },
        "assertions": {
            "pilot_count_is_100": len(pilot_pmids) == 100,
            "demonstrations_count_is_3": len(demo_pmids) == 3,
            "test_count_is_500": len(test_pmids) == 500,
            "dev_remaining_count_is_397": len(final_remaining_dev_pmids) == 397,
            "total_count_is_1000": len(total_union) == 1000,
            "pilot_and_demo_disjoint": len(overlap_pilot_demo) == 0,
            "pilot_and_test_disjoint": len(overlap_pilot_test) == 0,
            "demo_and_test_disjoint": len(overlap_demo_test) == 0
        }
    }

    # 11. Manifest
    manifest = {
        "dataset_name": "PubMedQA Expert-Labeled PQA-L Benchmark Splits",
        "source_repository": git_remote,
        "source_revision": git_rev,
        "source_files": {
            "ori_pqal.json": {
                "path": "data/ori_pqal.json",
                "sha256": ori_pqal_sha256,
                "total_instances": len(total_pmids)
            },
            "test_ground_truth.json": {
                "path": "data/test_ground_truth.json",
                "sha256": test_gt_sha256,
                "total_instances": len(test_pmids)
            }
        },
        "sampling_specification": {
            "method": "stratified",
            "seed": 42,
            "strata": "final_decision (yes, no, maybe)",
            "source_pool": "official development set (500 instances)"
        },
        "label_counts": {
            "total_pqal": total_label_counts,
            "official_dev": dev_label_counts,
            "official_test": test_label_counts,
            "pilot_100": pilot_label_counts,
            "demonstrations_3": {d["class"]: 1 for d in demonstrations.values()},
            "remaining_dev_397": dict(Counter(ori_pqal[p]['final_decision'] for p in final_remaining_dev_pmids))
        },
        "demonstrations": {
            cls: {
                "pmid": d["pmid"],
                "char_length": d["char_length"],
                "word_length": d["word_length"],
                "question": d["input"]["question"],
                "decision": d["reference"]["final_decision"]
            } for cls, d in demonstrations.items()
        },
        "files_generated": [
            "dataset_manifest.json",
            "split_assignments.json",
            "pilot_inputs.json",
            "pilot_ground_truth.json",
            "demonstrations.json",
            "test_inputs.json",
            "test_ground_truth.json",
            "overlap_verification.json"
        ]
    }

    # 12. Write Output Files to Output Directories
    files_to_write = {
        "dataset_manifest.json": manifest,
        "split_assignments.json": split_assignments,
        "pilot_inputs.json": pilot_inputs,
        "pilot_ground_truth.json": pilot_ground_truth,
        "demonstrations.json": demonstrations,
        "test_inputs.json": test_inputs,
        "test_ground_truth.json": test_ground_truth,
        "overlap_verification.json": overlap_report
    }

    for out_dir in OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting prepared files to: {out_dir}")
        for filename, content in files_to_write.items():
            filepath = out_dir / filename
            data_bytes = json.dumps(content, indent=2).encode('utf-8')
            with open(filepath, 'wb') as f:
                f.write(data_bytes)
            print(f"  -> {filename:<26} ({len(data_bytes)/1024:>6.1f} KB, sha256: {sha256_bytes(data_bytes)[:12]}...)")

    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY WITH 0% OVERLAP!")
    print("=" * 80)

if __name__ == '__main__':
    main()
