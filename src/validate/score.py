"""
Воспроизводит формулу оценки из ТЗ, чтобы калиброваться на ПУБЛИЧНОМ датасете,
для которого у нас есть ground_truth.json. На приватных данных ground truth не
будет — этот скрипт работает только с data/raw/ground_truth_PUBLIC_ONLY.json.

Правила (см. README задачи, раздел 4):
  status:            0.50, ровно совпадает с ключом, иначе вся ячейка = 0
  actual:            0.30 * max(0, 1 - e/0.05), e = |sub - key| / |key|
  evidence_txn_id:   0.20 если совпадает строкой; если в ключе null -
                     эти 0.20 тоже "плывут" по той же шкале e, что и actual
                     (используем ту же e, что для actual)

Если status неверен -> вся ячейка 0, независимо от остального.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GT_PATH = ROOT / "data/raw/ground_truth_PUBLIC_ONLY.json"


def score_cell(sub_cell: dict, key_cell: dict) -> tuple[float, dict]:
    detail = {}

    status_ok = sub_cell.get("status") == key_cell["status"]
    detail["status_ok"] = status_ok
    if not status_ok:
        detail["score"] = 0.0
        return 0.0, detail

    score = 0.50

    key_actual = key_cell["actual"]
    sub_actual = sub_cell.get("actual")
    if isinstance(sub_actual, (int, float)) and not isinstance(sub_actual, bool) and key_actual != 0:
        e = abs(sub_actual - key_actual) / abs(key_actual)
    elif isinstance(sub_actual, (int, float)) and not isinstance(sub_actual, bool) and key_actual == 0:
        e = 0.0 if sub_actual == 0 else 1.0
    else:
        e = 1.0  # отсутствует / не число -> максимальная погрешность

    actual_score = 0.30 * max(0.0, 1 - e / 0.05)
    score += actual_score
    detail["actual_error_rel"] = e
    detail["actual_score"] = actual_score

    key_ev = key_cell["evidence_txn_id"]
    if key_ev is None:
        # плывёт по той же шкале, что и actual
        ev_score = 0.20 * max(0.0, 1 - e / 0.05)
        detail["evidence_mode"] = "null-key-scaled-with-actual"
    else:
        ev_score = 0.20 if sub_cell.get("evidence_txn_id") == key_ev else 0.0
        detail["evidence_mode"] = "exact-match"
    score += ev_score
    detail["evidence_score"] = ev_score

    detail["score"] = score
    return score, detail


def main() -> int:
    sub_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/processed/submission.json"
    verbose = "-v" in sys.argv

    submission = json.loads(sub_path.read_text(encoding="utf-8"))
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))

    total = 0.0
    max_total = 0.0
    n_cells = 0
    worst: list[tuple[float, str]] = []

    for sid, covs in gt["scenarios"].items():
        for cov, key_cell in covs["covenants"].items():
            sub_cell = submission.get("answers", {}).get(sid, {}).get(cov)
            n_cells += 1
            max_total += 1.0
            if sub_cell is None:
                worst.append((0.0, f"{sid}/{cov}: ячейка отсутствует"))
                continue
            s, detail = score_cell(sub_cell, key_cell)
            total += s
            worst.append((s, f"{sid}/{cov}: {s:.3f}  (status_ok={detail.get('status_ok')}, "
                              f"e={detail.get('actual_error_rel', 'n/a')})"))

    worst.sort(key=lambda x: x[0])

    print(f"Ячеек оценено: {n_cells}")
    print(f"ИТОГО: {total:.3f} / {max_total:.3f}  ({100*total/max_total:.1f}%)")

    if verbose:
        print("\nХудшие ячейки:")
        for s, msg in worst[:20]:
            print(f"  {s:.3f}  {msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
