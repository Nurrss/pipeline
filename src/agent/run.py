"""
Оркестратор: собирает весь пайплайн в submission.json.

    python3 -m src.agent.run --out data/processed/submission.json

Шаги:
  1. Леджер + документы -> account_id<->scenario_id, пачки документов по сценарию.
  2. covenant_parser: Статья 6 каждого действующего договора -> CovenantFormula (LLM,
     батч-вызов на все сценарии сразу, с кэшем).
  3. fact_extractor: весь вспомогательный корпус документов сценария -> ScenarioFacts
     (LLM, батч-вызов на все сценарии сразу, с кэшем).
  4. calculate.apply_facts + evaluate_formula: детерминированный расчёт actual/status.
  5. evidence.find_evidence_txn_id: поиск evidence_txn_id.
  6. Сборка в структуру submission_template.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ на sys.path — см. ingest/documents.py

from agent.covenant_parser import parse_all_scenarios
from agent.fact_extractor import bundle_docs_text, extract_all_scenario_facts
from covenants.calculate import apply_facts, evaluate_formula
from covenants.evidence import find_evidence_txn_id
from ingest.documents import build_scenario_bundles, load_all_documents
from ingest.ledger import build_account_to_scenario_map, load_ledger, txns_for_scenario

ROOT = Path(__file__).resolve().parents[2]

BORROWER_RE = re.compile(r"Заёмщик[,:]?\s*([A-Z][\w\s]+(?:JSC|LLP|LLC))")


def _extract_article6(agreement_text: str) -> str:
    idxs = [m.start() for m in re.finditer(r"Статья 6\b", agreement_text)]
    if not idxs:
        raise ValueError("В договоре не найдена Статья 6")
    idx = idxs[-1]  # последнее вхождение — тело статьи, не оглавление
    end = agreement_text.find("Статья 7", idx)
    return agreement_text[idx:end if end != -1 else None]


def _guess_borrower(text: str) -> str | None:
    m = BORROWER_RE.search(text)
    return m.group(1).strip() if m else None


def build_submission(
    *,
    team: str,
    contact_email: str,
    model: str,
    data_raw: Path,
) -> dict:
    ledger = load_ledger(data_raw / "master_ledger_2025.csv", warn=False)
    template = json.loads((data_raw / "submission_template.json").read_text(encoding="utf-8"))
    known_scenarios = set(template["answers"].keys())
    acc_map = build_account_to_scenario_map(ledger, known_scenarios)

    docs = load_all_documents(data_raw / "documents", set(acc_map.keys()))
    bundles = build_scenario_bundles(docs, acc_map)

    missing_agreements = [sid for sid, b in bundles.items() if b.current_agreement is None]
    if missing_agreements:
        raise RuntimeError(f"Нет действующего договора для сценариев: {missing_agreements}")

    covenant_inputs: dict[str, tuple[str, str]] = {}
    fact_inputs: dict[str, tuple[str, str]] = {}
    for sid, b in bundles.items():
        article6 = _extract_article6(b.current_agreement.text)
        borrower = _guess_borrower(article6) or sid
        covenant_inputs[sid] = (borrower, article6)
        fact_inputs[sid] = (borrower, bundle_docs_text(b))

    print(f"[run] Парсинг ковенантов (LLM, батч на {len(covenant_inputs)} сценариев)...")
    all_formulas = parse_all_scenarios(covenant_inputs)

    print(f"[run] Извлечение фактов (LLM, батч на {len(fact_inputs)} сценариев)...")
    all_facts = extract_all_scenario_facts(fact_inputs)

    answers: dict[str, dict[str, dict]] = {}
    for sid in sorted(known_scenarios):
        txns = txns_for_scenario(ledger, sid)
        facts = all_facts[sid]
        resolved = apply_facts(txns, facts)
        formulas = {f.covenant_id: f for f in all_formulas[sid]}

        cell_answers: dict[str, dict] = {}
        for cov_id in template["answers"][sid]:
            formula = formulas.get(cov_id)
            if formula is None:
                print(f"[run] ВНИМАНИЕ: нет формулы для {sid}/{cov_id} — оставляю пустую ячейку")
                cell_answers[cov_id] = {"status": "COMPLIANT", "actual": 0.0, "evidence_txn_id": None}
                continue

            result = evaluate_formula(formula, resolved, facts)
            evidence = find_evidence_txn_id(formula, txns, resolved, facts, result)
            cell_answers[cov_id] = {
                "status": result.status,
                "actual": round(result.actual, 2),
                "evidence_txn_id": evidence,
            }
        answers[sid] = cell_answers

    return {
        "team": team,
        "contact_email": contact_email,
        "model": model,
        "answers": answers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать submission.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/submission.json")
    parser.add_argument("--data-raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--team", default="halyk-agent")
    parser.add_argument("--contact-email", default="nurrs.serkul@gmail.com")
    parser.add_argument("--model", default="gemini-2.5-flash")
    args = parser.parse_args()

    submission = build_submission(
        team=args.team,
        contact_email=args.contact_email,
        model=args.model,
        data_raw=args.data_raw,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run] Записано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
