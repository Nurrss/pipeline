"""
Читает все PDF из data/raw/documents/, классифицирует их по типу и привязывает
к account_id (а через ledger — к scenario_id).

Типы документов:
  CURRENT_AGREEMENT    - действующий договор банковского займа (используем его!)
  OUTDATED_AGREEMENT   - явно помечен "НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ" - игнорируем при расчётах,
                         но держим под рукой на случай явного запроса на сравнение
  AUDIT_NOTES          - примечания аудитора: переклассификации, cut-off, связанные стороны
  KYC_DOSSIER          - официальное досье с % долей контрагентов
  DECOY                - нерелевантный шум (статус-репорты, обобщённые процедуры и т.п.)

ВАЖНО: некоторые "decoy" документы специально похожи на настоящие (например,
"Процедура комплаенса" содержит слово KYC, но явно пишет, что не является
клиентским досье). Поэтому реальное KYC-досье распознаём по регистрационному
номеру формата KYC-ACC-XXXX-YYYY, а не просто по наличию слова "KYC".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

ACC_RE = re.compile(r"\bACC-\d{4}\b")
KYC_REGNUM_RE = re.compile(r"\bKYC-ACC-\d{4}-\d{4}\b")


@dataclass
class ParsedDoc:
    path: Path
    text: str
    doc_type: str
    account_ids: set[str] = field(default_factory=set)


def classify(text: str) -> str:
    if "ДОГОВОР БАНКОВСКОГО ЗАЙМА" in text:
        return "OUTDATED_AGREEMENT" if "НЕДЕЙСТВУЮЩАЯ" in text else "CURRENT_AGREEMENT"
    if "Процедура комплаенса" in text:
        return "DECOY"
    if KYC_REGNUM_RE.search(text):
        return "KYC_DOSSIER"
    if "Примечания к финансовой отчётности" in text or "АУДИТОР" in text.upper():
        return "AUDIT_NOTES"
    return "DECOY"


def load_all_documents(documents_dir: str | Path, known_accounts: set[str]) -> list[ParsedDoc]:
    documents_dir = Path(documents_dir)
    docs: list[ParsedDoc] = []
    for pdf_path in sorted(documents_dir.glob("*.pdf")):
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        accs = set(ACC_RE.findall(text)) & known_accounts
        docs.append(ParsedDoc(path=pdf_path, text=text, doc_type=classify(text), account_ids=accs))
    return docs


@dataclass
class ScenarioBundle:
    scenario_id: str
    account_id: str
    current_agreement: ParsedDoc | None = None
    outdated_agreement: ParsedDoc | None = None
    audit_notes: list[ParsedDoc] = field(default_factory=list)
    kyc_dossiers: list[ParsedDoc] = field(default_factory=list)
    decoys: list[ParsedDoc] = field(default_factory=list)


def build_scenario_bundles(docs: list[ParsedDoc], account_to_scenario: dict[str, str]) -> dict[str, ScenarioBundle]:
    bundles: dict[str, ScenarioBundle] = {}
    for acc, sid in account_to_scenario.items():
        bundles[sid] = ScenarioBundle(scenario_id=sid, account_id=acc)

    for d in docs:
        for acc in d.account_ids:
            sid = account_to_scenario.get(acc)
            if sid is None:
                continue
            b = bundles[sid]
            if d.doc_type == "CURRENT_AGREEMENT":
                if b.current_agreement is not None:
                    raise ValueError(f"{sid}: несколько действующих договоров! {b.current_agreement.path}, {d.path}")
                b.current_agreement = d
            elif d.doc_type == "OUTDATED_AGREEMENT":
                b.outdated_agreement = d
            elif d.doc_type == "AUDIT_NOTES":
                b.audit_notes.append(d)
            elif d.doc_type == "KYC_DOSSIER":
                b.kyc_dossiers.append(d)
            else:
                b.decoys.append(d)
    return bundles


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ingest.ledger import load_ledger, build_account_to_scenario_map
    import json

    root = Path(__file__).resolve().parents[2]
    ledger = load_ledger(root / "data/raw/master_ledger_2025.csv", warn=False)
    template = json.loads((root / "data/raw/submission_template.json").read_text())
    known_scenarios = set(template["answers"].keys())
    acc_map = build_account_to_scenario_map(ledger, known_scenarios)

    docs = load_all_documents(root / "data/raw/documents", set(acc_map.keys()))
    bundles = build_scenario_bundles(docs, acc_map)

    for sid in sorted(bundles):
        b = bundles[sid]
        ok = "OK" if b.current_agreement else "!! НЕТ ДЕЙСТВУЮЩЕГО ДОГОВОРА !!"
        print(
            f"{sid:>4} ({b.account_id}): договор={ok}, "
            f"аудит={len(b.audit_notes)}, kyc={len(b.kyc_dossiers)}, decoy={len(b.decoys)}"
        )
