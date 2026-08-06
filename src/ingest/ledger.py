"""
Загрузка master_ledger_2025.csv и извлечение scenario_id из txn_id.

Формат txn_id: TXN-<scenario_id>-<seq>, например TXN-P1-0039, TXN-B4-0012.
Важно: в леджере присутствует много "шумовых" транзакций с чисто числовыми
префиксами (TXN-9001-..., TXN-9529-... и т.д.) — они НЕ относятся ни к одному
из scenario_id в submission_template.json и должны игнорироваться при расчётах
по конкретному заёмщику. Мы их не выбрасываем из общего датафрейма (могут
пригодиться как "шум" для проверки устойчивости логики), но всегда фильтруем
по нужному account_id/scenario_id явно, а не "по всему файлу".
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


TXN_ID_RE = re.compile(r"^TXN-(?P<scenario>[A-Za-z0-9]+)-(?P<seq>\d+)$")


@dataclass(frozen=True)
class Txn:
    txn_id: str
    date: str
    account_id: str
    counterparty: str
    description: str
    amount: float | None  # None = сумма отсутствует в леджере (см. dirty-data ниже)
    currency: str

    @property
    def scenario_id(self) -> str | None:
        m = TXN_ID_RE.match(self.txn_id)
        if not m:
            return None
        return m.group("scenario")

    @property
    def abs_amount(self) -> float | None:
        return None if self.amount is None else abs(self.amount)

    @property
    def is_dirty(self) -> bool:
        """Строка с отсутствующей суммой — требует уточнения из документов."""
        return self.amount is None


def load_ledger(path: str | Path, warn: bool = True) -> list[Txn]:
    """
    Датасет намеренно "грязный": в паре строк отсутствует amount (пустая
    строка в CSV). Такие строки НЕ отбрасываем и НЕ падаем на них — сохраняем
    с amount=None и явно логируем, чтобы даунстрим-логика (ковенанты) решала,
    как их трактовать: обычно верная сумма находится в PDF-документе
    (счёт/акт), который и нужно найти для этой транзакции.
    """
    path = Path(path)
    txns: list[Txn] = []
    dirty: list[str] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_amount = row["amount"]
            amount = float(raw_amount) if raw_amount not in ("", None) else None
            if amount is None:
                dirty.append(row["txn_id"])
            txns.append(
                Txn(
                    txn_id=row["txn_id"],
                    date=row["date"],
                    account_id=row["account_id"],
                    counterparty=row["counterparty"],
                    description=row["description"],
                    amount=amount,
                    currency=row["currency"],
                )
            )
    if warn and dirty:
        print(f"[ledger] ВНИМАНИЕ: {len(dirty)} транзакций без суммы (нужно уточнение из документов): {dirty}")
    return txns


def build_account_to_scenario_map(txns: list[Txn], known_scenarios: set[str]) -> dict[str, str]:
    """
    Строит account_id -> scenario_id, используя ТОЛЬКО транзакции, чей
    scenario_id (префикс txn_id) уже входит в known_scenarios (т.е. это
    заёмщики из submission_template.json, а не шумовые TXN-9xxx-...).

    Бросает ValueError, если один account_id встречается более чем с одним
    scenario_id — это означает ошибку в данных или в нашем regex.
    """
    mapping: dict[str, str] = {}
    for t in txns:
        sid = t.scenario_id
        if sid is None or sid not in known_scenarios:
            continue
        prev = mapping.get(t.account_id)
        if prev is not None and prev != sid:
            raise ValueError(
                f"account_id {t.account_id} сопоставлен более чем одному scenario_id: {prev} и {sid}"
            )
        mapping[t.account_id] = sid
    return mapping


def txns_for_scenario(txns: list[Txn], scenario_id: str) -> list[Txn]:
    return [t for t in txns if t.scenario_id == scenario_id]


if __name__ == "__main__":
    import json
    import sys

    root = Path(__file__).resolve().parents[2]
    ledger = load_ledger(root / "data/raw/master_ledger_2025.csv")
    template = json.loads((root / "data/raw/submission_template.json").read_text())
    known = set(template["answers"].keys())

    acc_map = build_account_to_scenario_map(ledger, known)
    print(f"Всего транзакций в леджере: {len(ledger)}")
    print(f"Известных сценариев (из шаблона): {len(known)} -> {sorted(known)}")
    print(f"Найдено account_id для известных сценариев: {len(acc_map)}")
    for acc, sid in sorted(acc_map.items(), key=lambda kv: kv[1]):
        cnt = len(txns_for_scenario(ledger, sid))
        print(f"  {sid:>4}  <-  {acc}   ({cnt} транзакций)")
