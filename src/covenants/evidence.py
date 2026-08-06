"""
Поиск evidence_txn_id.

По CASE.md: "evidence_txn_id — единственная транзакция, определяющая результат: та,
чья переклассификация, включение, исключение или исправление приводит к нарушению
ковенанта. Уберите её — и вердикт изменится... Транзакция, которая лишь вносит вклад
в сумму, доказательством не является: ни самая крупная строка... ни та, что случайно
вывела накопленную сумму за порог."

Эмпирически (см. историю коммитов — калибровка по ground_truth_PUBLIC_ONLY.json) это
означает ДВА разных источника кандидатов, а не один общий брутфорс по всем
транзакциям:

1. Реклассификации (не void): проверяем ОТМЕНУ конкретной правки (остальное — как
   есть). cutoff-исключения и исправления отсутствующих сумм НЕ дают кандидатов —
   калибровка показала, что такие правки статистически дают ложные срабатывания
   (грейдер их не засчитывает evidence), видимо потому что это установление факта,
   а не классификационное суждение, которое можно "обжаловать".
2. Платежи связанным сторонам (related_party_payments): здесь, наоборот, работает
   именно прямой брутфорс — удаление ОДНОЙ конкретной транзакции из набора
   "платежи связанным сторонам" и проверка, меняется ли вердикт. Обычную сумму по
   категории (REVENUE/CAPEX/...) так НЕ проверяем: почти всегда такая категория состоит
   из одной-двух транзакций, и тест тривиально "находит" любую из них — что не
   является настоящим доказательством в смысле CASE.md ("не просто крупную сумму").

Единственный найденный кандидат -> evidence_txn_id; ноль или несколько -> null.
"""
from __future__ import annotations

from covenants.calculate import EvalResult, _normalize_name, apply_facts, evaluate_formula, resolve_value
from covenants.schema import CovenantFormula, ScenarioFacts
from ingest.ledger import Txn


def _match_reclass_txn_id(r, txns: list[Txn], facts: ScenarioFacts) -> str | None:
    if r.counterparty is None or r.amount is None:
        return None
    norm_cp = _normalize_name(r.counterparty)
    missing_by_id = {m.txn_id: m.resolved_amount for m in facts.missing_amounts}
    for t in txns:
        existing_amount = t.amount if t.amount is not None else missing_by_id.get(t.txn_id)
        if existing_amount is None:
            continue
        if _normalize_name(t.counterparty) == norm_cp and abs(abs(existing_amount) - r.amount) < 1.0:
            return t.txn_id
    return None


def _flips(formula: CovenantFormula, txns: list[Txn], reverted_facts: ScenarioFacts, baseline: EvalResult) -> bool:
    resolved = apply_facts(txns, reverted_facts)
    result = evaluate_formula(formula, resolved, reverted_facts)
    return result.status != baseline.status


def _related_party_source_ids(formula: CovenantFormula, resolved, facts: ScenarioFacts) -> list[str]:
    ids: list[str] = []
    for source in [formula.numerator, formula.denominator, *formula.terms]:
        if source is not None and source.kind == "related_party_payments":
            _, source_ids = resolve_value(source, resolved, facts, formula.period_start, formula.period_end)
            ids.extend(source_ids)
    return ids


def find_evidence_txn_id(
    formula: CovenantFormula,
    txns: list[Txn],
    resolved,
    facts: ScenarioFacts,
    baseline: EvalResult,
) -> str | None:
    candidates: list[str] = []

    for i, r in enumerate(facts.reclassifications):
        if r.void:
            continue
        txn_id = r.txn_id or _match_reclass_txn_id(r, txns, facts)
        if txn_id is None:
            continue
        reverted = facts.model_copy(deep=True)
        reverted.reclassifications = [rr for j, rr in enumerate(facts.reclassifications) if j != i]
        if _flips(formula, txns, reverted, baseline):
            candidates.append(txn_id)

    for txn_id in dict.fromkeys(_related_party_source_ids(formula, resolved, facts)):
        sub_resolved = [rt for rt in resolved if rt.txn.txn_id != txn_id]
        result = evaluate_formula(formula, sub_resolved, facts)
        if result.status != baseline.status:
            candidates.append(txn_id)

    candidates = list(dict.fromkeys(candidates))  # dedupe, сохраняя порядок
    return candidates[0] if len(candidates) == 1 else None
