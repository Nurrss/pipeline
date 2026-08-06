"""
Детерминированный расчётный слой. LLM здесь не участвует: на входе — уже
структурированные CovenantFormula (agent/covenant_parser.py) и ScenarioFacts
(agent/fact_extractor.py), на выходе — actual/status по чистой арифметике по
транзакциям сценария.

Общая идея по знакам, проверенная численно против ground truth публичного датасета
(см. историю коммитов): расходы записаны в леджере отрицательными суммами, доходы —
положительными; ЕСТЬ реверсирующие/сторнирующие проводки (возвраты, перерасчёты) —
поэтому категории всегда суммируются НЕТТО (по знаку), а не суммой модулей, иначе
сторно задваивает сумму вместо того чтобы её скомпенсировать.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from covenants.categorize import categorize
from covenants.schema import CovenantFormula, ScenarioFacts, ValueSource
from ingest.ledger import Txn

# Категории, чья "естественная" величина — расход (отчитываем как положительную
# magnitude); REVENUE и FINANCING_INFLOW естественно положительны (приток).
_EXPENSE_LIKE = {
    "OPEX", "CAPEX", "INTEREST", "TAX", "INSURANCE", "RENT",
    "PAYROLL", "UTILITIES", "MARKETING", "MAINTENANCE", "ADVISORY",
}

_TRANSFER_TO_SUBSIDIARY_RE = re.compile(r"transfer .* to subsidiary", re.I)


def _normalize_name(name: str) -> str:
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)  # отбросить конечное "(...)" — пометки филиала/локации
    return re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ]", "", name).upper()


@dataclass
class ResolvedTxn:
    txn: Txn
    category: str
    amount: float  # уже разрешённая сумма (missing_amounts применены), со знаком


def apply_facts(txns: list[Txn], facts: ScenarioFacts) -> list[ResolvedTxn]:
    """Применяет факты аудита к транзакциям сценария:
    - cutoffs: транзакция целиком исключается из периода;
    - missing_amounts: подставляется раскрытая сумма вместо amount=None;
    - reclassifications (не void): меняется категория транзакции. Матчим сначала по
      txn_id, если он не назван — по совпадению counterparty (нормализованному) и
      abs(amount) с точностью до $1.
    void-факты (только из промежуточных/черновых документов, не подтверждённые
    финальным отчётом) игнорируются полностью — см. докстринг fact_extractor.py.
    """
    cutoff_ids = {c.txn_id for c in facts.cutoffs}
    missing_by_id = {m.txn_id: m.resolved_amount for m in facts.missing_amounts}

    reclass_by_id: dict[str, str] = {}
    for r in facts.reclassifications:
        if r.void or r.to_category is None:
            continue
        if r.txn_id:
            reclass_by_id[r.txn_id] = r.to_category
        elif r.counterparty and r.amount is not None:
            norm_cp = _normalize_name(r.counterparty)
            for t in txns:
                existing_amount = t.amount if t.amount is not None else missing_by_id.get(t.txn_id)
                if existing_amount is None:
                    continue
                if _normalize_name(t.counterparty) == norm_cp and abs(abs(existing_amount) - r.amount) < 1.0:
                    reclass_by_id[t.txn_id] = r.to_category
                    break

    resolved: list[ResolvedTxn] = []
    for t in txns:
        if t.txn_id in cutoff_ids:
            continue
        amount = t.amount if t.amount is not None else missing_by_id.get(t.txn_id)
        if amount is None:
            continue  # сумма так и не разрешена — не можем учесть в расчёте
        category = reclass_by_id.get(t.txn_id, categorize(t.description).category)
        resolved.append(ResolvedTxn(txn=t, category=category, amount=amount))
    return resolved


def related_party_names(facts: ScenarioFacts) -> set[str]:
    return {
        _normalize_name(rp.name)
        for rp in facts.related_parties
        if rp.ownership_pct >= facts.related_party_threshold_pct
    }


def _in_period(date_str: str, start: str, end: str) -> bool:
    return start <= date_str <= end


def _category_net(resolved: list[ResolvedTxn], categories: list[str], start: str, end: str) -> tuple[float, list[str]]:
    total = 0.0
    ids = []
    for rt in resolved:
        if rt.category in categories and _in_period(rt.txn.date, start, end):
            total += rt.amount
            ids.append(rt.txn.txn_id)
    return total, ids


def _related_party_total(
    resolved: list[ResolvedTxn], related_names: set[str], start: str, end: str
) -> tuple[float, list[str]]:
    total = 0.0
    ids = []
    for rt in resolved:
        if _normalize_name(rt.txn.counterparty) in related_names and _in_period(rt.txn.date, start, end):
            total += rt.amount
            ids.append(rt.txn.txn_id)
    return total, ids


def _asset_transfer_subsidiary_total(
    resolved: list[ResolvedTxn], start: str, end: str
) -> tuple[float, list[str]]:
    total = 0.0
    ids = []
    for rt in resolved:
        if (
            rt.category == "CAPEX"
            and _TRANSFER_TO_SUBSIDIARY_RE.search(rt.txn.description)
            and _in_period(rt.txn.date, start, end)
        ):
            total += rt.amount
            ids.append(rt.txn.txn_id)
    return total, ids


def resolve_value(
    source: ValueSource,
    resolved: list[ResolvedTxn],
    facts: ScenarioFacts,
    period_start: str,
    period_end: str,
) -> tuple[float, list[str]]:
    """Возвращает (значение, [txn_id, ...] источников). Категориальные/агрегатные
    величины возвращаются как MAGNITUDE (>=0); EBITDA/adjusted_ebitda — со знаком
    (может быть отрицательной)."""
    related_names = related_party_names(facts)

    if source.kind == "category_sum":
        net, ids = _category_net(resolved, source.categories, period_start, period_end)
        return abs(net), ids

    if source.kind == "ebitda":
        rev_net, rev_ids = _category_net(resolved, ["REVENUE"], period_start, period_end)
        opex_net, opex_ids = _category_net(resolved, ["OPEX"], period_start, period_end)
        return abs(rev_net) - abs(opex_net), rev_ids + opex_ids

    if source.kind == "adjusted_ebitda":
        rev_net, rev_ids = _category_net(resolved, ["REVENUE"], period_start, period_end)
        opex_net, opex_ids = _category_net(resolved, ["OPEX"], period_start, period_end)
        addback_total = sum(a.amount for a in facts.addbacks)
        addback_ids = [a.txn_id for a in facts.addbacks if a.txn_id]
        return abs(rev_net) - abs(opex_net) + addback_total, rev_ids + opex_ids + addback_ids

    if source.kind == "related_party_payments":
        net, ids = _related_party_total(resolved, related_names, period_start, period_end)
        return abs(net), ids

    if source.kind == "asset_transfer_subsidiary":
        net, ids = _asset_transfer_subsidiary_total(resolved, period_start, period_end)
        return abs(net), ids

    if source.kind == "personnel_obligations":
        payroll_net, ids = _category_net(resolved, ["PAYROLL"], period_start, period_end)
        point_in_time_total = sum(p.amount for p in facts.point_in_time_items)
        return abs(payroll_net) + point_in_time_total, ids

    if source.kind == "group_capex":
        # Консолидированные капзатраты группы обычно не раскрываются документами —
        # детерминированный fallback: собственные капзатраты сценария (задокументировано
        # как приближение).
        net, ids = _category_net(resolved, ["CAPEX"], period_start, period_end)
        return abs(net), ids

    if source.kind == "constant":
        return (source.constant_value or 0.0), []

    raise ValueError(f"Неизвестный ValueKind: {source.kind}")


@dataclass
class EvalResult:
    status: str  # "COMPLIANT" | "BREACH"
    actual: float  # всегда положительное, см. правила сдачи (CASE.md)
    raw_value: float  # значение со знаком, до abs() — для отладки и брутфорса evidence
    contributing_txn_ids: list[str]
    covenant_inactive: bool  # True если было springing-условие и оно НЕ сработало (авто-COMPLIANT)


def _breach(raw: float, comparison: str, threshold: float) -> bool:
    return raw < threshold if comparison == "min" else raw > threshold


def evaluate_formula(
    formula: CovenantFormula, resolved: list[ResolvedTxn], facts: ScenarioFacts
) -> EvalResult:
    contributing: list[str] = []

    if formula.shape == "ratio":
        assert formula.numerator is not None and formula.denominator is not None
        num, num_ids = resolve_value(formula.numerator, resolved, facts, formula.period_start, formula.period_end)
        den, den_ids = resolve_value(formula.denominator, resolved, facts, formula.period_start, formula.period_end)
        raw = num / den if den != 0 else float("inf") * (1 if num >= 0 else -1)
        contributing = num_ids + den_ids

    elif formula.shape == "aggregate":
        assert formula.numerator is not None
        raw, contributing = resolve_value(formula.numerator, resolved, facts, formula.period_start, formula.period_end)

    elif formula.shape == "max_of":
        values = [resolve_value(t, resolved, facts, formula.period_start, formula.period_end) for t in formula.terms]
        raw = max(v for v, _ in values) if values else 0.0
        for v, ids in values:
            if v == raw:
                contributing = ids
                break

    elif formula.shape == "net_of_largest":
        assert formula.numerator is not None
        base, base_ids = resolve_value(formula.numerator, resolved, facts, formula.period_start, formula.period_end)
        values = [resolve_value(t, resolved, facts, formula.period_start, formula.period_end) for t in formula.terms]
        largest_val, largest_ids = max(values, key=lambda v: v[0]) if values else (0.0, [])
        raw = base - largest_val
        contributing = base_ids + largest_ids

    else:
        raise ValueError(f"Неизвестная форма ковенанта: {formula.shape}")

    # Springing-условие: comparison="min" на condition значит "ковенант активен, если
    # source >= threshold" (соответствует прочтению "применяется, если ... превышает");
    # "max" — зеркально, "активен, если source <= threshold". Используем тот же _breach:
    # _breach(cond_val, comparison, threshold) == True означает "условие НЕ выполнено".
    covenant_inactive = False
    if formula.condition is not None:
        cond_val, _ = resolve_value(
            formula.condition.source, resolved, facts, formula.period_start, formula.period_end
        )
        covenant_inactive = _breach(cond_val, formula.condition.comparison, formula.condition.threshold)

    if covenant_inactive:
        status = "COMPLIANT"
    else:
        status = "BREACH" if _breach(raw, formula.comparison, formula.threshold) else "COMPLIANT"

    return EvalResult(
        status=status,
        actual=abs(raw) if raw not in (float("inf"), float("-inf")) else 0.0,
        raw_value=raw,
        contributing_txn_ids=contributing,
        covenant_inactive=covenant_inactive,
    )
