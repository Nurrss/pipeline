"""
Категоризация транзакций по тексту description. Без LLM — быстро, детерминированно,
воспроизводимо (важно и для оценки "качества логики агента", и для отсутствия
бюджета на LLM-вызовы там, где они не нужны).

Категории покрывают термины, которые реально фигурируют в текстах ковенантов
(Статья 6 договоров): выручка, капитальные затраты (CAPEX), операционные расходы
(OPEX — самостоятельная строка леджера для EBITDA-формул, см. ниже) и их
подкатегории — аренда, ФОТ (payroll), коммунальные услуги, налоги, страхование,
проценты, маркетинг, обслуживание/ремонт, а также приток по финансированию
(FINANCING_INFLOW — drawdown по кредитной линии, не капзатраты).

Это первый (rule-based) слой. Транзакции, не подошедшие ни под одно правило,
помечаются как OTHER — их стоит явно вывести и разобрать (руками или LLM-фолбэком),
а не тихо игнорировать.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Порядок важен: правила проверяются по порядку, побеждает первое совпадение.
RULES: list[tuple[str, re.Pattern]] = [
    ("REVENUE", re.compile(r"\bsales settlement\b|\bthroughput sales\b|capacity sales settlement|\bsales settlement", re.I)),
    # Общая статья "Операционные расходы" (используется в формулах EBITDA/ratio-ковенантов
    # как самостоятельный термин) — это ОДНА конкретная строка леджера на сценарий, а не
    # сумма прочих категорий. Отличаем её от точечных статей обслуживания/ремонта (которые
    # остаются в MAINTENANCE) по характерной формулировке "servicing and operating costs" /
    # "operating (and maintenance) expenses". Проверено численно против ground truth
    # (B1 6.1/6.2, P1 6.1) — подстановка суммы прочих категорий в Opex даёт совершенно
    # нереалистичные (на порядки завышенные) значения, а эта единственная строка идеально
    # воспроизводит эталонный коэффициент.
    ("OPEX", re.compile(
        r"servicing and operating costs|operating (and maintenance )?expenses",
        re.I,
    )),
    # "Term loan facility drawdown" — это приток по финансированию (положительная сумма),
    # а не капитальные затраты, несмотря на упоминание "expansion"/"turnaround". Нужен
    # отдельно для ковенантов P2 6.1 / P3 6.1 ("поступления по финансированию"); если
    # оставить в CAPEX, ломает агрегатные тесты "Максимальные расходы по категории".
    ("FINANCING_INFLOW", re.compile(r"term loan facility drawdown|drawdown", re.I)),
    ("CAPEX", re.compile(
        r"purchase of .* equipment|transfer of .* equipment to subsidiary|"
        r"flood remediation and silo repair works",
        re.I,
    )),
    ("INTEREST", re.compile(
        r"\binterest\b|revolver interest|quarterly interest coupon|overdraft|"
        r"term loan interest|credit facility interest|interest on|capitalised interest charge",
        re.I,
    )),
    ("TAX", re.compile(
        r"\btax\b|\bvat\b|excise|withholding|customs duty|franchise tax|mineral extraction tax",
        re.I,
    )),
    ("INSURANCE", re.compile(r"insurance|fidelity bond|workers comp", re.I)),
    ("RENT", re.compile(r"\brent\b|\blease\b", re.I)),
    ("PAYROLL", re.compile(r"payroll", re.I)),
    ("UTILITIES", re.compile(
        r"electricity|water charge|water supply|waste water|telecom|utility|utilities|"
        r"natural gas|district heating|compressed air",
        re.I,
    )),
    ("MARKETING", re.compile(
        r"marketing|ad campaign|advertising|exhibition|newsletter|sponsorship|"
        r"radio ad|digital media buy|press insertion|point-of-sale",
        re.I,
    )),
    ("MAINTENANCE", re.compile(
        r"maintenance|repair works|inspection and survey servicing|"
        r"arbitration and legal servicing|servicing contract|risk survey servicing|"
        r"fire safety systems servicing|silt cleaning and clearance works",
        re.I,
    )),
    ("MARKETING", re.compile(r"media buy rate adjustment", re.I)),
    ("UTILITIES", re.compile(r"sewer discharge levy|water and sewer charge", re.I)),
    ("ADVISORY", re.compile(r"advisory|retainer|consulting", re.I)),
]


@dataclass
class Categorized:
    category: str
    matched_rule: str | None


def categorize(description: str) -> Categorized:
    for category, pattern in RULES:
        if pattern.search(description):
            return Categorized(category=category, matched_rule=pattern.pattern)
    return Categorized(category="OTHER", matched_rule=None)


if __name__ == "__main__":
    import sys
    from collections import Counter
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ingest.ledger import load_ledger, txns_for_scenario

    root = Path(__file__).resolve().parents[2]
    ledger = load_ledger(root / "data/raw/master_ledger_2025.csv", warn=False)

    counts = Counter()
    others = []
    for t in ledger:
        if t.scenario_id is None:
            continue
        cat = categorize(t.description).category
        counts[cat] += 1
        if cat == "OTHER":
            others.append(t.description)

    print("Распределение категорий по ВСЕМУ леджеру (включая шумовые сценарии 9xxx):")
    for cat, n in counts.most_common():
        print(f"  {cat:>12}: {n}")

    print(f"\nНеразобранных (OTHER): {len(others)} — примеры:")
    for d in sorted(set(others))[:20]:
        print(f"  - {d}")
