"""
Структуры данных для двух LLM-этапов пайплайна:

1. covenant_parser: текст Статьи 6 договора -> список CovenantFormula (по одной на
   пункт 6.1/6.2/6.3) — машиночитаемое представление того, ЧТО и КАК считать.
2. fact_extractor: аудит-примечания + KYC-досье (+ прочие документы пачки, включая
   "decoy" — среди них попадаются настоящие факты, см. src/ingest/documents.py) ->
   ScenarioFacts — конкретные поправки к леджеру для этого заёмщика.

Пайплайн намеренно не пытается быть полностью общим DSL: вместо этого — фиксированный
небольшой набор "форм" ковенанта (ValueKind/shape), выведенный эмпирически из анализа
всех 36 ячеек публичного датасета и проверенный численно против ground truth
(см. историю коммитов). LLM классифицирует текст в эти формы и извлекает параметры —
не изобретает произвольную арифметику.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Категории транзакций — см. src/covenants/categorize.py (RULES).
Category = Literal[
    "REVENUE", "OPEX", "CAPEX", "INTEREST", "TAX", "INSURANCE", "RENT",
    "PAYROLL", "UTILITIES", "MARKETING", "MAINTENANCE", "ADVISORY",
    "FINANCING_INFLOW", "OTHER",
]

ValueKind = Literal[
    "category_sum",              # сумма (нетто, по знаку) одной или нескольких категорий
    "ebitda",                    # REVENUE - OPEX
    "adjusted_ebitda",           # EBITDA + разовые статьи (ScenarioFacts.addbacks)
    "related_party_payments",    # платежи контрагентам из ScenarioFacts.related_parties
    "asset_transfer_subsidiary",  # подмножество CAPEX с описанием "transfer ... to subsidiary"
    "personnel_obligations",     # PAYROLL + разовое обязательство (ScenarioFacts.point_in_time_items)
    "group_capex",                # консолидированные капзатраты группы; обычно не раскрываются
                                   # документами -> детерминированный fallback на CAPEX сценария
    "constant",
]


class ValueSource(BaseModel):
    kind: ValueKind
    categories: list[Category] = Field(default_factory=list)  # для category_sum
    constant_value: float | None = None                        # для constant
    description: str = ""  # человекочитаемое пояснение — для отладки/логов, не для расчёта


class Condition(BaseModel):
    """Springing-условие: ковенант ограничивает поведение, только если source
    относительно threshold удовлетворяет comparison. Если условие не выполнено,
    ковенант считается COMPLIANT автоматически, но actual всё равно = вычисленное
    значение формулы (см. правило в CASE.md про 'применяется только при срабатывании
    условия')."""

    source: ValueSource
    comparison: Literal["min", "max"]
    threshold: float


class CovenantFormula(BaseModel):
    scenario_id: str
    covenant_id: str  # "6.1" / "6.2" / "6.3"
    borrower_name: str

    shape: Literal["ratio", "aggregate", "max_of", "net_of_largest"]
    # ratio:           numerator / denominator  op  threshold
    # aggregate:       numerator                op  threshold
    # max_of:          max(terms)               op  threshold
    # net_of_largest:  numerator - max(terms)    op  threshold

    comparison: Literal["min", "max"]
    # "min" = ковенант требует "не менее" threshold -> BREACH если actual < threshold
    # "max" = ковенант требует "не более/не превышал" threshold -> BREACH если actual > threshold

    threshold: float
    threshold_unit: Literal["USD", "ratio"]

    numerator: ValueSource | None = None
    denominator: ValueSource | None = None  # только для shape="ratio"
    terms: list[ValueSource] = Field(default_factory=list)  # для max_of / net_of_largest

    period_start: str  # ISO YYYY-MM-DD — фактическое окно теста (для квартальных
    period_end: str    # ковенантов уже сужено до квартала, не всегда календарный год)
    as_of_date: str | None = None  # для point-in-time тестов (обязательства "по состоянию на")

    condition: Condition | None = None  # springing-тест, см. Condition

    raw_text: str = ""  # исходный текст пункта — для отладки/аудита


class CovenantFormulaSet(BaseModel):
    """Обёртка для structured-output ответа LLM — три формулы одного сценария."""

    formulas: list[CovenantFormula]


class ScenarioCovenants(BaseModel):
    scenario_id: str
    formulas: list[CovenantFormula]


class AllCovenantFormulas(BaseModel):
    """Обёртка для батч-вызова: формулы сразу по всем сценариям за один запрос к LLM
    (бесплатный тариф Gemini жёстко ограничен по числу запросов в день — экономим их)."""

    scenarios: list[ScenarioCovenants]


# ---------------------------------------------------------------------------
# Факты из аудит-примечаний / KYC-досье
# ---------------------------------------------------------------------------


class Reclassification(BaseModel):
    """Переклассификация транзакции аудитором. Если txn_id не назван явно в тексте,
    матчим по counterparty+amount против леджера сценария. void=True — значит факт
    встретился только в промежуточном/черновом меморандуме и НЕ подтверждён финальным
    отчётом (в датасете такие черновики явно помечены как superseded) -> не применять."""

    txn_id: str | None = None
    counterparty: str | None = None
    amount: float | None = None  # абсолютная величина, для матчинга по сумме
    from_category: Category | None = None
    to_category: Category | None = None
    void: bool = False
    note: str = ""


class CutoffAdjustment(BaseModel):
    """Транзакция, чей период признания сдвинут аудитором за пределы ковенантного
    периода (2025 год) -> полностью исключить из всех расчётов за этот период."""

    txn_id: str
    note: str = ""


class MissingAmount(BaseModel):
    """Сумма транзакции отсутствует в выгрузке леджера (amount=null), но раскрыта
    в документе. resolved_amount — со знаком (расход отрицательный)."""

    txn_id: str
    resolved_amount: float
    note: str = ""


class RelatedParty(BaseModel):
    name: str  # точное название контрагента, как в KYC-досье и в леджере
    ownership_pct: float


class Addback(BaseModel):
    """Разовая статья, которую аудитор разрешает прибавить обратно для adjusted EBITDA."""

    description: str
    amount: float
    txn_id: str | None = None


class PointInTimeItem(BaseModel):
    """Балансовое обязательство 'по состоянию на дату', не отражённое отдельной
    транзакцией в леджере (например, обязательство по программе выходных пособий)."""

    description: str
    amount: float
    as_of_date: str | None = None


class ScenarioFacts(BaseModel):
    scenario_id: str
    reclassifications: list[Reclassification] = Field(default_factory=list)
    cutoffs: list[CutoffAdjustment] = Field(default_factory=list)
    missing_amounts: list[MissingAmount] = Field(default_factory=list)
    related_parties: list[RelatedParty] = Field(default_factory=list)
    related_party_threshold_pct: float = 20.0
    addbacks: list[Addback] = Field(default_factory=list)
    point_in_time_items: list[PointInTimeItem] = Field(default_factory=list)
    notes: str = ""  # свободный текст — прочие наблюдения, не влияющие на расчёт напрямую


class AllScenarioFacts(BaseModel):
    """Обёртка для батч-вызова: факты сразу по всем сценариям за один запрос к LLM."""

    scenarios: list[ScenarioFacts]
