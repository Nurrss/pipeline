"""
LLM-парсер Статьи 6 ("Финансовые ковенанты") кредитного договора в структурированную
формулу (см. src/covenants/schema.py::CovenantFormula) — по одной на каждый пункт
(6.1/6.2/6.3).

Формулировки пунктов сильно различаются между заёмщиками (разные метрики, разные
категории в числителе/знаменателе, ratio vs абсолютная сумма, min vs max, иногда
квартальное окно вместо годового, иногда springing-условие). Жёстко закодировать все
варианты нельзя — часть текста мы уже видели в публичном датасете, но приватный будет
использовать другие формулировки/цифры/компании при, предположительно, тех же архетипах
расчёта. Поэтому классификацию текста в фиксированный набор "форм" (shape/ValueKind)
делает LLM, а не мы руками — расчётный слой (src/covenants/calculate.py) уже детерминирован
и просто исполняет то, что вернул парсер.

Результат кэшируется в data/processed/covenant_formulas.json, чтобы не тратить вызовы
API повторно при итерациях на нижних слоях пайплайна.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent.llm_client import generate_structured
from covenants.schema import CovenantFormula, CovenantFormulaSet

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data/processed/covenant_formulas.json"


SYSTEM_PROMPT = """\
Ты — аналитик кредитного отдела банка. Тебе дают текст Статьи 6 ("Финансовые ковенанты")
кредитного договора одного заёмщика на русском (иногда с отдельными терминами на
английском) и просят перевести каждый из трёх пунктов (6.1, 6.2, 6.3) в строго
структурированную формулу для автоматического расчёта по бухгалтерскому леджеру.

У тебя ЕСТЬ фиксированный набор категорий транзакций леджера (уже размечены отдельным
детерминированным классификатором, не переизобретай их):
  REVENUE           — выручка
  OPEX              — "Операционные расходы" КАК САМОСТОЯТЕЛЬНЫЙ ТЕРМИН. Это ОТДЕЛЬНАЯ
                       строка леджера (одна транзакция вида "... operating expenses" /
                       "... servicing and operating costs"), а НЕ сумма аренды+ФОТ+
                       коммуналки и т.п. Используй OPEX только когда текст ковенанта
                       ссылается на "операционные расходы" ОБОБщЁННО, не перечисляя их
                       компоненты.
  CAPEX             — капитальные затраты
  INTEREST          — процентные расходы
  TAX               — налоги
  INSURANCE         — страховые премии
  RENT              — аренда/арендные платежи
  PAYROLL           — расходы на оплату труда / ФОТ
  UTILITIES         — коммунальные расходы
  MARKETING         — маркетинг/реклама
  MAINTENANCE       — обслуживание/ремонт (НЕ то же самое, что OPEX — это точечные
                       статьи вроде инспекций, ремонта, а не общая строка расходов)
  ADVISORY          — консультационные/консалтинговые услуги
  FINANCING_INFLOW  — поступления по финансированию (drawdown по кредитной линии)

ВАЖНОЕ ПРАВИЛО категорий: если пункт ковенанта ЯВНО ПЕРЕЧИСЛЯЕТ конкретные статьи
(например "Расходы на оплату труда и Коммунальные расходы" или "Налоги и Коммунальные
расходы"), используй именно перечисленные категории (PAYROLL+UTILITIES, TAX+UTILITIES
и т.п.), а НЕ OPEX. Категорию OPEX используй, только когда термин "операционные
расходы" упомянут САМ ПО СЕБЕ, без перечисления состава (типичный случай — формула
EBITDA = Выручка минус Операционные расходы).

Каждый пункт нужно классифицировать в одну из четырёх "форм" (shape):
  ratio           — числитель / знаменатель  сравнивается с threshold (threshold_unit=ratio)
  aggregate       — одно значение сравнивается с threshold (обычно threshold_unit=USD)
  max_of          — берётся максимум из нескольких значений (terms), он сравнивается с threshold
  net_of_largest  — из numerator вычитается максимум из terms, результат сравнивается с threshold

comparison:
  "min" — ковенант требует значение "не менее" threshold (нарушение, если фактическое
          значение НИЖЕ threshold)
  "max" — ковенант требует значение "не более"/"не должно превышать" threshold
          (нарушение, если фактическое значение ВЫШЕ threshold)

Каждое значение (numerator/denominator/terms) — это ValueSource с полем kind:
  category_sum              — сумма одной или нескольких категорий выше (поле categories)
  ebitda                    — Выручка минус Операционные расходы (OPEX), без корректировок
  adjusted_ebitda           — EBITDA с разовыми корректировками (addbacks), когда текст
                               явно говорит о "скорректированной EBITDA" и разовых статьях,
                               подлежащих обратному добавлению
  related_party_payments    — платежи связанным/аффилированным сторонам (круг определяется
                               по KYC-досье отдельным этапом, тебе не нужно перечислять
                               контрагентов)
  asset_transfer_subsidiary — капитальные активы, переданные (неограниченным) дочерним
                               организациям — это подмножество CAPEX
  personnel_obligations     — совокупные обязательства по персоналу (ФОТ за период +
                               обязательства по программам выходных пособий/сокращения,
                               раскрытые аудитором)
  group_capex                — капитальные затраты Группы/консолидированные (не сценария) —
                               используй, только если текст явно говорит о Группе/материнской
                               компании, а не о самом заёмщике
  constant                   — фиксированное число (используй крайне редко)

period_start/period_end — YYYY-MM-DD, фактическое окно теста ИЗ ЭТОГО ЖЕ ТЕКСТА пункта
(не выдумывай даты). Обычно это весь ковенантный период ("с 2025-01-01 по 2025-12-31"),
но если пункт ссылается на конкретный квартал (например "за четвёртый квартал периода,
оканчивающегося 2025-12-31"), сузь period_start/period_end до дат ИМЕННО этого квартала
(Q4 2025 = 2025-10-01..2025-12-31, Q1 = 01-01..03-31, Q2 = 04-01..06-30, Q3 = 07-01..09-30).

as_of_date — заполняй, только если тест "точечный" (обязательство "по состоянию на
конкретную дату"), иначе null.

condition — заполняй, ТОЛЬКО если в тексте явно написано, что ограничение "применяется
только при условии/только если" некоторая другая величина превышает/не достигает порога
(springing-тест). Если такого явного условия нет — condition = null.

Верни ровно 3 формулы (6.1, 6.2, 6.3) в поле formulas, каждая с scenario_id, covenant_id,
borrower_name (название заёмщика из текста), raw_text (дословный текст пункта).
"""


def _build_user_prompt(scenario_id: str, borrower_name: str, article6_text: str) -> str:
    return (
        f"scenario_id: {scenario_id}\n"
        f"borrower_name (по умолчанию, если не найдёшь в тексте): {borrower_name}\n\n"
        f"Текст Статьи 6:\n{article6_text}"
    )


def _load_cache() -> dict[str, list[dict]]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict[str, list[dict]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_scenario_covenants(
    scenario_id: str,
    borrower_name: str,
    article6_text: str,
    *,
    use_cache: bool = True,
) -> list[CovenantFormula]:
    cache = _load_cache() if use_cache else {}
    if use_cache and scenario_id in cache:
        return [CovenantFormula.model_validate(f) for f in cache[scenario_id]]

    result = generate_structured(
        SYSTEM_PROMPT,
        _build_user_prompt(scenario_id, borrower_name, article6_text),
        CovenantFormulaSet,
    )
    formulas = result.formulas
    for f in formulas:
        f.scenario_id = scenario_id  # LLM иногда путает регистр/формат — фиксируем сами

    cache[scenario_id] = [f.model_dump() for f in formulas]
    _save_cache(cache)
    return formulas


if __name__ == "__main__":
    import re
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ingest.documents import build_scenario_bundles, load_all_documents
    from ingest.ledger import build_account_to_scenario_map, load_ledger

    ledger = load_ledger(ROOT / "data/raw/master_ledger_2025.csv", warn=False)
    template = json.loads((ROOT / "data/raw/submission_template.json").read_text())
    known = set(template["answers"].keys())
    acc_map = build_account_to_scenario_map(ledger, known)
    docs = load_all_documents(ROOT / "data/raw/documents", set(acc_map.keys()))
    bundles = build_scenario_bundles(docs, acc_map)

    only = sys.argv[1:] if len(sys.argv) > 1 else sorted(bundles)
    for sid in only:
        b = bundles[sid]
        text = b.current_agreement.text
        idxs = [m.start() for m in re.finditer(r"Статья 6\b", text)]
        idx = idxs[-1]
        end = text.find("Статья 7", idx)
        article6 = text[idx:end]

        borrower_m = re.search(r"Заёмщик[,:]?\s*([A-Z][\w\s]+(?:JSC|LLP|LLC))", article6)
        borrower = borrower_m.group(1).strip() if borrower_m else sid

        formulas = parse_scenario_covenants(sid, borrower, article6)
        print(f"=== {sid} ===")
        for f in formulas:
            print(
                f"  {f.covenant_id}: shape={f.shape} comparison={f.comparison} "
                f"threshold={f.threshold}{f.threshold_unit} num={f.numerator} den={f.denominator} "
                f"terms={f.terms} period=[{f.period_start},{f.period_end}] cond={f.condition}"
            )
