"""
LLM-извлечение фактов из аудит-примечаний / KYC-досье / прочих документов пачки
заёмщика в структурированные поправки к леджеру (см. src/covenants/schema.py::
ScenarioFacts).

Два намеренных решения, основанных на разборе публичного датасета вручную:

1. Читаем ВСЕ документы пачки, кроме самого договора (в т.ч. классифицированные как
   DECOY) — классификатор в src/ingest/documents.py неидеален: как минимум один
   реальный факт (недостающая сумма транзакции) в публичном датасете обнаружился внутри
   документа, размеченного как decoy ("казначейская записка"). Не полагаемся на bucket,
   передаём модели весь текст.
2. В датасете встречаются "промежуточные"/черновые меморандумы аудитора, помеченные как
   ПРОЕКТ и явно указывающие, что финальную позицию нужно смотреть в итоговом отчёте;
   если реклассификация из черновика не повторена в финальном отчёте — она отменена.
   Модель явно проинструктирована не путать такие черновики с действующими фактами.

Результат кэшируется в data/processed/scenario_facts.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent.llm_client import generate_structured
from covenants.schema import AllScenarioFacts, ScenarioFacts

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data/processed/scenario_facts.json"


SYSTEM_PROMPT = """\
Ты — аудитор-методолог банка, проверяющий соблюдение кредитных ковенантов. Тебе дают
ВСЕ вспомогательные документы одного заёмщика (примечания аудитора к отчётности, KYC-
досье комплаенса, и иногда посторонние документы, которые могут оказаться нерелевантным
шумом, а могут — нет), кроме самого кредитного договора. Твоя задача — извлечь
конкретные факты, которые меняют результат расчёта ковенантов по бухгалтерскому
леджеру этого заёмщика.

ВАЖНО про черновики: некоторые аудиторские документы явно помечены как ПРОЕКТ /
ПРОМЕЖУТОЧНАЯ ВЕДОМОСТЬ и содержат оговорку, что это не финальная позиция аудитора, а
финальный отчёт (обычно называется "Примечания к финансовой отчётности" или "Отчёт о
выполнении согласованных процедур") может либо повторить, либо отменить вывод
черновика. Правило: если факт (например, реклассификация транзакции) встречается ТОЛЬКО
в черновике и НЕ повторён в финальном документе того же заёмщика — он НЕ действует.
Включи его в ответ с void=true и пояснением в note, но не как обычный факт. Если
финальный документ явно говорит "переклассификаций не требовалось" — это тоже сигнал,
что черновые реклассификации отменены.

ВАЖНО про "decoy"-подобные документы: не игнорируй документы только потому, что они
выглядят как внутренняя переписка, статус-репорт проекта или процедурный документ. Читай
их тоже — среди них изредка встречаются документы с настоящими данными (например,
казначейская записка с суммой транзакции, отсутствующей в выгрузке леджера). Если
документ не содержит ничего relevant — просто не извлекай из него факты, не обязательно
это указывать отдельно.

Извлекай факты СЛЕДУЮЩИХ типов (пропускай тип, если в документах для него ничего нет):

1. reclassifications — переклассификация конкретной транзакции аудитором из одной
   категории в другую. Категории (используй ровно эти коды): REVENUE, OPEX, CAPEX,
   INTEREST, TAX, INSURANCE, RENT, PAYROLL, UTILITIES, MARKETING, MAINTENANCE, ADVISORY,
   FINANCING_INFLOW. Если в тексте назван TXN-ID — укажи его. Если TXN-ID не назван, но
   указан контрагент и сумма — заполни counterparty и amount (положительное число),
   чтобы транзакцию можно было найти по этим признакам отдельно.

2. cutoffs — транзакция, чей период признания аудитор сдвигает за пределы ковенантного
   периода (например, выручка/расход фактически относится к другому году/кварталу) —
   такую транзакцию нужно полностью исключить из расчётов за этот период. Обязательно
   укажи TXN-ID.

3. missing_amounts — транзакция с отсутствующей суммой в выгрузке леджера, для которой
   документ раскрывает фактическую сумму. Укажи TXN-ID и resolved_amount СО ЗНАКОМ
   (расход — отрицательное число, поступление — положительное).

4. related_parties — из KYC-досье: точное название каждой организации-контрагента и её
   доля голосующих прав (%). Также извлеки related_party_threshold_pct — порог доли,
   начиная с которого контрагент признаётся связанной стороной для целей договора (в
   разных досье этот порог РАЗНЫЙ, не предполагай 20% по умолчанию — используй то
   значение, которое явно названо в тексте; если явного порога нет — оставь 20.0).
   ВАЖНО: если для какой-то организации в тексте отдельно указано, что её доля
   удерживается КОСВЕННО через другую организацию, и раскрыта ЭФФЕКТИВНАЯ доля Группы
   в этой цепочке — используй именно эффективную (сквозную) долю для сравнения с
   порогом, а НЕ номинальную долю из основной таблицы (номинальная доля в таком случае
   вводит в заблуждение и не отражает реальный контроль).

5. addbacks — разовые статьи, которые аудитор разрешает прибавить обратно к EBITDA
   ("скорректированная EBITDA"), с суммой и (если есть) TXN-ID.

6. point_in_time_items — балансовые обязательства "по состоянию на дату" (например,
   обязательство по программе выходных пособий), НЕ отражённые отдельной транзакцией.

7. fx_rates / прочие наблюдения, которые не укладываются в поля выше — опиши в notes
   свободным текстом (например, раскрытый аудитором курс конвертации).

Если для какого-то типа фактов в документах ничего нет — верни пустой список, не
выдумывай факты.
"""


def _build_user_prompt(scenario_id: str, borrower_name: str, docs_text: str) -> str:
    return (
        f"scenario_id: {scenario_id}\n"
        f"borrower_name: {borrower_name}\n\n"
        f"Документы пачки (тип и имя файла указаны в заголовке каждого блока):\n\n{docs_text}"
    )


def _load_cache() -> dict[str, dict]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict[str, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_scenario_facts(
    scenario_id: str,
    borrower_name: str,
    docs_text: str,
    *,
    use_cache: bool = True,
) -> ScenarioFacts:
    cache = _load_cache() if use_cache else {}
    if use_cache and scenario_id in cache:
        return ScenarioFacts.model_validate(cache[scenario_id])

    facts = generate_structured(
        SYSTEM_PROMPT,
        _build_user_prompt(scenario_id, borrower_name, docs_text),
        ScenarioFacts,
    )
    facts.scenario_id = scenario_id

    cache[scenario_id] = facts.model_dump()
    _save_cache(cache)
    return facts


def extract_all_scenario_facts(
    scenarios: dict[str, tuple[str, str]],
    *,
    use_cache: bool = True,
) -> dict[str, ScenarioFacts]:
    """scenarios: scenario_id -> (borrower_name, docs_text). Один запрос к LLM на ВСЕ
    отсутствующие в кэше сценарии разом (бесплатный тариф Gemini жёстко ограничен по
    числу запросов в день)."""
    cache = _load_cache() if use_cache else {}
    missing = {sid: v for sid, v in scenarios.items() if not (use_cache and sid in cache)}

    if missing:
        blocks = "\n\n".join(
            f"########## scenario_id: {sid} (borrower_name: {name}) ##########\n{docs_text}"
            for sid, (name, docs_text) in missing.items()
        )
        user_prompt = (
            "Ниже — вспомогательные документы НЕСКОЛЬКИХ заёмщиков, разделённые "
            "заголовками '########## scenario_id: XXX ... ##########'. Для каждого "
            "верни отдельный элемент в scenarios с соответствующим scenario_id.\n\n" + blocks
        )
        result = generate_structured(SYSTEM_PROMPT, user_prompt, AllScenarioFacts)
        for facts in result.scenarios:
            cache[facts.scenario_id] = facts.model_dump()
        _save_cache(cache)

    return {sid: ScenarioFacts.model_validate(cache[sid]) for sid in scenarios}


def bundle_docs_text(bundle) -> str:
    """Конкатенирует текст всех НЕ-договорных документов пачки с заголовками типа —
    договор не передаём (его текст парсится отдельно covenant_parser'ом), а весь
    остальной корпус (включая decoy) передаём модели целиком, см. докстринг модуля."""
    parts = []
    for d in bundle.audit_notes:
        parts.append(f"----- AUDIT_NOTES {d.path.name} -----\n{d.text}")
    for d in bundle.kyc_dossiers:
        parts.append(f"----- KYC {d.path.name} -----\n{d.text}")
    for d in bundle.decoys:
        parts.append(f"----- OTHER(?) {d.path.name} -----\n{d.text}")
    return "\n\n".join(parts)


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
    scenarios: dict[str, tuple[str, str]] = {}
    for sid in only:
        b = bundles[sid]
        text = b.current_agreement.text
        borrower_m = re.search(r"Заёмщик[,:]?\s*([A-Z][\w\s]+(?:JSC|LLP|LLC))", text)
        borrower = borrower_m.group(1).strip() if borrower_m else sid
        scenarios[sid] = (borrower, bundle_docs_text(b))

    all_facts = extract_all_scenario_facts(scenarios)
    for sid in only:
        print(f"=== {sid} ===")
        print(all_facts[sid].model_dump_json(indent=2))
