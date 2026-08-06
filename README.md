# Halyk AI Challenge — агент проверки ковенантов

AI-агент, который читает корпоративные кредитные документы (PDF) и реестр транзакций,
определяет для каждого ковенанта каждого заёмщика: `status` (COMPLIANT/BREACH), `actual`
(фактическое значение показателя), `evidence_txn_id` (транзакция-доказательство, если есть).

## Структура проекта

```
data/
  raw/            # входные данные — В GIT НЕ ПОПАДАЮТ (см. .gitignore)
    master_ledger_2025.csv
    documents/            # PDF-документы
    submission_template.json
    ground_truth_PUBLIC_ONLY.json   # только для публичного датасета, для калибровки
  processed/       # кэши извлечённых фактов, промежуточные артефакты — тоже не в git
src/
  ingest/          # парсинг CSV и PDF
  linking/         # сопоставление account_id <-> scenario_id, документ <-> заёмщик
  covenants/        # логика расчёта по каждому пункту ковенанта
  agent/            # промпты и вызовы LLM
  validate/          # validate_submission.py, score.py
tests/
submission.json      # финальный файл для сдачи (генерируется, не редактируется руками)
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # вписать туда GEMINI_API_KEY / GROQ_API_KEY
```

## Данные

Датасет (`master_ledger_2025.csv`, `documents/`, `submission_template.json`,
`ground_truth_PUBLIC_ONLY.json` для публичного этапа) кладём в `data/raw/` —
эта папка в `.gitignore`, каждый член команды раскладывает файлы у себя локально
(файлы передаём не через git, а напрямую — архивом/диском).

## Запуск

```bash
# 1. Проверить связку account_id <-> scenario_id
python3 src/ingest/ledger.py

# 2. (когда будет готов весь пайплайн) сгенерировать submission.json
python3 -m src.agent.run --out data/processed/submission.json

# 3. Проверить структуру перед сдачей
python3 src/validate/validate_submission.py data/processed/submission.json

# 4. На ПУБЛИЧНОМ датасете — свериться с ground truth, чтобы понять качество
python3 src/validate/score.py data/processed/submission.json -v
```

`score.py` работает только пока есть `ground_truth_PUBLIC_ONLY.json` — на приватном
датасете эталонных ответов нет, там просто генерируем `submission.json` и сдаём.

## Как устроена логика (кратко)

1. **Ledger**: читаем `master_ledger_2025.csv`, вычленяем `scenario_id` из префикса
   `txn_id`, строим карту `account_id -> scenario_id`. Датасет специально "грязный" —
   есть строки без суммы (`amount` пустой), они не отбрасываются, а помечаются как
   требующие уточнения из документов.
2. **Документы**: каждый PDF в `documents/` парсится, классифицируется (тип документа,
   какой `account_id`/компания в нём фигурирует, дата), включая отличение актуальной
   версии от устаревшей, если есть несколько версий одного документа.
3. **Ковенанты**: для каждого пункта (6.1/6.2/6.3) находим текст условия в договоре
   нужного заёмщика, вычисляем `actual` точным расчётом по леджеру (не угадыванием
   LLM), сравниваем с лимитом -> `status`.
3. **Evidence**: для ячеек, где в ключе не `null`, ищем единственную транзакцию, чьё
   исключение меняет вердикт (не просто крупную сумму).
4. **Сборка**: заполняем `submission_template.json`, валидируем, сдаём.

## Git-воркфлоу команды

Работаем через отдельные ветки, чтобы не конфликтовать в общем коде:

```bash
git clone <url-репозитория>
cd halyk-agent
git checkout -b feature/имя-задачи      # например feature/pdf-parsing
# ... работаем, коммитим ...
git add -A
git commit -m "Понятное описание изменения"
git push -u origin feature/имя-задачи
# затем открываем Pull Request в main на GitHub, второй участник ревьюит и мержит
```

Никогда не коммитим:
- реальные датасеты (`data/raw/*`) — они в `.gitignore`;
- API-ключи (`.env`) — тоже в `.gitignore`, используем `.env.example` как образец;
- `data/processed/*` (кэши, могут быть тяжёлыми/содержать данные).

Ноутбуки участников не обязаны быть включены — весь код и история в GitHub,
любой может клонировать репозиторий и продолжить работу с любой машины.
