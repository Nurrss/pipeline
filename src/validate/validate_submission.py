"""
Проверяет submission.json на структурное соответствие submission_template.json:
- те же верхнеуровневые поля;
- те же scenario_id и те же номера ковенантов, ничего не добавлено/удалено/переименовано;
- у каждой ячейки ровно поля status/actual/evidence_txn_id;
- status - ровно "COMPLIANT" или "BREACH" (не None на выходе, иначе 0 баллов по правилам);
- actual - число (int/float), положительное;
- evidence_txn_id - строка или None.

Использование:
    python3 src/validate/validate_submission.py [путь_к_submission.json]
Без аргумента ищет data/processed/submission.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = ROOT / "data/raw/submission_template.json"


def validate(submission: dict, template: dict) -> list[str]:
    errors: list[str] = []

    for field in ("team", "contact_email", "model"):
        if field not in submission:
            errors.append(f"Отсутствует верхнеуровневое поле '{field}'")
        elif not submission[field]:
            errors.append(f"Поле '{field}' пустое — заполните перед отправкой")

    if "answers" not in submission:
        errors.append("Отсутствует ключ 'answers' — дальнейшая проверка невозможна")
        return errors

    tmpl_scenarios = set(template["answers"].keys())
    sub_scenarios = set(submission["answers"].keys())

    for missing in tmpl_scenarios - sub_scenarios:
        errors.append(f"Отсутствует scenario_id '{missing}'")
    for extra in sub_scenarios - tmpl_scenarios:
        errors.append(f"Лишний scenario_id '{extra}' — в шаблоне такого нет")

    for sid in tmpl_scenarios & sub_scenarios:
        tmpl_cells = set(template["answers"][sid].keys())
        sub_cells = set(submission["answers"][sid].keys())
        for missing in tmpl_cells - sub_cells:
            errors.append(f"[{sid}] Отсутствует ковенант '{missing}'")
        for extra in sub_cells - tmpl_cells:
            errors.append(f"[{sid}] Лишний ковенант '{extra}'")

        for cov in tmpl_cells & sub_cells:
            cell = submission["answers"][sid][cov]
            prefix = f"[{sid}/{cov}]"

            if not isinstance(cell, dict):
                errors.append(f"{prefix} ячейка должна быть объектом")
                continue

            expected_keys = {"status", "actual", "evidence_txn_id"}
            actual_keys = set(cell.keys())
            if actual_keys != expected_keys:
                errors.append(
                    f"{prefix} неверный набор ключей ячейки: {sorted(actual_keys)}, ожидалось {sorted(expected_keys)}"
                )

            status = cell.get("status")
            if status not in ("COMPLIANT", "BREACH"):
                errors.append(f"{prefix} status должен быть 'COMPLIANT' или 'BREACH', получено: {status!r}")

            actual = cell.get("actual")
            if actual is None:
                errors.append(f"{prefix} actual не заполнен (null) — 0 баллов за это поле")
            elif not isinstance(actual, (int, float)) or isinstance(actual, bool):
                errors.append(f"{prefix} actual должен быть числом, получено: {actual!r}")
            elif actual < 0:
                errors.append(f"{prefix} actual должен быть положительным, получено: {actual!r}")

            ev = cell.get("evidence_txn_id")
            if ev is not None and not isinstance(ev, str):
                errors.append(f"{prefix} evidence_txn_id должен быть строкой или null, получено: {ev!r}")

    return errors


def main() -> int:
    sub_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/processed/submission.json"
    if not sub_path.exists():
        print(f"Файл не найден: {sub_path}")
        return 2

    try:
        submission = json.loads(sub_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Повреждённый JSON: {e}")
        return 2

    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    errors = validate(submission, template)

    if not errors:
        print(f"OK: {sub_path} структурно валиден ({len(template['answers'])} сценариев).")
        return 0

    print(f"Найдено {len(errors)} проблем(ы) в {sub_path}:")
    for e in errors:
        print(f"  - {e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
