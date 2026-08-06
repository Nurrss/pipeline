"""
Тонкая обёртка над Gemini API (google-genai) для structured-output вызовов:
на входе system+user промпт и pydantic-модель ответа, на выходе — провалидированный
экземпляр этой модели. temperature=0 и один retry на сетевые/парсинговые сбои —
детерминированность важна, т.к. на публичном датасете мы калибруемся по ground truth,
а на приватном сверяться будет не с чем.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
_client: genai.Client | None = None

T = TypeVar("T", bound=BaseModel)


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY не задан — впишите его в .env (см. .env.example)"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def generate_structured(
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    *,
    model: str | None = None,
    retries: int = 2,
) -> T:
    """Один вызов Gemini с ответом, провалидированным по response_model (pydantic)."""
    client = get_client()
    config = types.GenerateContentConfig(
        temperature=0,
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=response_model,
    )

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.models.generate_content(
                model=model or _MODEL,
                contents=user_prompt,
                config=config,
            )
            if resp.parsed is not None:
                return resp.parsed
            # Иногда SDK не может авто-распарсить (напр. пустой ответ) —
            # пробуем вручную через .text.
            return response_model.model_validate_json(resp.text)
        except Exception as e:  # сетевые сбои, невалидный JSON, rate limit
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
    raise RuntimeError(f"Gemini structured-output вызов не удался после {retries + 1} попыток: {last_err}") from last_err
