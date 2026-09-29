"""LLM 呼叫層。

所有供應商都走 OpenAI 相容介面，所以換供應商只需要換 base_url 與 key。
呼叫失敗（額度用完、服務中斷）時自動換下一個模型或供應商。
"""

import logging
import time

import numpy as np
from openai import OpenAI, OpenAIError, RateLimitError

from . import config

log = logging.getLogger(__name__)
RATE_LIMIT_WAIT = 62  # 秒；免費層的額度以「每分鐘」計算


class NoProviderError(RuntimeError):
    pass


def _client(p: config.Provider) -> OpenAI:
    return OpenAI(api_key=p.api_key, base_url=p.base_url, max_retries=1, timeout=60)


def available_providers() -> list[config.Provider]:
    return [p for p in config.PROVIDERS if p.api_key]


def chat(messages: list[dict], *, temperature: float = 0.2) -> tuple[str, str]:
    """回傳 (回覆文字, 使用的「供應商/模型」)。"""
    errors = []
    for p in available_providers():
        client = _client(p)
        for model in p.chat_models:
            try:
                resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
                return resp.choices[0].message.content or "", f"{p.name}/{model}"
            except OpenAIError as e:
                log.warning("%s/%s 呼叫失敗，改用下一個：%s", p.name, model, str(e)[:200])
                errors.append(f"{p.name}/{model}: {str(e)[:200]}")
    raise NoProviderError("沒有可用的 LLM 供應商。" + (" / ".join(errors) if errors else "請設定 API key。"))


def embedding_provider() -> config.Provider | None:
    return next((p for p in available_providers() if p.embed_model), None)


def embed(texts: list[str], batch_size: int = 50, max_waits: int = 5) -> np.ndarray | None:
    """回傳 L2 正規化後的向量；沒有 embedding 服務時回傳 None（系統退回純關鍵字檢索）。

    免費層每分鐘有請求上限（實測 Gemini 為 100 筆），超過時等一分鐘再繼續，而不是直接放棄。
    """
    p = embedding_provider()
    if p is None:
        return None
    client = _client(p)
    vectors, i, waits = [], 0, 0
    while i < len(texts):
        try:
            resp = client.embeddings.create(model=p.embed_model, input=texts[i : i + batch_size])
            vectors += [d.embedding for d in resp.data]
            i += batch_size
        except RateLimitError as e:
            waits += 1
            if waits > max_waits:
                log.warning("embedding 額度持續不足，退回純關鍵字檢索：%s", str(e)[:200])
                return None
            log.info("embedding 達到每分鐘上限，%d 秒後繼續（第 %d 次等待）", RATE_LIMIT_WAIT, waits)
            time.sleep(RATE_LIMIT_WAIT)
        except OpenAIError as e:
            log.warning("embedding 失敗，退回純關鍵字檢索：%s", str(e)[:200])
            return None
    arr = np.asarray(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)
