"""LLM 呼叫層。

所有供應商都走 OpenAI 相容介面，所以換供應商只需要換 base_url 與 key。
呼叫失敗（額度用完、服務中斷）時自動換下一個模型或供應商；
全部失敗時，把錯誤分類成使用者看得懂的原因（ApiIssue），而不是丟出原始錯誤碼。
"""

import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np
from openai import OpenAI, OpenAIError, RateLimitError

from . import config

log = logging.getLogger(__name__)
RATE_LIMIT_WAIT = 62  # 秒；免費層的額度以「每分鐘」計算


@dataclass
class ApiIssue:
    """API 無法使用的原因，用來在畫面上給出具體說明。"""

    kind: str  # rate_minute / rate_day / overloaded / unavailable / no_key
    retry_seconds: int | None = None

    @property
    def message(self) -> str:
        if self.kind == "rate_minute":
            wait = f"請約 {self.retry_seconds} 秒後再試" if self.retry_seconds else "請稍候一分鐘再試"
            return f"⏳ 已達 Google Gemini 免費方案的「每分鐘」使用上限，{wait}。"
        if self.kind == "rate_day":
            return ("🚫 今日的 Google Gemini 免費額度已用完。額度依太平洋時間午夜重置"
                    "（約台灣時間下午 3~4 點），請屆時再試。")
        if self.kind == "overloaded":
            return "⚠️ Google AI 伺服器目前忙碌（非本系統錯誤），請稍後再試。"
        if self.kind == "no_key":
            return "⚠️ 尚未設定 LLM API key，問答功能無法使用。"
        return "⚠️ Google AI 服務暫時無法使用（非本系統錯誤），請稍後再試。"


# 多種失敗同時發生時，優先顯示使用者能採取行動的原因
ISSUE_PRIORITY = ["rate_minute", "rate_day", "overloaded", "unavailable"]


def classify_error(exc: Exception) -> ApiIssue:
    text = str(exc)
    if isinstance(exc, RateLimitError) or "429" in text or "RESOURCE_EXHAUSTED" in text:
        if re.search(r"PerDay|per_day|perday", text, re.I):
            return ApiIssue("rate_day")
        m = re.search(r"retry in ([\d.]+)\s*s", text) or re.search(r"retryDelay'?\"?:\s*'?\"?(\d+)s", text)
        return ApiIssue("rate_minute", int(float(m.group(1))) + 1 if m else None)
    if "503" in text or "UNAVAILABLE" in text or "high demand" in text or "overloaded" in text.lower():
        return ApiIssue("overloaded")
    return ApiIssue("unavailable")


class NoProviderError(RuntimeError):
    def __init__(self, issue: ApiIssue, detail: str = ""):
        super().__init__(issue.message + (f"（{detail}）" if detail else ""))
        self.issue = issue


@dataclass
class ChatResult:
    text: str
    model: str            # 實際回應的「供應商/模型」
    degraded: bool        # 是否不是首選模型（主模型失敗後改用備用模型）
    notes: list[str] = field(default_factory=list)


def _client(p: config.Provider) -> OpenAI:
    return OpenAI(api_key=p.api_key, base_url=p.base_url, max_retries=1, timeout=60)


def available_providers() -> list[config.Provider]:
    return [p for p in config.PROVIDERS if p.api_key]


def chat(messages: list[dict], *, temperature: float = 0.2, purpose: str = "answer") -> ChatResult:
    """依序嘗試各模型。purpose="rewrite" 時優先用輕量模型：
    改寫是簡單任務，而且輕量模型的免費額度與主模型分開計算，不會擠掉回答用的額度。"""
    candidates = []
    for p in available_providers():
        models = p.chat_models[::-1] if purpose == "rewrite" else p.chat_models
        candidates += [(p, m) for m in models]
    if not candidates:
        raise NoProviderError(ApiIssue("no_key"))

    issues = []
    for i, (p, model) in enumerate(candidates):
        try:
            resp = _client(p).chat.completions.create(model=model, messages=messages, temperature=temperature)
            return ChatResult(resp.choices[0].message.content or "", f"{p.name}/{model}", degraded=i > 0)
        except OpenAIError as e:
            log.warning("%s/%s 呼叫失敗，改用下一個：%s", p.name, model, str(e)[:200])
            issues.append(classify_error(e))
    issue = min(issues, key=lambda x: ISSUE_PRIORITY.index(x.kind) if x.kind in ISSUE_PRIORITY else 99)
    if issue.kind == "rate_minute":  # 多個模型都要等時，取最長的等待時間才保險
        waits = [x.retry_seconds for x in issues if x.kind == "rate_minute" and x.retry_seconds]
        issue.retry_seconds = max(waits) if waits else None
    raise NoProviderError(issue)


def embedding_provider() -> config.Provider | None:
    return next((p for p in available_providers() if p.embed_model), None)


def embed(texts: list[str], batch_size: int = 50, max_waits: int = 5) -> np.ndarray | None:
    """回傳 L2 正規化後的向量；沒有 embedding 服務時回傳 None（系統退回純關鍵字檢索）。

    建立索引時（max_waits > 0）：免費層每分鐘有請求上限（實測 Gemini 為 100 筆），超過時等一分鐘再繼續。
    使用者提問時（max_waits = 0）：不等待，直接退回關鍵字檢索，避免畫面卡住一分鐘。
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
                log.warning("embedding 額度不足，退回純關鍵字檢索：%s", str(e)[:200])
                return None
            log.info("embedding 達到每分鐘上限，%d 秒後繼續（第 %d 次等待）", RATE_LIMIT_WAIT, waits)
            time.sleep(RATE_LIMIT_WAIT)
        except OpenAIError as e:
            log.warning("embedding 失敗，退回純關鍵字檢索：%s", str(e)[:200])
            return None
    arr = np.asarray(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)
