"""LLM 呼叫層。

所有供應商都走 OpenAI 相容介面，所以換供應商只需要換 base_url 與 key。
呼叫失敗（額度用完、服務中斷）時自動換下一個模型或供應商；
全部失敗時，把錯誤分類成使用者看得懂的原因（ApiIssue），而不是丟出原始錯誤碼。
"""

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
import numpy as np
from openai import APITimeoutError, OpenAI, OpenAIError, RateLimitError

from . import config
from .progress import NullProgress, Progress

log = logging.getLogger(__name__)
RATE_LIMIT_WAIT = 62  # 秒；免費層的額度以「每分鐘」計算


@dataclass
class ApiIssue:
    """API 無法使用的原因，用來在畫面上給出具體說明。"""

    kind: str  # rate_minute / rate_day / overloaded / timeout / unavailable / no_key
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
        if self.kind == "timeout":
            return "⚠️ Google AI 長時間沒有回應（非本系統錯誤），請稍後再試。"
        if self.kind == "no_key":
            return "⚠️ 尚未設定 LLM API key，問答功能無法使用。"
        return "⚠️ Google AI 服務暫時無法使用（非本系統錯誤），請稍後再試。"


# 多種失敗同時發生時，優先顯示使用者能採取行動的原因
ISSUE_PRIORITY = ["rate_minute", "rate_day", "overloaded", "timeout", "unavailable"]
SHORT_REASON = {"rate_minute": "每分鐘額度已滿", "rate_day": "今日額度已用完", "overloaded": "伺服器忙碌",
                "timeout": "無回應，逾時", "unavailable": "暫時無法使用"}


class StreamTimeout(Exception):
    """串流生成超過總時間上限。"""


class EmptyResponse(Exception):
    """模型回應了，但內容是空的。"""


def classify_error(exc: Exception) -> ApiIssue:
    if isinstance(exc, (APITimeoutError, httpx.TimeoutException, StreamTimeout)):
        return ApiIssue("timeout")
    if isinstance(exc, EmptyResponse):
        return ApiIssue("unavailable")
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
    def __init__(self, issue: ApiIssue, failures: list[tuple[str, ApiIssue]] | None = None):
        super().__init__(issue.message)
        self.issue = issue
        self.failures = failures or []


@dataclass
class ChatResult:
    text: str
    model: str            # 實際回應的「供應商/模型」
    degraded: bool        # 是否不是首選模型（主模型失敗或被跳過，改用備用模型）
    failures: list[tuple[str, ApiIssue]] = field(default_factory=list)  # 這次失敗的模型與原因


# ── 逾時設定（2026-10-02 實測）──────────────────────────────
# 回答用「串流」：模型一邊生成一邊回傳，因此可以把兩件事分開判斷：
#   1. 伺服器有沒有在處理 → 看「第一個字」多久回來
#   2. 回答很長、還在生成 → 只要片段持續回來就不算逾時
# 非串流時，回應要等整段回答寫完才回來，「沒有回應」與「回答比較長」無法區分。
#
# 實測（同一題）：完整版 3.5／3.6 會先在伺服器內部思考，第一個字 9.5~11.9 秒才出現，之後片段間隔 ≤ 0.3 秒；
# 輕量版第一個字約 0.9 秒。所以等第一個字不能設太短，否則正常的模型會被誤判逾時。
FIRST_TOKEN_TIMEOUT = 25   # 等第一個字、以及片段之間最多等多久（httpx 的 read 逾時，每次讀取都適用）
STREAM_TOTAL_LIMIT = 90    # 整段生成的總時間上限，只是防止無限卡住的保險
REWRITE_TIMEOUT = 15       # 改寫不串流：只輸出一句話，輕量版約 1 秒
# 不在同一個模型上重試：本來就有備用模型可接手，重試只會讓使用者多等
CHAT_TIMEOUT = REWRITE_TIMEOUT


def _client(p: config.Provider, timeout: float | httpx.Timeout = CHAT_TIMEOUT) -> OpenAI:
    return OpenAI(api_key=p.api_key, base_url=p.base_url, max_retries=0, timeout=timeout)


def _stream(p: config.Provider, model: str, messages: list[dict], temperature: float, progress: Progress,
            label: str, on_text: Callable[[str], None]) -> str:
    """串流取得回答。一次嘗試在畫面上分成「等待回應」與「生成中」兩個階段。"""
    progress.stage(f"{label}（{model}）：等待回應")
    client = _client(p, httpx.Timeout(FIRST_TOKEN_TIMEOUT, connect=10))
    start, text = time.time(), ""
    stream = client.chat.completions.create(model=model, messages=messages, temperature=temperature, stream=True)
    try:
        for chunk in stream:
            if time.time() - start > STREAM_TOTAL_LIMIT:
                raise StreamTimeout()
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if not text:
                    progress.stage(f"{label}（{model}）：生成中")
                text += delta
                on_text(text)
    finally:
        stream.close()
    if not text:
        raise EmptyResponse()
    return text


def available_providers() -> list[config.Provider]:
    return [p for p in config.PROVIDERS if p.api_key]


def model_name(model: str) -> str:
    """畫面顯示用的短名稱：gemini/gemini-flash-latest → gemini-flash-latest。"""
    return model.split("/", 1)[-1]


def chat(messages: list[dict], *, temperature: float = 0.2, purpose: str = "answer",
         skip: set[str] | frozenset[str] = frozenset(), progress: Progress | None = None,
         stage_label: str = "產生回答", on_text: Callable[[str], None] | None = None) -> ChatResult:
    """依序嘗試各模型。

    - on_text：有提供時用串流，每收到新片段就以「目前累積的全文」呼叫一次；
      某個模型生成到一半失敗時，會以空字串呼叫一次，讓畫面清掉半段文字，再由下一個模型重來

    - purpose="rewrite" 時優先用輕量模型：改寫是簡單任務，而且輕量模型的免費額度與主模型分開計算
    - skip：呼叫端記得最近失敗的模型（只存在該使用者的 session），先跳過，避免每題都重新等一次逾時；
      全部都在 skip 裡時就不跳過，寧可再試一次也不要直接放棄
    - 每試一個模型就開始一個新的計時階段，畫面能看到「主模型逾時 → 改用備用模型」的過程
    """
    progress = progress or NullProgress()
    candidates = []
    for p in available_providers():
        models = p.chat_models[::-1] if purpose == "rewrite" else p.chat_models
        candidates += [(p, m) for m in models]
    if not candidates:
        raise NoProviderError(ApiIssue("no_key"))
    preferred = f"{candidates[0][0].name}/{candidates[0][1]}"
    usable = [(p, m) for p, m in candidates if f"{p.name}/{m}" not in skip] or candidates

    failures: list[tuple[str, ApiIssue]] = []
    for p, model in usable:
        full = f"{p.name}/{model}"
        try:
            if on_text is not None:
                text = _stream(p, model, messages, temperature, progress, stage_label, on_text)
            else:
                progress.stage(f"{stage_label}（{model}）")
                resp = _client(p).chat.completions.create(model=model, messages=messages, temperature=temperature)
                text = resp.choices[0].message.content or ""
            return ChatResult(text, full, degraded=full != preferred, failures=failures)
        except (OpenAIError, httpx.HTTPError, StreamTimeout, EmptyResponse) as e:
            log.warning("%s 呼叫失敗，改用下一個：%s", full, str(e)[:200])
            issue = classify_error(e)
            failures.append((full, issue))
            progress.fail(SHORT_REASON.get(issue.kind, "失敗"))
            if on_text is not None:
                on_text("")  # 清掉這個模型生成到一半的文字
    issues = [i for _, i in failures]
    issue = min(issues, key=lambda x: ISSUE_PRIORITY.index(x.kind) if x.kind in ISSUE_PRIORITY else 99)
    if issue.kind == "rate_minute":  # 多個模型都要等時，取最長的等待時間才保險
        waits = [x.retry_seconds for x in issues if x.kind == "rate_minute" and x.retry_seconds]
        issue.retry_seconds = max(waits) if waits else None
    raise NoProviderError(issue, failures)


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
