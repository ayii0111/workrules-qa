"""輸入防護：長度限制、個資遮蔽、單一 session 的提問頻率。

都是程式規則，不呼叫 LLM：便宜、穩定、結果可預期。
"""

import re
import time
from dataclasses import dataclass

MAX_CHARS = 500
# 本分頁每分鐘最多提問次數。Gemini 免費版主模型約每分鐘 10 次，
# 每次提問最多呼叫主模型 1 次，留一些餘裕給同時使用的其他人
MAX_QUESTIONS_PER_MINUTE = 6

# 不用 \b：Python 會把中文字也當成「單字字元」，「身分證A123456789」這種中英文相連的寫法會比對不到
_L, _R = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"
PII_PATTERNS = [
    ("身分證字號", re.compile(_L + r"[A-Za-z][12]\d{8}" + _R)),
    ("居留證號", re.compile(_L + r"[A-Za-z][A-Da-d89]\d{8}" + _R)),
    ("手機號碼", re.compile(_L + r"09\d{2}-?\d{3}-?\d{3}" + _R)),
    ("Email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")),
]


@dataclass
class Masked:
    text: str
    found: list[str]  # 被遮蔽的個資類型，用來提示使用者


def mask_pii(text: str) -> Masked:
    found = []
    for label, pattern in PII_PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            found.append(label)
    return Masked(text, found)


def too_long(text: str) -> bool:
    return len(text) > MAX_CHARS


def wait_seconds(timestamps: list[float], now: float | None = None,
                 limit: int = MAX_QUESTIONS_PER_MINUTE, window: float = 60.0) -> int:
    """最近 window 秒內已提問 limit 次時，回傳還要等幾秒；不用等則回傳 0。

    timestamps 存在該使用者自己的 session 裡，不與其他使用者共用。
    """
    now = time.time() if now is None else now
    recent = [t for t in timestamps if now - t < window]
    if len(recent) < limit:
        return 0
    return int(window - (now - min(recent))) + 1


def record(timestamps: list[float], now: float | None = None, window: float = 60.0) -> list[float]:
    """記錄一次提問，並丟掉過期的時間點，避免清單無限增長。"""
    now = time.time() if now is None else now
    return [t for t in timestamps if now - t < window] + [now]
