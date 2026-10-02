"""集中管理設定：路徑、收錄的法規、LLM 供應商。

API key 一律從環境變數（或 .env / Streamlit secrets）讀取，不寫進程式碼。
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DB_PATH = Path(os.getenv("WR_DB_PATH", ROOT / "data" / "workrules.db"))
HANDBOOK_PATH = ROOT / "data" / "handbook.md"
INTERPRETATIONS_PATH = ROOT / "data" / "interpretations.json"  # 精選的勞動部函釋文號清單
GUIDANCE_PATH = ROOT / "data" / "guidance.json"  # 精選的主管機關說明文章清單

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128 Safari/537.36"
)
HTTP_TIMEOUT = 30
POLITE_DELAY = 1.0  # 對同一網站連續請求之間的間隔（秒）

# 收錄的法規：全國法規資料庫的法規代碼（pcode）
LAWS = {
    "N0030001": "勞動基準法",
    "N0030002": "勞動基準法施行細則",
    "N0030006": "勞工請假規則",
    "N0030014": "性別平等工作法",
    "N0030020": "勞工退休金條例",
}


@dataclass(frozen=True)
class Provider:
    """一個 OpenAI 相容的 LLM 供應商。"""

    name: str
    base_url: str
    api_key_env: str
    chat_models: tuple[str, ...]  # 同一供應商內依序嘗試：主模型忙碌時降級到較輕量的模型
    embed_model: str | None = None

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)


def _models(env: str, default: str) -> tuple[str, ...]:
    """模型清單可用環境變數覆寫，逗號分隔。"""
    return tuple(m.strip() for m in os.getenv(env, default).split(",") if m.strip())


# 依序嘗試；前一個失敗（額度用完、服務中斷、逾時）就自動換下一個
# 主模型指定穩定版本，而不用 gemini-flash-latest 別名：別名指向最新版，免費方案最容易塞車
# （2026-10-02 實測：最新版 3.8 逾時，3.5 約 4 秒、3.6 約 6 秒）。
# 指定版本日後可能被停用，但停用時 API 會立刻回 404，依序嘗試機制會馬上換下一個，幾乎不影響速度；
# 最後一個用輕量版的別名保底。可用環境變數 GEMINI_CHAT_MODELS 覆寫，不用改程式
PROVIDERS = [
    Provider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        chat_models=_models("GEMINI_CHAT_MODELS", "gemini-3.5-flash,gemini-3.6-flash,gemini-flash-lite-latest"),
        embed_model=os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
    ),
    Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        chat_models=_models("GROQ_CHAT_MODELS", "llama-3.3-70b-versatile"),
    ),
]
