"""工作階段計時：記錄問答流程每個階段的開始、結束與結果，讓介面即時顯示讀秒。

問答在背景執行緒中跑（呼叫 API 時畫面才不會凍結），介面每隔一小段時間讀取 snapshot() 重畫。
這裡只用一般 Python 物件與鎖，不呼叫任何 Streamlit 函式：Streamlit 的函式不能在背景執行緒中使用。
"""

import threading
import time
from dataclasses import dataclass


@dataclass
class Stage:
    label: str
    start: float
    end: float | None = None
    state: str = "running"  # running / ok / failed
    note: str = ""

    def seconds(self, now: float | None = None) -> float:
        return (self.end or now or time.time()) - self.start


class Progress:
    def __init__(self):
        self._lock = threading.Lock()
        self._stages: list[Stage] = []
        self.started = time.time()
        self.finished: float | None = None

    def stage(self, label: str):
        """開始新階段；上一個還在進行中的階段視為成功結束。"""
        now = time.time()
        with self._lock:
            self._close(now, "ok")
            self._stages.append(Stage(label, now))

    def fail(self, note: str):
        """目前階段失敗（例如模型逾時），記下原因。"""
        with self._lock:
            self._close(time.time(), "failed", note)

    def finish(self):
        with self._lock:
            self._close(time.time(), "ok")
            self.finished = time.time()

    def _close(self, now: float, state: str, note: str = ""):
        if self._stages and self._stages[-1].state == "running":
            s = self._stages[-1]
            s.end, s.state, s.note = now, state, note

    def snapshot(self) -> list[Stage]:
        with self._lock:
            return [Stage(s.label, s.start, s.end, s.state, s.note) for s in self._stages]

    def total(self) -> float:
        return (self.finished or time.time()) - self.started


class NullProgress(Progress):
    """不需要計時的呼叫端（命令列、評估）用這個，流程程式碼不必到處判斷 None。"""

    def stage(self, label: str):
        pass

    def fail(self, note: str):
        pass


def format_stage(s: Stage, now: float | None = None) -> str:
    icon = {"running": "⏳", "ok": "✅", "failed": "❌"}[s.state]
    line = f"{icon} {s.label}　**{s.seconds(now):.1f} 秒**"
    if s.note:
        line += f"　（{s.note}）"
    return line
