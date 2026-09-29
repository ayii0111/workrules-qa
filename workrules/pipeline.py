"""更新流程：抓法規 → 讀工作規則 → 建索引 → 補向量。

法規有修正時（修正日期改變）才會整部替換；沒變就略過，所以可以放心重複執行。
每個步驟獨立記錄在 runs 表，單一步驟失敗不會中斷其他步驟。
"""

import logging
import sqlite3
import traceback
from collections.abc import Callable

from . import config, db, index, sources

log = logging.getLogger(__name__)


def _step(conn: sqlite3.Connection, job: str, fn: Callable[[], int]) -> int:
    try:
        n = fn()
        db.log_run(conn, job, "ok", n)
        conn.commit()
        log.info("%-9s ok，更新 %d", job, n)
        return n
    except Exception as e:  # noqa: BLE001 —— 刻意全部攔下，記錄後繼續下一步
        conn.rollback()
        db.log_run(conn, job, "error", 0, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
        conn.commit()
        log.error("%-9s 失敗：%s", job, e)
        return 0


def update(conn: sqlite3.Connection):
    _step(conn, "laws", lambda: sum(sources.save_source(conn, s) for s in sources.fetch_laws(list(config.LAWS))))
    _step(conn, "handbook", lambda: sources.save_source(conn, sources.load_handbook()))
    _step(conn, "index", lambda: index.index_missing(conn))
    _step(conn, "embed", lambda: index.embed_missing(conn))
