"""多輪對話、輸入防護、API 錯誤處理的測試。LLM 一律用假的回應替代，不消耗 API 額度。"""

from openai import RateLimitError

from workrules import conversation, guard, llm, qa, rewrite
from workrules.conversation import Turn
from workrules.retrieval import Hit

# ── 輸入防護 ───────────────────────────────────────────────


def test_mask_pii_handles_chinese_adjacent_text():
    m = guard.mask_pii("我身分證A123456789，手機0912-345-678，信箱 a.b@example.com")
    assert "A123456789" not in m.text and "0912" not in m.text and "example.com" not in m.text
    assert m.found == ["身分證字號", "手機號碼", "Email"]


def test_mask_pii_leaves_normal_numbers_alone():
    text = "月薪 30000 元，加班 2 小時，函釋 1140149454 號"
    assert guard.mask_pii(text).text == text


def test_rate_limit_is_per_session_list():
    times = [100.0 + i for i in range(guard.MAX_QUESTIONS_PER_MINUTE)]
    assert guard.wait_seconds(times, now=110.0) == 51   # 最早一次在 100 秒，要等到 160 秒
    assert guard.wait_seconds(times, now=161.0) == 0
    assert guard.record(times, now=200.0) == [200.0]     # 過期的時間點會被丟掉


def test_too_long():
    assert guard.too_long("字" * (guard.MAX_CHARS + 1))
    assert not guard.too_long("字" * guard.MAX_CHARS)


# ── API 錯誤分類 ───────────────────────────────────────────


def test_classify_error_minute_and_day():
    minute = llm.classify_error(Exception("Error code: 429 ... GenerateRequestsPerMinutePerProjectPerModel-FreeTier ... Please retry in 47.4s"))
    assert (minute.kind, minute.retry_seconds) == ("rate_minute", 48)
    day = llm.classify_error(Exception("Error code: 429 ... GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    assert day.kind == "rate_day"
    assert llm.classify_error(Exception("Error code: 503 high demand")).kind == "overloaded"
    assert "47" not in day.message and "下午" in day.message


# ── 對話紀錄規則 ───────────────────────────────────────────


def test_context_turns_only_keeps_recent_ok_turns():
    turns = [Turn("q1", "q1"), Turn("q2", "q2"), Turn("bad", status="error"),
             Turn("q3", "q3"), Turn("?", status="clarify"), Turn("long", status="blocked")]
    assert [t.query for t in conversation.context_turns(turns)] == ["q2", "q3"]


def test_summarize_strips_markdown_and_disclaimer():
    s = conversation.summarize("### 結論\n**可以請 7 天**特休。\n\n※ 以上說明僅供參考")
    assert s == "結論 可以請 7 天特休。"


# ── 查詢改寫 ───────────────────────────────────────────────


def test_rewrite_parse_falls_back_on_bad_output():
    assert rewrite.parse('{"query": "休息日加班費怎麼算？", "followup": true}', "那休息日呢？").query == "休息日加班費怎麼算？"
    bad = rewrite.parse("抱歉我不懂", "那休息日呢？")
    assert (bad.query, bad.failed) == ("那休息日呢？", True)


def test_rewrite_skips_llm_without_history(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("不應呼叫")))
    assert rewrite.rewrite([], "特休幾天？").query == "特休幾天？"


# ── 問答流程 ───────────────────────────────────────────────

HIT = Hit(1, "N0030001", "勞動基準法", "law", "24", None, "勞動基準法 第 24 條\n加班費…", None, 1)


def _fake_search(monkeypatch, method="vector"):
    seen = {}

    def fake(conn, query, *a, **k):
        seen["query"] = query
        return [HIT], method
    monkeypatch.setattr(qa, "search_with_method", fake)
    return seen


def test_followup_is_rewritten_and_history_sent(monkeypatch):
    seen = _fake_search(monkeypatch)
    sent = {}

    def fake_chat(messages, **k):
        if k.get("purpose") == "rewrite":
            return llm.ChatResult('{"query": "休息日加班費怎麼算？", "followup": true}', "gemini/lite", False)
        sent["user"] = messages[-1]["content"]
        return llm.ChatResult("結論：前 2 小時加給 1又1/3【勞動基準法 第 24 條】", "gemini/main", False)
    monkeypatch.setattr(llm, "chat", fake_chat)

    history = [Turn("平日加班費怎麼算？", "平日加班費怎麼算？", summary="前 2 小時 4/3")]
    turn = qa.ask(None, "那休息日呢？", history)
    assert seen["query"] == "休息日加班費怎麼算？"
    assert "【先前對話】" in sent["user"] and "平日加班費怎麼算？" in sent["user"]
    assert turn.status == "ok" and turn.sources == [HIT] and turn.summary


def test_unclear_reference_asks_user_to_restate(monkeypatch):
    _fake_search(monkeypatch)
    monkeypatch.setattr(llm, "chat", lambda m, **k: llm.ChatResult('{"query": "x", "unclear": true}', "gemini/lite", False))
    turn = qa.ask(None, "回到第一題", [Turn("q", "q")])
    assert turn.status == "clarify" and "完整" in turn.answer


def test_api_error_becomes_error_turn_with_message(monkeypatch):
    _fake_search(monkeypatch)

    def fail(messages, **k):
        raise llm.NoProviderError(llm.ApiIssue("rate_minute", 30))
    monkeypatch.setattr(llm, "chat", fail)
    turn = qa.ask(None, "特休幾天？")
    assert turn.status == "error" and "30 秒" in turn.answer


def test_degraded_model_and_keyword_fallback_are_noted(monkeypatch):
    _fake_search(monkeypatch, method="keyword_fallback")
    monkeypatch.setattr(llm, "chat", lambda m, **k: llm.ChatResult("答案【勞動基準法 第 24 條】", "gemini/lite", True))
    notes = " ".join(qa.ask(None, "特休幾天？").notes)
    assert "關鍵字檢索" in notes and "備用模型" in notes


def test_chat_falls_back_then_reports_best_issue(monkeypatch):
    class FakeClient:
        def __init__(self, fail_with):
            self.chat = type("C", (), {"completions": type("X", (), {"create": staticmethod(self._create)})})()
            self.fail_with = fail_with

        def _create(self, **k):
            raise FakeClient.current

    def make_error(msg):
        import httpx
        return RateLimitError(msg, response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None)

    FakeClient.current = make_error("429 PerMinute Please retry in 20s")
    monkeypatch.setattr(llm, "_client", lambda p, *a: FakeClient(None))
    monkeypatch.setattr(llm, "available_providers", lambda: llm.config.PROVIDERS[:1])
    try:
        llm.chat([{"role": "user", "content": "hi"}])
    except llm.NoProviderError as e:
        assert e.issue.kind == "rate_minute" and e.issue.retry_seconds == 21
    else:
        raise AssertionError("應該丟出 NoProviderError")


# ── 工作階段計時與跳過失敗的模型 ─────────────────────────────

from workrules.progress import Progress


def test_progress_records_stages_and_failures():
    p = Progress()
    p.stage("查詢資料")
    p.stage("產生回答（A）")
    p.fail("無回應，逾時")
    p.stage("產生回答（B）")
    p.finish()
    stages = p.snapshot()
    assert [(s.label, s.state) for s in stages] == [
        ("查詢資料", "ok"), ("產生回答（A）", "failed"), ("產生回答（B）", "ok")]
    assert stages[1].note == "無回應，逾時"
    assert all(s.end is not None for s in stages) and p.finished


def _fake_models(monkeypatch, behaviour):
    """behaviour: 模型名稱 → "ok" 或例外。記錄實際呼叫順序。"""
    import httpx
    from openai import APITimeoutError
    calls = []

    class Client:
        def __init__(self):
            self.chat = type("C", (), {"completions": self})()

        def create(self, model, **k):
            calls.append(model)
            if behaviour[model] == "timeout":
                raise APITimeoutError(request=httpx.Request("POST", "http://x"))
            msg = type("M", (), {"content": f"from {model}"})()
            return type("R", (), {"choices": [type("Ch", (), {"message": msg})()]})()

    provider = llm.config.Provider("gemini", "http://x", "K", tuple(behaviour))
    monkeypatch.setattr(llm, "available_providers", lambda: [provider])
    monkeypatch.setattr(llm, "_client", lambda p, *a: Client())
    return calls


def test_chat_skips_models_marked_by_session(monkeypatch):
    calls = _fake_models(monkeypatch, {"main": "ok", "lite": "ok"})
    r = llm.chat([{"role": "user", "content": "hi"}], skip=frozenset({"gemini/main"}))
    assert calls == ["lite"] and r.degraded


def test_chat_ignores_skip_when_everything_is_skipped(monkeypatch):
    calls = _fake_models(monkeypatch, {"main": "ok", "lite": "ok"})
    llm.chat([{"role": "user", "content": "hi"}], skip=frozenset({"gemini/main", "gemini/lite"}))
    assert calls == ["main"]


def test_chat_timeout_is_reported_as_failure_and_timed(monkeypatch):
    _fake_models(monkeypatch, {"main": "timeout", "lite": "ok"})
    p = Progress()
    r = llm.chat([{"role": "user", "content": "hi"}], progress=p)
    assert r.model == "gemini/lite"
    assert [(m, i.kind) for m, i in r.failures] == [("gemini/main", "timeout")]
    assert [s.state for s in p.snapshot()] == ["failed", "running"]


# ── 串流 ───────────────────────────────────────────────────

def _fake_stream_models(monkeypatch, behaviour):
    """behaviour: 模型名稱 → 片段清單（字串）；清單中的例外會在該位置丟出，模擬生成到一半中斷。"""
    import httpx

    class Stream:
        def __init__(self, parts):
            self.parts = parts

        def __iter__(self):
            for part in self.parts:
                if isinstance(part, Exception):
                    raise part
                delta = type("D", (), {"content": part})()
                yield type("Chunk", (), {"choices": [type("Ch", (), {"delta": delta})()]})()

        def close(self):
            pass

    class Client:
        def __init__(self):
            self.chat = type("C", (), {"completions": self})()

        def create(self, model, stream=False, **k):
            assert stream
            return Stream(behaviour[model])

    provider = llm.config.Provider("gemini", "http://x", "K", tuple(behaviour))
    monkeypatch.setattr(llm, "available_providers", lambda: [provider])
    monkeypatch.setattr(llm, "_client", lambda p, *a: Client())
    return httpx


def test_stream_reports_partial_text_and_two_stages(monkeypatch):
    _fake_stream_models(monkeypatch, {"main": ["特休", "沒休完", "要發工資"]})
    seen, p = [], Progress()
    r = llm.chat([{"role": "user", "content": "hi"}], on_text=seen.append, progress=p)
    assert r.text == "特休沒休完要發工資"
    assert seen == ["特休", "特休沒休完", "特休沒休完要發工資"]
    labels = [s.label for s in p.snapshot()]
    assert labels == ["產生回答（main）：等待回應", "產生回答（main）：生成中"]


def test_stream_failure_midway_clears_text_and_falls_back(monkeypatch):
    httpx = _fake_stream_models(monkeypatch, {
        "main": ["寫到一半", httpx_timeout()],
        "lite": ["備用模型的完整回答"],
    })
    seen = []
    r = llm.chat([{"role": "user", "content": "hi"}], on_text=seen.append)
    assert seen == ["寫到一半", "", "備用模型的完整回答"]   # 中間的空字串讓畫面清掉半段文字
    assert r.model == "gemini/lite" and r.degraded
    assert [(m, i.kind) for m, i in r.failures] == [("gemini/main", "timeout")]


def test_stream_empty_response_counts_as_failure(monkeypatch):
    _fake_stream_models(monkeypatch, {"main": [], "lite": ["有內容"]})
    r = llm.chat([{"role": "user", "content": "hi"}], on_text=lambda t: None)
    assert r.text == "有內容" and r.failures[0][1].kind == "unavailable"


def httpx_timeout():
    import httpx
    return httpx.ReadTimeout("read timed out")
