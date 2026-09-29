# 測試紀錄

## 自動化測試

執行：`uv run pytest`（12 項）

| 範圍 | 測試 | 驗證內容 |
|---|---|---|
| 資料來源 | `test_parse_law_extracts_version_chapter_and_skips_deleted` | 法規頁解析出修正日期、章名、條號（含 32-1 這類），並略過已刪除的條文 |
| | `test_parse_handbook_skips_notes_and_keeps_titles` | 工作規則解析出條號、標題、章名，說明文字不會混進條文 |
| | `test_save_source_skips_same_version_and_replaces_new_version` | 版本沒變不重寫；版本改變時整部替換，舊的全文索引一併清除 |
| 檢索 | `test_keyword_search_finds_chinese_terms` | 中文關鍵字檢索 |
| | `test_explicit_refs_match_law_name_and_alias` | 「勞基法第 32-1 條」「工作規則第 7 條」直接命中；「施行細則第 32-1 條」不會誤配到勞基法 |
| | `test_rrf_rewards_items_ranked_by_both_methods` | RRF 合併排名 |
| | `test_cited_hits_keeps_only_cited_in_order` | 只列出回答實際引用的條文 |
| 試算 | `test_months_between_counts_full_months_only` | 未滿一個月不計 |
| | `test_annual_leave_days_follow_article_38` | 特休天數對照勞基法第 38 條各級距：未滿 6 個月、6 個月、1、2、4、5、10、25 年（上限 30 日） |
| | `test_leave_schedule_handles_month_end` | 月底到職的邊界情況（見錯誤紀錄 E3） |
| | `test_overtime_pay_workday_tiers` | 平日加班 3 小時：前 2 小時 ×4/3、第 3 小時 ×5/3 |
| | `test_overtime_pay_restday_tiers_and_limit` | 休息日加班 10 小時三段倍率；平日超過 4 小時會報錯 |

## 檢索品質評估

執行：`uv run python -m workrules eval`，結果見 [README](../README.md#檢索品質評估)。

## 手動驗收測試

### 2026-09-30（第一版）

| # | 測試項目 | 預期 | 結果 |
|---|---|---|---|
| 1 | 抓取 5 部法規 | 條文數與網站一致，版本為修正日期 | ✅ 共 290 條（勞基法 98、施行細則 63、請假規則 13、性平法 49、勞退條例 67） |
| 2 | 重複執行更新 | 法規沒修正時不重抓、不重建 | ✅ 第二次執行各來源「更新 0」 |
| 3 | 向量額度限制 | 遇到 429 時等待後繼續 | ✅ 等待 3 次後 311 條全部完成 |
| 4 | 指出違法內規 | 問「加班只能換補休不能領錢合法嗎」 | ✅ 指出工作規則第 7 條違反勞基法第 32-1 條，並引用第 1 條 |
| 5 | 內規與法規並列 | 問「滿一年特休幾天、沒休完怎麼辦」 | ✅ 7 天；遞延與發給工資，同時引用工作規則第 9 條與勞基法第 38 條 |
| 6 | 分級規定 | 問「離職要提前多久說」 | ✅ 正確列出 10／20／30 日三個級距 |
| 7 | 拒答 | 問「公司的股票代號是多少」 | ✅ 回答找不到相關條文，建議詢問人資，不列出不相關的條文 |
| 8 | 特休試算 | 到職 2024-03-01，查詢 2026-09-30 | ✅ 10 天（年資 2 年 6 個月） |
| 9 | 加班費試算 | 月薪 36,000、休息日加班 10 小時 | ✅ 2,700 元（400 + 1,500 + 800） |
| 10 | 網頁介面 | AppTest 無頭執行四個分頁、切換試算選項、篩選條文、提問 | ✅ 無例外 |
