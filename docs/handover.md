# 交接文件

給接手維護的工程師。

## 專案結構

```
app.py                      Streamlit 介面（問答、試算、條文查詢、資料狀態）
workrules/
  __main__.py               命令列入口（update / ask / eval / status）
  config.py                 收錄的法規清單、LLM 供應商、路徑
  db.py                     SQLite schema 與連線
  sources.py                全國法規資料庫爬取、工作規則解析、寫入（依版本決定是否替換）
  text.py                   中文斷詞（含人資用語詞典）
  index.py                  全文索引與向量
  retrieval.py              檢索策略（條號直達 + 向量，退回關鍵字）
  qa.py                     問答流程、提示詞、引用解析
  rewrite.py                追問改寫成獨立問題
  conversation.py           對話回合的資料結構，哪些回合可進入下一輪脈絡
  guard.py                  輸入防護：長度、個資遮蔽、單一 session 提問頻率
  calc.py                   特休與加班費試算
  evaluate.py               檢索品質評估
  pipeline.py               更新流程：各步驟獨立執行與記錄
data/
  handbook.md               公司工作規則（虛構範例）
  interpretations.json      精選函釋的文號清單（原文由程式從勞動部抓取）
  guidance.json             主管機關說明文章清單（網址、擷取範圍）
  eval_set.json             檢索評估題目
  eval_multiturn.json       多輪追問評估題目
  workrules.db              資料庫（提交進 repo，部署時直接使用）
.github/workflows/weekly.yml 每週檢查法規更新
```

## 環境設定

| 變數 | 用途 | 必填 |
|---|---|---|
| `GEMINI_API_KEY` | Gemini 對話與 embedding | 建議（沒有時只能用關鍵字檢索，問答無法使用） |
| `GROQ_API_KEY` | 備援對話模型 | 否 |
| `GEMINI_CHAT_MODELS` | 覆寫 Gemini 模型清單，逗號分隔 | 否 |

本機放在 `.env`；GitHub Actions 放在 repo 的 Secrets；Streamlit Cloud 放在 App settings → Secrets。**key 不要提交進 git。**

## 換成貴公司的工作規則

1. 把公司的工作規則整理成 `data/handbook.md` 的格式：
   ```markdown
   # 公司名稱 員工工作規則
   ## 第一章 章名
   ### 第 1 條 條文標題
   條文內容……
   ```
   以 `>` 開頭的行是說明文字，不會被收錄
2. 執行 `uv run python -m workrules update`。系統依檔案內容判斷有沒有變，有變才會整份替換並重建索引
3. **把常見問題加進 `data/eval_set.json`**，執行 `uv run python -m workrules eval` 確認都找得到

## 新增一部法規

1. 到[全國法規資料庫](https://law.moj.gov.tw/)找到法規，網址中的 `pcode=` 就是法規代碼
2. 加進 `config.py` 的 `LAWS`
3. 執行 `uv run python -m workrules update`

## 新增函釋或主管機關說明

- **函釋**：在[勞動部勞動法令查詢系統](https://laws.mol.gov.tw/)找到函釋，把文號中的數字（例如 `1140149454`）與主題加進 `data/interpretations.json`
- **說明文章**：在 `data/guidance.json` 加一筆，填網址、發布機關、日期、文章標題（擷取起點），以及下一篇文章的標題（擷取終點）
- 執行 `uv run python -m workrules update`：清單內容有變才會重新抓取

## 多輪對話的規則

見[設計文件](design-multiturn.md)。修改時特別注意：錯誤、被擋下、需澄清的回合不能進入對話脈絡（`conversation.context_turns`）。

## 修改檢索或提示詞之後

一定要跑 `uv run python -m workrules eval`，確認命中率沒有下降。目前的檢索策略就是依評估結果決定的（見 README）。

## 常見維護工作

| 狀況 | 處理 |
|---|---|
| 法規網站改版，抓不到條文 | 「資料狀態」會出現 error；修改 `sources.parse_law` 的解析邏輯，並更新測試中的 `LAW_HTML` |
| LLM 模型被停用（404） | 用環境變數 `GEMINI_CHAT_MODELS` 改模型名稱，不用改程式 |
| 換了 embedding 模型 | 清空向量後重建：`UPDATE articles SET embedding = NULL`，再執行 update（向量維度不同時，系統會自動退回關鍵字檢索，不會算出錯誤結果） |
| 員工反映某類問題答不好 | 先把問題加進 `eval_set.json`，確認是「沒找到條文」還是「找到但回答錯」，再決定改檢索還是改提示詞 |
