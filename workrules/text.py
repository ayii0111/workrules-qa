"""中文斷詞。"""

import logging
import re

import jieba

jieba.setLogLevel(logging.WARNING)

# 常見的人資用語，加進詞典避免被切碎（例如「特休」被切成「特」「休」）
for word in ["特休", "特別休假", "加班費", "延長工作時間", "休息日", "例假", "資遣費", "預告期間",
             "生理假", "產假", "陪產檢及陪產假", "家庭照顧假", "普通傷病假", "事假", "婚假", "喪假",
             "平均工資", "工作規則", "補休", "育嬰留職停薪", "勞工退休金", "提繳"]:
    jieba.add_word(word)

_WORD_RE = re.compile(r"[\w一-鿿]")
STOPWORDS = set(
    "的 了 是 在 和 與 及 或 也 就 都 而 等 有 為 對 於 之 其 這 那 嗎 呢 吧 我 你 他 要 會 可以 能 "
    "什麼 如何 怎麼 怎樣 哪些 多少 幾 請問 請 規定 公司".split()
)


def tokenize(text: str) -> list[str]:
    """中文斷詞（搜尋引擎模式），去掉標點與停用詞。"""
    return [
        t.lower()
        for t in jieba.cut_for_search(text)
        if t.strip() and _WORD_RE.search(t) and t not in STOPWORDS
    ]
