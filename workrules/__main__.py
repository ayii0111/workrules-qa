"""命令列入口：uv run python -m workrules <指令>"""

import argparse
import logging

from . import db, evaluate, guard, pipeline, qa


def main():
    parser = argparse.ArgumentParser(prog="workrules", description="規章與勞動法規問答助理")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("update", help="更新法規與工作規則、建立索引")
    ask = sub.add_parser("ask", help="提問")
    ask.add_argument("question")
    sub.add_parser("eval", help="評估三種檢索方式的命中率")
    sub.add_parser("status", help="顯示資料量與最近執行紀錄")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    with db.session() as conn:
        match args.cmd:
            case "update":
                pipeline.update(conn)
            case "ask":
                turn = qa.ask(conn, guard.mask_pii(args.question).text)
                print(turn.answer, "\n")
                for h in turn.sources:
                    print(f"- {h.label} {h.url or ''}")
                for n in turn.notes:
                    print(f"（{n}）")
            case "eval":
                print(evaluate.report(conn))
                print()
                print(evaluate.report_multiturn(conn))
            case "status":
                for r in conn.execute(
                    "SELECT s.name, s.version, count(a.id) n, sum(a.embedding IS NOT NULL) e "
                    "FROM sources s LEFT JOIN articles a ON a.code = s.code GROUP BY s.code"
                ):
                    print(f"{r['name']:20} {r['version']:20} 條文 {r['n']:>3}  已向量化 {r['e']:>3}")
                for r in conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 8"):
                    print(r["started_at"], r["job"], r["status"], r["n_new"], (r["message"] or "")[:80])


if __name__ == "__main__":
    main()
