"""Runs the full graph N times on one ticker to measure stability and grounding.
Run:  uv run python -m evals.run_runtime_eval --ticker WIPRO.NS --runs 5"""
import argparse
import asyncio
import json
import statistics
from collections import Counter
from pathlib import Path

from finagent.graph import run_analysis

RESULTS = Path(__file__).resolve().parent / "results"


async def main(ticker: str, runs: int):
    rows = []
    for i in range(runs):
        try:
            state = await run_analysis(ticker)
            final = state.get("final")
            if not final:
                raise RuntimeError("no final report produced")
            rows.append(
                {
                    "run": i + 1,
                    "verdict": final["verdict"],
                    "agreement": final["agreement"],
                    "action": final["suggested_action"],
                    "confidence": final["overall_confidence"],
                    "ungrounded": final.get("ungrounded_numbers", []),
                    "banned": final.get("banned_phrases", []),
                    "warnings": final.get("warnings", []),
                }
            )
        except Exception as e:
            rows.append({"run": i + 1, "error": str(e)})
        print(rows[-1])

    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("All runs failed.")
        return

    def modal_share(key):
        return Counter(r[key] for r in ok).most_common(1)[0][1] / len(ok)

    confs = [r["confidence"] for r in ok]
    clean = sum(1 for r in ok if not r["ungrounded"] and not r["banned"])
    lines = [
        f"## Runtime evaluation: {ticker} ({runs} runs)",
        "",
        f"- Successful runs: **{len(ok)}/{runs}**",
        f"- Verdict stability (most common verdict share): **{modal_share('verdict'):.0%}** {dict(Counter(r['verdict'] for r in ok))}",
        f"- Agreement stability: **{modal_share('agreement'):.0%}** {dict(Counter(r['agreement'] for r in ok))}",
        f"- Confidence: mean **{statistics.mean(confs):.2f}**, stdev **{statistics.stdev(confs) if len(confs) > 1 else 0:.2f}**",
        f"- Grounding pass rate (no untraceable numbers, no banned phrases): **{clean / len(ok):.0%}**",
        "",
        "Note: headlines can change between runs, so some instability comes from the data, not the model.",
    ]
    md = "\n".join(lines)
    RESULTS.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace(".", "_")
    (RESULTS / f"runtime_eval_{safe}.md").write_text(md, encoding="utf-8")
    (RESULTS / f"runtime_eval_{safe}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="WIPRO.NS")
    p.add_argument("--runs", type=int, default=5)
    a = p.parse_args()
    asyncio.run(main(a.ticker.upper(), a.runs))