"""Measures the Macro agent's sentiment accuracy on labeled headlines, plus injection resistance.
Run:  uv run python -m evals.run_sentiment_eval"""
import asyncio
import json
from collections import Counter
from pathlib import Path

from finagent.agents.macro import analyze_headlines
from finagent.guardrails import sanitize_headlines

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LABELS = ["bullish", "neutral", "bearish"]


def _item(text: str) -> dict:
    return {"title": text, "summary": "", "publisher": "eval", "published": None}


async def main():
    data = json.loads((HERE / "sentiment_cases.json").read_text(encoding="utf-8"))
    sem = asyncio.Semaphore(5)

    async def classify(case):
        async with sem:
            out, _ = await analyze_headlines("EVAL", [_item(case["headline"])], "none")
            return {**case, "predicted": out["sentiment"], "confidence": out["confidence"]}

    rows = await asyncio.gather(*(classify(c) for c in data["cases"]))
    correct = sum(r["predicted"] == r["label"] for r in rows)
    accuracy = correct / len(rows)

    matrix = {t: Counter() for t in LABELS}
    for r in rows:
        matrix[r["label"]][r["predicted"]] += 1
    misses = [r for r in rows if r["predicted"] != r["label"]]

    inj_rows = []
    for case in data["injection_cases"]:
        items = [_item(h) for h in case["headlines"]]
        _, dropped = sanitize_headlines(items)
        out, _ = await analyze_headlines("EVAL", items, "none")
        passed = dropped >= 1 and out["sentiment"] != case["must_not_be"]
        inj_rows.append({"id": case["id"], "dropped": dropped, "sentiment": out["sentiment"], "passed": passed})

    lines = [
        "## Macro agent: sentiment evaluation",
        "",
        f"- Labeled single-headline cases: **{len(rows)}**",
        f"- Accuracy: **{accuracy:.0%}** ({correct}/{len(rows)})",
        f"- Prompt-injection cases resisted: **{sum(r['passed'] for r in inj_rows)}/{len(inj_rows)}**",
        "",
        "| actual \\ predicted | bullish | neutral | bearish |",
        "|---|---|---|---|",
    ]
    for t in LABELS:
        lines.append(f"| {t} | " + " | ".join(str(matrix[t][p]) for p in LABELS) + " |")
    if misses:
        lines += ["", "Misclassified headlines:"]
        lines += [f"- ({r['label']} -> {r['predicted']}) {r['headline']}" for r in misses]

    md = "\n".join(lines)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "sentiment_eval.md").write_text(md, encoding="utf-8")
    (RESULTS / "sentiment_eval.json").write_text(
        json.dumps({"accuracy": accuracy, "rows": rows, "injection": inj_rows}, indent=2), encoding="utf-8"
    )
    print(md)
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    asyncio.run(main())