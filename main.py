import argparse
import asyncio
import json
import sys

from finagent.config import REPORTS_DIR
from finagent.graph import resume_analysis, start_analysis

CHOICES = {"1": "weigh_news", "2": "weigh_technicals", "3": "neither"}


def ask_human(info: dict):
    macro, quant = info.get("macro") or {}, info.get("quant") or {}
    print("\n--- The analysts disagree: your input is needed ---")
    print(f"Macro (news):       {macro.get('sentiment')} | themes: {', '.join(macro.get('key_themes', []))}")
    print(f"Quant (technicals): {quant.get('bias')} | {quant.get('summary', '')}")
    print("\n  1) Weigh the NEWS more\n  2) Weigh the TECHNICALS more\n  3) Neither (stay cautious)")
    while True:
        pick = input("Your choice [1/2/3]: ").strip()
        if pick in CHOICES:
            break
    note = input("Optional note (press Enter to skip): ").strip()
    return CHOICES[pick], note


async def run(ticker: str, hitl: bool) -> dict:
    def show(node, update):
        print(f"[done] {node}")

    result = await start_analysis(ticker, hitl=hitl, on_update=show)
    while result["status"] == "waiting":
        choice, note = ask_human(result["interrupt"] or {})
        result = await resume_analysis(result["run_id"], choice, note, on_update=show)
    return result["state"]


def main():
    parser = argparse.ArgumentParser(description="Run the multi-agent asset analysis")
    parser.add_argument("--ticker", required=True, help="e.g. RELIANCE.NS, AAPL, BTC-USD")
    parser.add_argument("--hitl", action="store_true", help="pause and ask me when the analysts conflict")
    args = parser.parse_args()

    try:
        state = asyncio.run(run(args.ticker, args.hitl))
    except ValueError as e:
        print(f"Input error: {e}")
        sys.exit(1)

    final = state.get("final")
    problems = state.get("errors", [])
    if problems:
        print("\nProblems:", *problems, sep="\n - ")
    if not final:
        print("No report produced.")
        return
    print(json.dumps(final, indent=2))

    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"{final['ticker']}_{state['run_id']}.json"
    path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()