import asyncio
import re
from datetime import date, datetime, timezone

# ---------- ticker validation ----------
TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-^=]{0,14}$")


def validate_ticker(raw: str) -> str:
    t = (raw or "").strip().upper()
    if not TICKER_RE.match(t):
        raise ValueError(
            f"Invalid ticker '{raw}'. Use letters, digits, '.' or '-', e.g. TCS.NS, AAPL, BTC-USD."
        )
    return t


# ---------- data quality ----------
MIN_CANDLES = 60
MAX_STALE_DAYS = 5


def data_quality_warnings(metrics: dict, today: date | None = None) -> list[str]:
    warns = []
    today = today or datetime.now(timezone.utc).date()
    n = metrics.get("n_candles") or 0
    if n < MIN_CANDLES:
        warns.append(
            f"Only {n} daily candles available (fewer than {MIN_CANDLES}); long-term indicators may be missing or unreliable."
        )
    as_of = metrics.get("as_of")
    if as_of:
        try:
            age = (today - date.fromisoformat(as_of)).days
            if age > MAX_STALE_DAYS:
                warns.append(f"Latest price data is {age} days old (as of {as_of}); the analysis may be stale.")
        except ValueError:
            pass
    return warns


# ---------- prompt-injection defense for headlines ----------
# Regexes catch only obvious attacks. The real defenses are the <headlines> delimiters
# and the system-prompt rule that headlines are data, never instructions.
INJECTION_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"ignore\b.{0,30}\b(instructions|prompts?|rules)",
        r"disregard\b.{0,30}\b(instructions|prompts?|rules)",
        r"\byou are now\b",
        r"\bsystem prompt\b",
        r"\bnew instructions?\b",
        r"\bconfidence\s*(of|=|:)\s*(1|1\.0|100)\b",
    ]
]


def _clean_text(s, limit: int = 300) -> str:
    s = re.sub(r"[\x00-\x1f\x7f]", " ", str(s or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


def sanitize_headlines(items):
    """Return (clean_items, dropped_count)."""
    clean, dropped = [], 0
    for it in items or []:
        if not isinstance(it, dict):
            continue
        title = _clean_text(it.get("title"))
        summary = _clean_text(it.get("summary"))
        if not title:
            continue
        if any(p.search(title) or p.search(summary) for p in INJECTION_PATTERNS):
            dropped += 1
            continue
        clean.append({**it, "title": title, "summary": summary})
    return clean, dropped


# ---------- language guardrail ----------
BANNED_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"\bguaranteed\s+(?:returns?|profits?|gains?|growth)\b",
        r"\bguarantees?\s+(?:returns?|profits?|gains?)\b",
        r"\brisk[- ]free\b(?!\s+rate)",
        r"\bcan'?t lose\b",
        r"\bsure (?:thing|bet|profit)\b",
        r"\bwill (?:definitely |certainly |surely )?(?:rise|go up|skyrocket|soar|crash|plunge|double)\b",
        r"\bno risk\b",
        r"\b(?:buy|sell) now\b",
        r"\bto the moon\b",
    ]
]


def find_banned_phrases(text: str) -> list[str]:
    return sorted({m.group(0).lower() for p in BANNED_PATTERNS for m in p.finditer(text or "")})


# ---------- grounding check ----------
NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
SAFE_NUMBERS = {1, 2, 3, 4, 5, 6, 10, 12, 14, 20, 24, 30, 50, 60, 70, 90, 100, 180, 200, 252, 365}


def _numbers_in_text(text: str):
    out = []
    for tok in NUM_RE.findall(text or ""):
        try:
            out.append((tok, float(tok.replace(",", ""))))
        except ValueError:
            pass
    return out


def _collect(obj, acc: set):
    if obj is None or isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if obj == obj:  # skip NaN
            acc.add(float(obj))
    elif isinstance(obj, str):
        for _, v in _numbers_in_text(obj):
            acc.add(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, acc)


def _tol(a: float) -> float:
    return max(0.06, 0.5 if abs(a) >= 10 else 0.0, 0.005 * abs(a))


def ungrounded_numbers(text: str, sources) -> list[str]:
    """Numbers that appear in `text` but cannot be traced to any number in `sources`.
    Fractions are also matched as percentages (0.2667 -> 26.67). A heuristic, so it flags and does not block."""
    base: set = set()
    _collect(sources, base)
    allowed = set()
    for v in base:
        allowed.add(abs(v))
        allowed.add(abs(v) * 100)
    bad = []
    for tok, n in _numbers_in_text(text):
        if n in SAFE_NUMBERS or (n.is_integer() and 1990 <= n <= 2100):
            continue
        if any(abs(n - x) <= _tol(x) for x in allowed):
            continue
        if tok not in bad:
            bad.append(tok)
    return bad


# ---------- retries ----------
async def with_retry(fn, attempts: int = 3, base_delay: float = 1.0):
    """Await fn() (a zero-arg function returning a coroutine) with exponential backoff."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return await fn()
        except Exception as e:
            last = e
            if i < attempts - 1:
                await asyncio.sleep(base_delay * (2**i))
    if last is None:
        raise RuntimeError("with_retry needs attempts >= 1")
    raise last