import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DB_PATH = ROOT / "data" / "portfolio.db"
CHECKPOINT_DB = ROOT / "data" / "checkpoints.db"
REPORTS_DIR = ROOT / "reports"