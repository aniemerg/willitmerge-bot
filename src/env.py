"""
Load .env from the repo root. Import this module before accessing any env vars.
Importing it multiple times is safe (dotenv skips vars already set in the environment,
so shell exports still take precedence — use override=True below if you want .env to win).
"""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)
