"""Where Needle keeps its data.

Everything persistent (Chroma, the SQLite stores, uploaded originals, caches)
lives under one directory. It defaults to `backend/` so existing installs keep
their files; set NEEDLE_DATA_DIR to run an isolated copy, as the eval does.
Read at import time, so set the variable before importing the engine.
"""

import os

DATA_DIR = os.path.abspath(os.getenv("NEEDLE_DATA_DIR", "").strip() or os.path.dirname(__file__))


def data_path(*parts: str) -> str:
    return os.path.join(DATA_DIR, *parts)
