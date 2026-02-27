"""src/main.py
Redirect shim — the pipeline entry point has moved to the project root (main.py).

Both invocation styles still work:
    python main.py --run-all               # preferred (from project root)
    python -m src.main --run-all           # Airflow DAG / legacy callers
"""
import sys
import runpy
from pathlib import Path

# Ensure project root is on sys.path so root-level main.py can import src.*
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Delegate entirely to the root-level main.py
runpy.run_path(str(_root / "main.py"), run_name="__main__")
