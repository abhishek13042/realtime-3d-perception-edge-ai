import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Make scripts/download_data.py importable as `scripts_download` for testing its CLI.
_spec = importlib.util.spec_from_file_location("scripts_download", ROOT / "scripts" / "download_data.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["scripts_download"] = _mod
_spec.loader.exec_module(_mod)
