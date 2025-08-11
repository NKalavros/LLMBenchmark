import json
from pathlib import Path
import sys

# Ensure root path on sys.path for direct pytest runs without installation
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_benchmark.cli import main


def test_discover(tmp_path, monkeypatch):
    # Create a fake structure
    (tmp_path / 'AgentA').mkdir()
    (tmp_path / 'AgentA' / 'model.py').write_text("# test\nprint('hi')\n")
    outdir = tmp_path / 'analyses'
    main(["discover", "--root", str(tmp_path), "--outdir", str(outdir)])
    data = json.loads((outdir / 'files_index.json').read_text())
    assert data['agents'] == ['AgentA']
    assert data['files'][0]['loc'] > 0
