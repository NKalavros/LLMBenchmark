import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional

AGENT_FOLDERS_EXCLUDE = {"HeadlessAgents", "analyses", "agent_benchmark", "__pycache__","tests"}
CODE_EXTENSIONS = {".py", ".R"}


def is_agent_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    name = path.name
    if name.startswith('.'):
        return False
    if name in AGENT_FOLDERS_EXCLUDE:
        return False
    # Heuristic: contains at least one code file of interest
    for ext in CODE_EXTENSIONS:
        if any(path.rglob(f"*{ext}")):
            return True
    return False


def discover_files(root: Path, outdir: Path, agent_filter: Optional[List[str]] = None) -> Dict:
    agents = []
    files = []
    for child in sorted(root.iterdir()):
        if not is_agent_dir(child):
            continue
        if agent_filter and child.name not in agent_filter:
            continue
        agents.append(child.name)
        for f in child.rglob('*'):
            if f.is_file() and f.suffix in CODE_EXTENSIONS:
                rel = f.relative_to(root)
                try:
                    text = f.read_text(errors='ignore')
                except Exception:
                    text = ''
                loc = text.count('\n') + 1 if text else 0
                files.append({
                    'agent': child.name,
                    'path': str(rel),
                    'language': 'python' if f.suffix == '.py' else 'r',
                    'size_bytes': f.stat().st_size,
                    'loc': loc,
                })

    index = {"agents": agents, "files": files}
    (outdir / "files_index.json").write_text(json.dumps(index, indent=2))
    return index
