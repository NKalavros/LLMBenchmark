import json
from pathlib import Path
from typing import List, Optional
from statistics import mean

try:
    from sklearn.metrics import mean_squared_error
except Exception:  # pragma: no cover
    mean_squared_error = None  # type: ignore


def evaluate_models(root: Path, outdir: Path, agent_filter: Optional[List[str]] = None):
    """Skeleton evaluator.

    Strategy now: look for CSV files that might contain predictions with columns
    y_true / y_pred (case-insensitive). Compute RMSE if found.
    Future: actively run scripts to generate predictions.
    """
    import pandas as pd  # local import

    index_path = outdir / "files_index.json"
    if not index_path.exists():
        raise SystemExit("Run discover first")
    index = json.loads(index_path.read_text())

    results = []
    for agent in index['agents']:
        if agent_filter and agent not in agent_filter:
            continue
        agent_dir = root / agent
        # Find candidate csvs
        for csv in agent_dir.rglob('*.csv'):
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            cols = {c.lower(): c for c in df.columns}
            if 'y_true' in cols and 'y_pred' in cols and mean_squared_error:
                try:
                    rmse = mean_squared_error(df[cols['y_true']], df[cols['y_pred']], squared=False)
                except Exception:
                    continue
                results.append({
                    'agent': agent,
                    'file': str(csv.relative_to(root)),
                    'rows': len(df),
                    'rmse': rmse,
                })
    summary = {
        'results': results,
        'agents_evaluated': sorted({r['agent'] for r in results}),
    }
    (outdir / "rmse_results.json").write_text(json.dumps(summary, indent=2))
    print(f"Computed RMSE for {len(results)} prediction files.")
