import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

GEVAL_CRITERIA = {
    'readability': "Assess how readable and understandable the code snippet is for another engineer. Consider clarity, naming, structure.",
    'modularity': "Assess how modular the code is. Consider function decomposition, single-responsibility, and reuse potential.",
}


def _heuristic_quality(text: str):
    lines = text.splitlines()
    n = len(lines)
    comment = sum(1 for l in lines if l.strip().startswith(('#', '##', '//')))
    blank = sum(1 for l in lines if not l.strip())
    comment_ratio = comment / n if n else 0
    blank_ratio = blank / n if n else 0
    score = 0.4 * comment_ratio + 0.2 * (1 - blank_ratio) + 0.4 * min(1.0, 200 / (n + 1))
    return {
        'loc': n,
        'comment_ratio': round(comment_ratio, 3),
        'blank_ratio': round(blank_ratio, 3),
        'heuristic_score': round(score * 5, 2),
    }


def quality_scores(root: Path, outdir: Path, agent_filter: Optional[List[str]] = None, mock: bool = False):
    index_path = outdir / "files_index.json"
    if not index_path.exists():
        raise SystemExit("Run discover first")
    index = json.loads(index_path.read_text())

    api_key = None if mock else os.getenv("OPENAI_API_KEY")
    use_llm = bool(api_key) and not mock
    geval_metrics = {}
    if use_llm:
        try:  # pragma: no cover - optional
            from deepeval.metrics import GEval  # type: ignore
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # type: ignore
            for name, crit in GEVAL_CRITERIA.items():
                geval_metrics[name] = GEval(
                    name=name,
                    criteria=crit,
                    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
                    threshold=0.0,
                )
        except Exception:
            use_llm = False

    scores = []
    for f in index['files']:
        if agent_filter and f['agent'] not in agent_filter:
            continue
        path = root / f['path']
        try:
            text = path.read_text(errors='ignore')
        except Exception:
            text = ''
        heur = _heuristic_quality(text)
        entry = {**f, **heur}
        if use_llm and geval_metrics:
            # Limit size of code chunk for evaluation
            snippet = text[:4000]
            from deepeval.test_case import LLMTestCase  # type: ignore
            tc = LLMTestCase(input="code quality assessment", actual_output=snippet)
            llm_scores: Dict[str, Any] = {}
            for metric_name, metric in geval_metrics.items():
                try:
                    metric.measure(tc)  # may call external LLM
                    llm_scores[metric_name] = getattr(metric, 'score', None)
                except Exception:
                    llm_scores[metric_name] = None
            entry['llm_metrics'] = llm_scores
        scores.append(entry)
    out = {'files': scores, 'used_llm': use_llm, 'llm_metric_names': list(geval_metrics.keys()) if use_llm else []}
    (outdir / "quality_scores.json").write_text(json.dumps(out, indent=2))
    print(f"Quality scores generated for {len(scores)} files. LLM metrics: {use_llm}")
