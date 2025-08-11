import os
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any

try:  # Optional external deps
    import numpy as np  # noqa
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _hash_embedding(text: str, dim: int = 384):
    """Deterministic pseudo-embedding using SHA256 hashing for offline mode."""
    h = hashlib.sha256(text.encode('utf-8')).digest()
    # Repeat digest to fill dim
    raw = (h * ((dim // len(h)) + 1))[:dim]
    # Normalize to [0,1]
    vec = [b / 255.0 for b in raw]
    return vec


def _basic_summary(text: str) -> Dict[str, Any]:
    """Return a lightweight heuristic summary structure.

    This is used when DSPy summarization is unavailable. Keeps keys consistent
    with the richer DSPy path for downstream uniformity.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return {
            'high_level_summary': '(empty file)',
            'key_components': [],
            'modeling_approach': None,
        }
    header = lines[:8]
    model_words = [w for w in lines if any(k in w.lower() for k in ['glm', 'lm(', 'regression', 'elasticnet', 'randomforest', 'rf', 'xgboost'])]
    approach = None
    if model_words:
        # crude guess of first modeling keyword line
        approach_line = model_words[0].lower()
        if 'elastic' in approach_line:
            approach = 'elastic_net'
        elif 'xgboost' in approach_line:
            approach = 'xgboost'
        elif 'random' in approach_line or 'rf' in approach_line:
            approach = 'random_forest'
        elif 'glm' in approach_line:
            approach = 'glm'
        elif 'lm(' in approach_line or 'regression' in approach_line:
            approach = 'linear_regression'
    return {
        'high_level_summary': " | ".join(header)[:500] + (' ...' if len(lines) > len(header) else ''),
        'key_components': [],
        'modeling_approach': approach,
    }


def _try_dspy_summarize(text: str):
    """Attempt a richer DSPy-based summarization; fall back on failure.

    Returns dict with keys: high_level_summary, key_components, modeling_approach
    """
    if not text.strip():
        return _basic_summary(text)
    if not os.getenv('OPENAI_API_KEY'):
        return _basic_summary(text)
    try:
        # Local import to avoid dependency cost when not installed
        import dspy  # type: ignore
        from .dspy_summarizer import CodeSummarizer
        # Configure only once (idempotent)
        # Using a lightweight model name; user can override via env if desired
        if not getattr(dspy, '_agent_benchmark_configured', False):
            try:
                dspy.configure(lm=dspy.OpenAI(model=os.getenv('AGENT_BENCHMARK_DSPY_MODEL', 'gpt-4o-mini')))
                dspy._agent_benchmark_configured = True  # type: ignore
            except Exception:
                return _basic_summary(text)
        module = CodeSummarizer()
        result = module(code=text[:14000])  # limit tokens
        return {
            'high_level_summary': (result.high_level_summary or '').strip() or _basic_summary(text)['high_level_summary'],
            'key_components': [c.strip() for c in (result.key_components or '').split('\n') if c.strip()],
            'modeling_approach': (result.modeling_approach or '').strip() or _basic_summary(text)['modeling_approach'],
        }
    except Exception:
        return _basic_summary(text)


def analyze_codebase(
    root: Path,
    outdir: Path,
    agent_filter: Optional[List[str]] = None,
    mock: bool = False,
    model: str = "text-embedding-3-small",
    max_files: Optional[int] = None,
    store_full: bool = False,
):
    index_path = outdir / "files_index.json"
    if not index_path.exists():
        raise SystemExit("Run discover first to create files_index.json")
    index = json.loads(index_path.read_text())

    api_key = os.getenv("OPENAI_API_KEY")
    use_real = bool(api_key) and not mock

    if use_real:
        try:
            import openai  # type: ignore
            client = openai.OpenAI(api_key=api_key)
        except Exception as e:  # pragma: no cover
            print(f"Falling back to mock embeddings due to import error: {e}")
            use_real = False
    else:
        client = None  # type: ignore

    analyzed = []
    full_embeddings_matrix = []  # optional accumulation when store_full
    for i, f in enumerate(index['files']):
        if max_files and i >= max_files:
            break
        if agent_filter and f['agent'] not in agent_filter:
            continue
        file_path = root / f['path']
        try:
            text = file_path.read_text(errors='ignore')
        except Exception:
            text = ''
        if use_real and client:
            try:  # pragma: no cover - network dependent
                emb_resp = client.embeddings.create(model=model, input=text[:8000] or ' ')
                embedding = emb_resp.data[0].embedding  # type: ignore
            except Exception as e:  # Fallback to hash
                print(f"Embedding failure for {f['path']}: {e}. Using hash embedding.")
                embedding = _hash_embedding(text)
        else:
            embedding = _hash_embedding(text)
        # Summarization (DSPy if possible, else heuristic)
        summary_struct = _try_dspy_summarize(text)
        record = {
            **f,
            'summary': summary_struct['high_level_summary'],
            'modeling_approach': summary_struct.get('modeling_approach'),
            'key_components': summary_struct.get('key_components', [])[:10],
            'embedding_dim': len(embedding),
            'embedding_method': 'openai' if use_real else 'hash-deterministic',
            'embedding_preview': embedding[:16],
        }
        if store_full:
            record['embedding_full'] = embedding  # WARNING: large
            full_embeddings_matrix.append(embedding)
        analyzed.append(record)

    out = {
        "files": analyzed,
        "total": len(analyzed),
        "used_real_embeddings": use_real,
        "used_dspy": any(f.get('key_components') for f in analyzed),
        "embedding_model": model if use_real else None,
        "stored_full_embeddings": store_full,
    }
    (outdir / "code_semantics.json").write_text(json.dumps(out, indent=2))
    if store_full and full_embeddings_matrix:
        try:
            import numpy as np  # type: ignore
            np.savez_compressed(outdir / "embeddings.npz", embeddings=np.array(full_embeddings_matrix, dtype='float32'))
        except Exception as e:  # pragma: no cover
            print(f"Failed to write embeddings.npz: {e}")
    print(f"Analyzed {len(analyzed)} files. Real embeddings: {use_real}")
