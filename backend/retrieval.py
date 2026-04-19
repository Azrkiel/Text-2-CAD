"""
Few-Shot Retrieval for Machinist Context Augmentation.

Builds and queries a simple in-memory retrieval index over successful
CadQuery generation attempts from telemetry. Uses TF-IDF-style word
overlap for similarity (no external dependencies required).

The index is populated from the telemetry JSONL log on startup and
updated incrementally as new successes are logged.
"""

import json
import math
import os
import pathlib
import re
from collections import Counter, defaultdict
from typing import Optional

_TELEMETRY_DIR = pathlib.Path(
    os.environ.get("MIRUM_TELEMETRY_DIR", "/tmp/mirum/telemetry")
)
_LOG_FILE = _TELEMETRY_DIR / "critic_loop.jsonl"

# In-memory store: list of (description, code, domain) tuples
_INDEX: list[dict] = []
_IDF: dict[str, float] = {}
_TFIDF: list[dict[str, float]] = []
_INDEX_BUILT = False


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer (lowercase, alphanumeric only)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_idf(docs: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency for a corpus."""
    n = len(docs)
    if n == 0:
        return {}
    df: dict[str, int] = defaultdict(int)
    for doc in docs:
        for term in set(doc):
            df[term] += 1
    return {term: math.log(n / (1 + count)) for term, count in df.items()}


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    total = max(sum(tf.values()), 1)
    return {t: (count / total) * idf.get(t, 0.0) for t, count in tf.items()}


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in b)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a * norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_index() -> None:
    """Load successful telemetry records and build TF-IDF retrieval index."""
    global _INDEX, _IDF, _TFIDF, _INDEX_BUILT

    if not _LOG_FILE.exists():
        _INDEX_BUILT = True
        return

    records = []
    try:
        with open(_LOG_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if (
                        r.get("success_status")
                        and r.get("event_type") != "assembly"
                        and r.get("generated_code")
                        and r.get("part_id")
                    ):
                        records.append({
                            "description": r.get("part_id", ""),  # part_id ≈ description key
                            "code": r["generated_code"],
                            "domain": r.get("domain_classification", "A"),
                        })
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass

    # De-duplicate by code
    seen_codes: set[str] = set()
    unique = []
    for r in records:
        key = r["code"][:200]
        if key not in seen_codes:
            seen_codes.add(key)
            unique.append(r)

    _INDEX = unique[-500:]  # Keep latest 500 unique successes

    tokenized = [_tokenize(r["description"]) for r in _INDEX]
    _IDF = _build_idf(tokenized)
    _TFIDF = [_tfidf_vector(tokens, _IDF) for tokens in tokenized]
    _INDEX_BUILT = True


def retrieve_few_shots(
    description: str,
    domain: str,
    k: int = 3,
) -> list[dict]:
    """Return the k most similar successful (description, code) pairs.

    Filters by domain first (same domain = more relevant), then ranks
    by TF-IDF cosine similarity.

    Returns list of {"description": str, "code": str, "domain": str}.
    """
    global _INDEX_BUILT
    if not _INDEX_BUILT:
        build_index()

    if not _INDEX:
        return []

    query_tokens = _tokenize(description)
    query_vec = _tfidf_vector(query_tokens, _IDF)

    # Score all candidates
    scored = []
    for i, record in enumerate(_INDEX):
        # Prefer same domain (1.5× similarity boost)
        domain_boost = 1.5 if record["domain"] == domain else 1.0
        sim = _cosine_similarity(query_vec, _TFIDF[i]) * domain_boost
        scored.append((sim, i))

    scored.sort(reverse=True)
    top_k = [_INDEX[i] for _, i in scored[:k] if scored[0][0] > 0.05]

    return top_k


def add_to_index(description: str, code: str, domain: str) -> None:
    """Add a new successful generation to the retrieval index."""
    global _INDEX_BUILT
    if not _INDEX_BUILT:
        build_index()

    tokens = _tokenize(description)
    vec = _tfidf_vector(tokens, _IDF)

    _INDEX.append({"description": description, "code": code, "domain": domain})
    _TFIDF.append(vec)

    # Trim to 500 entries
    if len(_INDEX) > 500:
        _INDEX.pop(0)
        _TFIDF.pop(0)
