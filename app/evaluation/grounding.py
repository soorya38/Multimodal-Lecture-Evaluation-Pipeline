"""
Lightweight retrieval for grounding the technical evaluation.

The technical score is only as trustworthy as what it's compared against. Without
a reference, the LLM judges "does this *sound* correct"; with one, it can judge
"does this *match* the source material". This module provides that source
material to the evaluator by retrieving the passages of a supplied reference
document most relevant to what the lecturer actually said/showed.

Retrieval is deliberately dependency-free: reference text is split into chunks
and scored against the lecture content with a small TF-IDF ranker. This is not a
vector database — it needs no embedding model or external service — but it is
enough to surface the right passages for grounding. The ranker is isolated behind
``build_grounding_context`` so it can later be swapped for embeddings/a vector
store without touching the evaluator.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import structlog

logger = structlog.get_logger(__name__)

# Minimal English/Tamil-transliteration stopword set — just enough to stop the
# ranker from being dominated by function words. Not meant to be exhaustive.
_STOPWORDS = frozenset(
    """
    a an and the is are was were be been being of to in on for with as by at from
    this that these those it its it's we you they he she i or nor not but if then
    so such can will would should could may might do does did done has have had
    about into over under above below more most some any all each our your their
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2 and t not in _STOPWORDS]


def chunk_text(text: str, chunk_chars: int) -> list[str]:
    """
    Split text into ~``chunk_chars``-sized chunks on paragraph/sentence-ish
    boundaries, so retrieved passages stay coherent rather than cut mid-sentence.
    """
    text = text.strip()
    if not text:
        return []

    # Prefer to break on blank lines; fall back to packing sentences.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for para in paragraphs:
        if len(para) > chunk_chars:
            flush()
            # Split an oversized paragraph on sentence boundaries.
            for sentence in re.split(r"(?<=[.!?])\s+", para):
                if len(buf) + len(sentence) + 1 > chunk_chars:
                    flush()
                buf = f"{buf} {sentence}".strip()
            flush()
        elif len(buf) + len(para) + 2 > chunk_chars:
            flush()
            buf = para
        else:
            buf = f"{buf}\n\n{para}".strip()

    flush()
    return chunks


def retrieve_relevant(query: str, chunks: list[str], top_k: int) -> list[str]:
    """
    Return the ``top_k`` chunks most relevant to ``query`` by TF-IDF score,
    preserving each chunk's original document order in the result.
    """
    if not chunks:
        return []

    query_terms = set(_tokenize(query))
    if not query_terms:
        return chunks[:top_k]

    tokenized = [_tokenize(c) for c in chunks]
    n = len(chunks)

    # Document frequency per term across chunks -> idf.
    df: Counter[str] = Counter()
    for toks in tokenized:
        for t in set(toks):
            df[t] += 1
    idf = {t: math.log(1 + n / (1 + df[t])) for t in df}

    scored: list[tuple[float, int]] = []
    for i, toks in enumerate(tokenized):
        if not toks:
            continue
        counts = Counter(toks)
        length = len(toks)
        # Sum of tf*idf over query terms present in this chunk (length-normalised).
        score = sum((counts[t] / length) * idf.get(t, 0.0) for t in query_terms if t in counts)
        if score > 0:
            scored.append((score, i))

    scored.sort(key=lambda s: s[0], reverse=True)
    top_indices = sorted(i for _, i in scored[:top_k])
    return [chunks[i] for i in top_indices]


def build_grounding_context(
    reference_text: str,
    query_text: str,
    top_k: int,
    chunk_chars: int,
) -> str:
    """
    Build a grounding context string from reference material: the ``top_k`` chunks
    most relevant to the lecture content, joined for inclusion in the evaluation
    prompt. Returns "" when there is no usable reference.
    """
    chunks = chunk_text(reference_text, chunk_chars)
    if not chunks:
        return ""

    relevant = retrieve_relevant(query_text, chunks, top_k)
    logger.info(
        "Built grounding context",
        reference_chunks=len(chunks),
        selected_chunks=len(relevant),
        top_k=top_k,
    )
    return "\n\n---\n\n".join(relevant)
