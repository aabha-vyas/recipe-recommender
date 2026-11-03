"""Build every field representation used in the experiments."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .models import FieldRetriever, LsaEncoder, SentenceEncoder, TfidfEncoder
from .text import ingredient_analyzer

TFIDF_FIELDS = ("title", "ing_words", "ing_phrase", "steps")
LSA_FIELDS = ("lsa_title", "lsa_ing", "lsa_steps")
EMBED_FIELDS = ("emb_title", "emb_ing", "emb_steps")
COURSE_FIELDS = ("course_search", "course_steps")


def field_texts(df: pd.DataFrame) -> dict:
    ing_words = df["ingredients_text"].str.replace("_", " ", regex=False)
    return {
        "title": df["title_text"],
        "ing_words": ing_words,
        "ing_phrase": df["ingredients_text"],
        "steps": df["steps_text"],
        "course_search": df["title_text"] + " " + df["ingredients_text"],
    }


def _cached_embeddings(name: str, texts: pd.Series, model: str, cache_dir: Optional[Path], get_encoder):
    """Embeddings are slow on CPU, so cache them keyed by the model and the exact texts."""
    key = hashlib.sha1((model + "\0" + "\0".join(texts)).encode()).hexdigest()[:16]
    path = cache_dir / f"{name}-{key}.npy" if cache_dir else None
    if path and path.exists():
        return np.load(path)
    emb = get_encoder().transform(texts)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
    return emb


def build_retriever(df: pd.DataFrame, groups: Iterable[str] = ("tfidf", "course", "lsa"),
                    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2", seed: int = 0,
                    cache_dir: Optional[str | Path] = "artifacts") -> FieldRetriever:
    t = field_texts(df)
    groups = set(groups)
    r = FieldRetriever()
    if "tfidf" in groups:
        r.add("title", TfidfEncoder(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(t["title"]))
        r.add("ing_words", TfidfEncoder(analyzer=ingredient_analyzer, min_df=2, sublinear_tf=True).fit_transform(t["ing_phrase"]))
        r.add("ing_phrase", TfidfEncoder(token_pattern=r"\S+", min_df=2).fit_transform(t["ing_phrase"]))
        r.add("steps", TfidfEncoder(stop_words="english", min_df=2, sublinear_tf=True,
                                    max_features=50000).fit_transform(t["steps"]))
    if "course" in groups:
        # The original CS 410 setup: title + ingredient phrases, and steps, mixed 0.85 / 0.15.
        r.add("course_search", TfidfEncoder(ngram_range=(1, 2), min_df=2, max_features=10000)
              .fit_transform(t["course_search"]))
        r.add("course_steps", TfidfEncoder(stop_words="english", max_features=10000).fit_transform(t["steps"]))
    if "lsa" in groups:
        r.add("lsa_title", LsaEncoder(100, seed, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(t["title"]))
        r.add("lsa_ing", LsaEncoder(200, seed, analyzer=ingredient_analyzer, min_df=2, sublinear_tf=True).fit_transform(t["ing_phrase"]))
        r.add("lsa_steps", LsaEncoder(200, seed, stop_words="english", min_df=2, sublinear_tf=True,
                                      max_features=50000).fit_transform(t["steps"]))
    if "embed" in groups:
        cache = Path(cache_dir) if cache_dir else None
        encoder = []

        def get_encoder():
            if not encoder:
                encoder.append(SentenceEncoder(embed_model))
            return encoder[0]

        for name, texts in (("emb_title", t["title"]), ("emb_ing", t["ing_words"]),
                            ("emb_steps", t["steps"].str.slice(0, 2000))):
            r.add(name, _cached_embeddings(name, texts, embed_model, cache, get_encoder))
    return r
