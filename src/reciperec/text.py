"""Text normalization for ingredients and cooking steps."""
from __future__ import annotations

import ast
import json
import re
from typing import Iterable, List

_UNITS = (
    "cups?|c|tablespoons?|tbsps?|tbs|teaspoons?|tsps?|grams?|g|kilograms?|kg|"
    "milliliters?|ml|liters?|l|ounces?|oz|pounds?|lbs?|pinch(?:es)?|dash(?:es)?|"
    "cloves?|cans?|packages?|pints?|quarts?|slices?|sticks?|large|medium|small"
)
_QTY = r"(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?|[¼-¾⅐-⅞])"
_QTY_UNIT_RE = re.compile(rf"\b{_QTY}\s*(?:-\s*{_QTY}\s*)?(?:(?:{_UNITS})\b\.?)?", re.I)
_PAREN_RE = re.compile(r"\([^)]*\)")
_NON_WORD_RE = re.compile(r"[^a-z\s]")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_PREP_WORDS = {
    "chopped", "diced", "minced", "sliced", "grated", "shredded", "crushed", "cubed",
    "finely", "coarsely", "thinly", "roughly", "freshly", "fresh", "divided", "softened",
    "melted", "beaten", "peeled", "seeded", "drained", "rinsed", "trimmed", "halved",
    "quartered", "packed", "optional", "taste", "to", "or", "more", "as", "needed",
    "for", "garnish", "room", "temperature", "at", "into", "cut", "pieces", "inch",
    "and", "of", "the", "a",
}


def parse_list(value) -> List[str]:
    """Parse a JSON- or Python-style list stored as a string."""
    if value is None or (isinstance(value, float) and value != value):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
        return [str(parsed).strip()]
    return [s.strip() for s in text.split(",") if s.strip()]


def normalize_ingredient(text: str) -> str:
    """'1 1/2 cups soy sauce (low sodium)' -> 'soy sauce'."""
    s = _PAREN_RE.sub(" ", text.lower())
    s = _QTY_UNIT_RE.sub(" ", s)
    s = _NON_WORD_RE.sub(" ", s)
    return " ".join(w for w in s.split() if w not in _PREP_WORDS)


def ingredient_tokens(ingredients: Iterable[str]) -> str:
    """Join each ingredient into one token so 'soy sauce' becomes 'soy_sauce'."""
    out = []
    for ing in ingredients:
        norm = normalize_ingredient(ing)
        if norm:
            out.append(norm.replace(" ", "_"))
    return " ".join(out)


def split_sentences(steps: Iterable[str]) -> List[str]:
    sentences: List[str] = []
    for step in steps:
        sentences.extend(s.strip() for s in _SENT_RE.split(step) if s.strip())
    return sentences


def ingredient_analyzer(text: str) -> List[str]:
    """Tokenizer for the 'soy_sauce chicken_thighs' format: words plus word pairs that
    stay inside one ingredient, so no pair spans two ingredients ('sauce chicken')."""
    out: List[str] = []
    for phrase in text.split():
        words = [w for w in phrase.split("_") if w]
        out.extend(words)
        out.extend(f"{a} {b}" for a, b in zip(words, words[1:]))
    return out


_GENERIC = {
    "sauce", "powder", "ground", "paste", "juice", "extract", "dried", "whole", "white", "black",
    "red", "green", "yellow", "large", "small", "fresh", "cold", "hot", "warm", "stalk", "stalks",
    "leaf", "leaves", "seasoning", "mix", "style", "kosher", "sea", "all", "purpose",
}


def _singular(word: str) -> str:
    if word.endswith("oes") and len(word) > 4:
        return word[:-2]          # tomatoes -> tomato
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"    # berries -> berry
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]          # onions -> onion
    return word


def ingredient_keywords(normalized: str) -> set:
    """Key words of a normalized ingredient, singular and without generic words."""
    return {_singular(w) for w in normalized.split() if w not in _GENERIC} or {_singular(w) for w in normalized.split()}
