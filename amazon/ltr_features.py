"""Shared OpenSearch LTR feature definitions and XGBoost export helpers.

OpenSearch LTR 3.7 represents a feature query that does not match as NaN, while
its legacy XGBoost JSON evaluator does not follow XGBoost's ``missing`` branch.
Every lexical feature therefore includes a tiny match-all clause.  The feature
is present for every candidate and has value EPSILON when the real query does
not match.  Training data must be collected from the same feature set.
"""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from typing import Any, Iterable


EPSILON = 1.0e-6
NO_TOKEN = "zzzzltrnomatchzzzz"
NEGATION_MARKERS = ("非対応", "未使用", "テープなし", "無し", "なし", "不要", "除く", "以外")
CONTRARY_RULES = {
    "テープなし": "テープ付 テープ付き",
    "非対応": "対応",
    "未使用": "使用済み",
    "不要": "必要",
    "無し": "付き 付属",
    "なし": "付き 付属",
}


def _dense_query(query: dict[str, Any]) -> dict[str, Any]:
    """Make a scoring query match every candidate without changing useful scores."""
    return {
        "bool": {
            "should": [
                query,
                {
                    "constant_score": {
                        "filter": {"match_all": {}},
                        "boost": EPSILON,
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }


def feature(name: str, query: dict[str, Any], params: Iterable[str] = ("keywords",)) -> dict[str, Any]:
    return {"name": name, "params": list(params), "template": _dense_query(query)}


def feature_set(name: str) -> dict[str, Any]:
    """Canonical Japanese ESCI lexical feature set used for training and serving."""
    features = [
        feature("title_bm25", {"match": {"product_title": "{{keywords}}"}}),
        feature("title_phrase", {"match_phrase": {"product_title": "{{keywords}}"}}),
        feature("brand_bm25", {"match": {"product_brand": "{{keywords}}"}}),
        feature("brand_phrase", {"match_phrase": {"product_brand": "{{keywords}}"}}),
        feature("title_reading", {"match": {"product_title.reading": "{{keywords}}"}}),
        feature("brand_reading", {"match": {"product_brand.reading": "{{keywords}}"}}),
        feature("title_ngram", {"match": {"product_title.ngram": "{{keywords}}"}}),
        feature("brand_ngram", {"match": {"product_brand.ngram": "{{keywords}}"}}),
        feature("bullet_bm25", {"match": {"product_bullet_point": "{{keywords}}"}}),
        feature("description_bm25", {"match": {"product_description": "{{keywords}}"}}),
        feature(
            "lexical_v2",
            {
                "bool": {
                    "minimum_should_match": 1,
                    "should": [
                        {"match_phrase": {"product_title": {"query": "{{keywords}}", "boost": 8.0}}},
                        {"match": {"product_title": {"query": "{{keywords}}", "boost": 4.0, "operator": "or"}}},
                        {"match_phrase": {"product_brand": {"query": "{{keywords}}", "boost": 6.0}}},
                        {"match": {"product_brand": {"query": "{{keywords}}", "boost": 3.0, "operator": "or"}}},
                        {"match": {"product_bullet_point": {"query": "{{keywords}}", "boost": 0.5}}},
                        {"match": {"product_description": {"query": "{{keywords}}", "boost": 0.1}}},
                        {"match": {"product_title.reading": {"query": "{{keywords}}", "boost": 0.8, "operator": "or"}}},
                        {"match": {"product_brand.reading": {"query": "{{keywords}}", "boost": 0.8, "operator": "or"}}},
                        {"match": {"product_title.ngram": {"query": "{{keywords}}", "boost": 0.25, "minimum_should_match": "70%"}}},
                        {"match": {"product_brand.ngram": {"query": "{{keywords}}", "boost": 0.20, "minimum_should_match": "70%"}}},
                    ],
                },
            },
        ),
    ]
    return {
        "featureset": {"name": name, "features": features},
        "validation": {"params": {"keywords": "サングラス"}, "index": "amazon-jp-v2"},
    }


def query_params(query: str) -> dict[str, Any]:
    """Deterministic query-side signals shared by training and serving."""
    normalized = unicodedata.normalize("NFKC", str(query)).lower().strip()
    latin = re.findall(r"[a-z][a-z0-9._+-]*", normalized)
    numbers = re.findall(r"\d+(?:\.\d+)?", normalized)
    codes = [token for token in latin if any(char.isdigit() for char in token)]
    codes.extend(number for number in numbers if number not in codes)
    markers = []
    for marker in NEGATION_MARKERS:
        if marker in normalized and not any(marker in selected for selected in markers):
            markers.append(marker)
    contrary = [CONTRARY_RULES[marker] for marker in markers if marker in CONTRARY_RULES]
    constraint = normalized.replace("なし", " ").replace("無し", " ")
    constraint = constraint.replace("非対応", "対応").replace("未使用", "使用")
    for marker in ("不要", "除く", "以外"):
        constraint = constraint.replace(marker, " ")
    constraint = re.sub(r"\s+", " ", constraint).strip()
    chunks = re.findall(r"[a-z0-9._+-]+|[一-龯々〆ヵヶぁ-んァ-ヴー]+", normalized)
    return {
        "keywords": normalized or NO_TOKEN,
        "normalized_query": normalized or NO_TOKEN,
        "code_tokens": " ".join(dict.fromkeys(codes)) or NO_TOKEN,
        "numeric_tokens": " ".join(dict.fromkeys(numbers)) or NO_TOKEN,
        "negation_terms": " ".join(markers) or NO_TOKEN,
        "contrary_terms": " ".join(contrary) or NO_TOKEN,
        "constraint_keywords": constraint or NO_TOKEN,
        "query_token_count": float(min(len(chunks), 10)) + EPSILON,
        "has_numeric": 1.0 if numbers else EPSILON,
        "has_latin": 1.0 if latin else EPSILON,
        "has_negation": 1.0 if markers else EPSILON,
        "is_short_query": 1.0 if len(normalized.replace(" ", "")) <= 4 else EPSILON,
    }


def _constant_feature(name: str, param: str) -> dict[str, Any]:
    return {
        "name": name,
        "params": [param],
        "template": {
            "constant_score": {
                "filter": {"match_all": {}},
                "boost": "{{" + param + "}}",
            }
        },
    }


def feature_set_v3(name: str) -> dict[str, Any]:
    """v2 plus strict coverage, attribute, code, negation, and query-type signals."""
    base = feature_set(name)["featureset"]["features"]
    added = [
        feature("title_and", {"match": {"product_title": {"query": "{{keywords}}", "operator": "and"}}}),
        feature("brand_and", {"match": {"product_brand": {"query": "{{keywords}}", "operator": "and"}}}),
        feature("bullet_and", {"match": {"product_bullet_point": {"query": "{{keywords}}", "operator": "and"}}}),
        feature("description_and", {"match": {"product_description": {"query": "{{keywords}}", "operator": "and"}}}),
        feature("title_raw_exact", {"term": {"product_title.raw": "{{normalized_query}}"}}, ("normalized_query",)),
        feature("brand_raw_exact", {"term": {"product_brand.raw": "{{normalized_query}}"}}, ("normalized_query",)),
        feature("color_bm25", {"match": {"product_color": "{{keywords}}"}}),
        feature("code_title", {"match": {"product_title": {"query": "{{code_tokens}}", "operator": "and"}}}, ("code_tokens",)),
        feature("code_brand", {"match": {"product_brand": {"query": "{{code_tokens}}", "operator": "and"}}}, ("code_tokens",)),
        feature("numeric_title", {"match": {"product_title": {"query": "{{numeric_tokens}}", "operator": "and"}}}, ("numeric_tokens",)),
        feature("negation_title", {"match": {"product_title": {"query": "{{negation_terms}}", "operator": "and"}}}, ("negation_terms",)),
        feature("negation_bullet", {"match": {"product_bullet_point": {"query": "{{negation_terms}}", "operator": "and"}}}, ("negation_terms",)),
        feature("contrary_title", {"match": {"product_title": {"query": "{{contrary_terms}}", "operator": "or"}}}, ("contrary_terms",)),
        feature("contrary_bullet", {"match": {"product_bullet_point": {"query": "{{contrary_terms}}", "operator": "or"}}}, ("contrary_terms",)),
        feature("constraint_title_and", {"match": {"product_title": {"query": "{{constraint_keywords}}", "operator": "and"}}}, ("constraint_keywords",)),
        _constant_feature("query_token_count", "query_token_count"),
        _constant_feature("has_numeric", "has_numeric"),
        _constant_feature("has_latin", "has_latin"),
        _constant_feature("has_negation", "has_negation"),
        _constant_feature("is_short_query", "is_short_query"),
    ]
    validation_params = query_params("PP袋 A4 テープなし")
    return {
        "featureset": {"name": name, "features": base + added},
        "validation": {"params": validation_params, "index": "amazon-jp-v2"},
    }


def semantic_feature(name: str = "semantic_cosine") -> dict[str, Any]:
    """Experimental semantic feature definition.

    OpenSearch k-NN script scoring works when ``query_value`` is a literal JSON
    number array.  In this environment, however, the LTR plugin's Mustache
    parameter expansion stringifies dynamic vector params before the k-NN script
    receives them.  As a result, this feature is kept as documentation of the
    intended server-side shape, but v4 semantic reranking is implemented outside
    the LTR plugin in ``run_semantic_hybrid_v4.py``.
    """
    return {
        "name": name,
        "params": ["query_vector"],
        "template": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "lang": "knn",
                    "source": "knn_score",
                    "params": {
                        "field": "product_embedding",
                        "query_value": "{{#toJson}}query_vector{{/toJson}}",
                        "space_type": "cosinesimil",
                    },
                },
            }
        },
    }


def feature_set_v4(name: str, validation_index: str = "amazon-jp-semantic-v1") -> dict[str, Any]:
    """v3 features plus semantic query-product similarity.

    This is not registered by the current v4 workflow; see ``semantic_feature``.
    """
    features = feature_set_v3(name)["featureset"]["features"] + [semantic_feature()]
    validation_params = query_params("自転車スピーカー")
    validation_params["query_vector"] = [0.0] * 768
    validation_params["query_vector"][0] = 1.0
    return {
        "featureset": {"name": name, "features": features},
        "validation": {"params": validation_params, "index": validation_index},
    }


def nonnegative_xgboost_dump(booster: Any) -> tuple[list[dict[str, Any]], float]:
    """Export trees for the LTR plugin, shifting leaves to nonnegative scores.

    Lucene query scores cannot be negative.  Adding one constant per tree keeps
    document ordering exactly unchanged.  The returned offset can be added to
    XGBoost's raw-margin prediction for numerical parity checks.
    """
    trees = [json.loads(tree) for tree in booster.get_dump(dump_format="json")]
    total_offset = 0.0

    def leaves(node: dict[str, Any]) -> list[dict[str, Any]]:
        if "leaf" in node:
            return [node]
        return [leaf for child in node["children"] for leaf in leaves(child)]

    output = copy.deepcopy(trees)
    for tree in output:
        tree_leaves = leaves(tree)
        offset = max(0.0, -min(float(node["leaf"]) for node in tree_leaves))
        total_offset += offset
        for node in tree_leaves:
            node["leaf"] = float(node["leaf"]) + offset
    return output, total_offset


def create_model_payload(model_name: str, featureset_name: str, booster: Any) -> tuple[dict[str, Any], float]:
    trees, offset = nonnegative_xgboost_dump(booster)
    return (
        {
            "model": {
                "name": model_name,
                "model": {"type": "model/xgboost+json", "definition": json.dumps(trees)},
            }
        },
        offset,
    )
