"""Semantic hybrid / XGBoost v4 experiments for Japanese Amazon ESCI.

This script keeps the production-safe v3 OpenSearch LTR model intact and adds a
semantic v4 path that is evaluated as an external reranker:

1. lexical and/or k-NN retrieval creates candidates in OpenSearch;
2. the application/notebook computes dense query-product similarity;
3. XGBoost reranks candidates with v3 lexical features + semantic_cosine;
4. a cheap pre-rerank score is fitted to approximate the v4 rank using only
   lexical_v2 and semantic_cosine.

Why external reranking?  OpenSearch k-NN script scoring works with a literal
query vector, but OpenSearch LTR feature templates stringify dynamic vector
parameters in this environment.  That makes a direct ``query_vector`` LTR
feature fail before the k-NN script receives a numeric list.  The semantic
feature is therefore implemented in Python for v4.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge

from ltr_features import feature_set_v3, query_params
from run_ltr_xgboost_v3 import (
    BASE_URL,
    GAINS,
    INDEX,
    LABELS,
    paired,
    per_query_ndcg,
    ranked_ndcg,
    render,
    validation_query,
)
from semantic_embeddings import (
    ARTIFACT_DIR as SEMANTIC_ARTIFACT_DIR,
    DIMENSION,
    SEMANTIC_INDEX,
    encode_queries,
    load_model,
    load_products,
    product_passage,
)


HERE = Path(__file__).resolve().parent
ARTIFACT_DIR = HERE / "artifacts/semantic_hybrid_v4"
V3_ARTIFACT_DIR = HERE / "artifacts/ltr_xgboost_v3"
FULL_EMBEDDINGS = SEMANTIC_ARTIFACT_DIR / "product_embeddings_full.npy"
FULL_PRODUCT_IDS = SEMANTIC_ARTIFACT_DIR / "product_ids_full.json"
JUDGED_EMBEDDINGS = ARTIFACT_DIR / "judged_product_embeddings.npy"
JUDGED_PRODUCT_IDS = ARTIFACT_DIR / "judged_product_ids.json"
QUERY_EMBEDDINGS = ARTIFACT_DIR / "query_embeddings.npy"
QUERY_IDS = ARTIFACT_DIR / "query_ids.json"
MODEL_NAME = "esci_jp_xgb_v4_external_semantic"


def check(response: requests.Response) -> requests.Response:
    if not response.ok:
        raise RuntimeError(f"OpenSearch {response.status_code}: {response.text[:2000]}")
    return response


def feature_names_v3() -> list[str]:
    return [item["name"] for item in feature_set_v3("esci_jp_features_v3")["featureset"]["features"]]


def load_v3_features() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    names = feature_names_v3()
    train = pd.read_parquet(V3_ARTIFACT_DIR / "train_features_v3.parquet")
    test = pd.read_parquet(V3_ARTIFACT_DIR / "test_features_v3.parquet")
    train["is_validation"] = train.query_id.map(validation_query)
    return train, test, names


def matrix(frame: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort_values(["query_id", "example_id"])
    return (
        ordered[names].to_numpy(np.float32),
        ordered.esci_label.map(LABELS).to_numpy(np.int32),
        ordered.groupby("query_id", sort=False).size().to_numpy(np.int32),
    )


def _load_json_list(path: Path) -> list[str]:
    return [str(item) for item in json.loads(path.read_text())]


def query_embedding_map(examples: pd.DataFrame, batch_size: int = 64) -> dict[str, np.ndarray]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    query_table = (
        examples[["query_id", "query"]]
        .drop_duplicates("query_id")
        .sort_values("query_id")
        .reset_index(drop=True)
    )
    ids = query_table.query_id.astype(str).tolist()
    if QUERY_EMBEDDINGS.exists() and QUERY_IDS.exists() and _load_json_list(QUERY_IDS) == ids:
        embeddings = np.load(QUERY_EMBEDDINGS)
    else:
        embeddings = encode_queries(query_table.query.astype(str).tolist(), batch_size=batch_size)
        np.save(QUERY_EMBEDDINGS, embeddings.astype("float32"))
        QUERY_IDS.write_text(json.dumps(ids, ensure_ascii=False))
    return dict(zip(ids, embeddings, strict=True))


def _load_full_embedding_subset(product_ids: list[str]) -> dict[str, np.ndarray] | None:
    if not FULL_EMBEDDINGS.exists() or not FULL_PRODUCT_IDS.exists():
        return None
    all_ids = _load_json_list(FULL_PRODUCT_IDS)
    wanted = set(product_ids)
    positions = {pid: i for i, pid in enumerate(all_ids) if pid in wanted}
    if len(positions) != len(wanted):
        return None
    embeddings = np.load(FULL_EMBEDDINGS, mmap_mode="r")
    return {pid: np.asarray(embeddings[pos], dtype=np.float32) for pid, pos in positions.items()}


def judged_product_embedding_map(examples: pd.DataFrame, batch_size: int = 128) -> dict[str, np.ndarray]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    product_ids = sorted(examples.product_id.astype(str).unique())

    cached_full = _load_full_embedding_subset(product_ids)
    if cached_full is not None:
        return cached_full

    if JUDGED_EMBEDDINGS.exists() and JUDGED_PRODUCT_IDS.exists() and _load_json_list(JUDGED_PRODUCT_IDS) == product_ids:
        embeddings = np.load(JUDGED_EMBEDDINGS, mmap_mode="r")
        return {pid: np.asarray(embeddings[i], dtype=np.float32) for i, pid in enumerate(product_ids)}

    products = load_products()
    products["product_id"] = products.product_id.astype(str)
    products = products[products.product_id.isin(product_ids)].copy()
    missing = sorted(set(product_ids) - set(products.product_id))
    if missing:
        raise RuntimeError(f"{len(missing)} judged products are missing from product table")
    products = products.set_index("product_id").loc[product_ids].reset_index()
    model = load_model()
    texts = [product_passage(row) for _, row in products.iterrows()]
    started = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    np.save(JUDGED_EMBEDDINGS, embeddings)
    JUDGED_PRODUCT_IDS.write_text(json.dumps(product_ids, ensure_ascii=False))
    print(f"encoded {len(product_ids):,} judged products in {time.time() - started:.1f}s")
    return {pid: embeddings[i] for i, pid in enumerate(product_ids)}


def add_semantic_feature(frame: pd.DataFrame, query_vectors: dict[str, np.ndarray], product_vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    output = frame.copy()
    scores = np.empty(len(output), dtype=np.float32)
    for i, row in enumerate(output[["query_id", "product_id"]].itertuples(index=False)):
        qv = query_vectors[str(row.query_id)]
        pv = product_vectors[str(row.product_id)]
        scores[i] = float(np.dot(qv, pv))
    output["semantic_cosine"] = scores
    return output


CONFIGS = [
    ("pairwise_d3", "rank:pairwise", 3),
    ("pairwise_d5", "rank:pairwise", 5),
    ("ndcg_d3", "rank:ndcg", 3),
    ("ndcg_d5", "rank:ndcg", 5),
]


def select_model(train: pd.DataFrame, names: list[str], feature_version: str) -> tuple[dict, pd.DataFrame]:
    fit = train[~train.is_validation].sort_values(["query_id", "example_id"])
    valid = train[train.is_validation].sort_values(["query_id", "example_id"]).copy()
    x_fit, y_fit, g_fit = matrix(fit, names)
    x_valid, y_valid, g_valid = matrix(valid, names)
    rows = []
    best = None
    for config, objective, depth in CONFIGS:
        model = xgb.XGBRanker(
            objective=objective,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=depth,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1,
            tree_method="hist",
            random_state=42,
            early_stopping_rounds=40,
            eval_metric="ndcg",
        )
        model.fit(x_fit, y_fit, group=g_fit, eval_set=[(x_valid, y_valid)], eval_group=[g_valid], verbose=False)
        column = f"prediction_{feature_version}_{config}"
        valid[column] = model.predict(x_valid, output_margin=True)
        score = float(per_query_ndcg(valid, column).mean())
        trees = int(model.best_iteration) + 1
        record = {
            "feature_version": feature_version,
            "config": config,
            "objective": objective,
            "max_depth": depth,
            "trees": trees,
            "validation_official_nDCG": score,
        }
        rows.append(record)
        print(record, flush=True)
        candidate = {
            "score": score,
            "config": config,
            "objective": objective,
            "depth": depth,
            "trees": trees,
            "model": model,
            "valid": valid,
            "column": column,
        }
        if best is None or score > best["score"]:
            best = candidate
    assert best is not None
    return best, pd.DataFrame(rows)


def fit_final(train: pd.DataFrame, names: list[str], choice: dict) -> xgb.XGBRanker:
    x_all, y_all, g_all = matrix(train, names)
    model = xgb.XGBRanker(
        objective=choice["objective"],
        n_estimators=choice["trees"],
        learning_rate=0.05,
        max_depth=choice["depth"],
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1,
        tree_method="hist",
        random_state=42,
    )
    model.fit(x_all, y_all, group=g_all, verbose=False)
    return model


def add_v3_predictions(test: pd.DataFrame, names_v3: list[str]) -> pd.DataFrame:
    output = test.copy()
    booster = xgb.Booster()
    booster.load_model(V3_ARTIFACT_DIR / "xgboost_model.json")
    output["v3"] = booster.predict(
        xgb.DMatrix(output[names_v3].to_numpy(np.float32), feature_names=names_v3),
        output_margin=True,
    )
    return output


def normalize_per_query(frame: pd.DataFrame, column: str, output: str) -> pd.DataFrame:
    def transform(values: pd.Series) -> pd.Series:
        span = values.max() - values.min()
        if span <= 1e-12:
            return pd.Series(np.zeros(len(values), dtype=np.float32), index=values.index)
        return (values - values.min()) / span

    frame[output] = frame.groupby("query_id", sort=False)[column].transform(transform)
    return frame


def add_rank_target(frame: pd.DataFrame, score: str = "v4") -> pd.DataFrame:
    output = frame.copy()
    output["_rank"] = output.groupby("query_id", sort=False)[score].rank(method="first", ascending=False)
    output["_count"] = output.groupby("query_id", sort=False)[score].transform("count")
    denom = (output["_count"] - 1).clip(lower=1)
    output["v4_rank_target"] = 1.0 - ((output["_rank"] - 1) / denom)
    return output.drop(columns=["_rank", "_count"])


def fit_cheap_score(valid: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    work = valid.copy()
    work = normalize_per_query(work, "lexical_v2", "lexical_v2_norm")
    work = normalize_per_query(work, "semantic_cosine", "semantic_norm")
    work = add_rank_target(work, "v4")
    x = work[["lexical_v2_norm", "semantic_norm"]].to_numpy(np.float32)
    y = work["v4_rank_target"].to_numpy(np.float32)
    candidates = {
        "linear": LinearRegression(),
        "ridge_0.1": Ridge(alpha=0.1),
        "ridge_1.0": Ridge(alpha=1.0),
    }
    rows = []
    best = None
    for name, model in candidates.items():
        model.fit(x, y)
        work[f"cheap_{name}"] = model.predict(x)
        query_rows = []
        for qid, group in work.groupby("query_id", sort=True):
            n = len(group)
            k = min(10, n)
            target_top = set(group.nlargest(k, "v4").product_id.astype(str))
            cheap_top = set(group.nlargest(k, f"cheap_{name}").product_id.astype(str))
            query_rows.append(
                {
                    "query_id": qid,
                    "top10_overlap": len(target_top & cheap_top) / k if k else 0.0,
                    "ndcg": ranked_ndcg(group.rename(columns={f"cheap_{name}": "cheap"}), "cheap"),
                }
            )
        per_query = pd.DataFrame(query_rows)
        record = {
            "cheap_model": name,
            "intercept": float(model.intercept_),
            "beta_lexical_v2_norm": float(model.coef_[0]),
            "beta_semantic_norm": float(model.coef_[1]),
            "mean_top10_overlap_with_v4": float(per_query.top10_overlap.mean()),
            "mean_official_nDCG": float(per_query.ndcg.mean()),
        }
        rows.append(record)
        if best is None or record["mean_top10_overlap_with_v4"] > best["mean_top10_overlap_with_v4"]:
            best = record
    assert best is not None
    return best, pd.DataFrame(rows)


def lexical_query_from_v3(query: str) -> dict:
    features = feature_set_v3("esci_jp_features_v3")["featureset"]["features"]
    lexical_dense = next(item["template"] for item in features if item["name"] == "lexical_v2")
    lexical_template = lexical_dense["bool"]["should"][0]
    return render(lexical_template, query_params(query))


def hybrid_candidate_body(query: str, query_vector: np.ndarray, size: int = 100, k: int = 100, semantic_boost: float = 1.0) -> dict:
    """Build a simple bool.should lexical+kNN body.

    The initial OpenSearch score is the sum of the matched should clauses.  This
    is not RRF.  It is intentionally cheap and monotonic in BM25/k-NN scores, so
    its ``semantic_boost`` can later be replaced by the cheap_score regression
    coefficient ratio.
    """
    lexical = lexical_query_from_v3(query)
    return {
        "size": size,
        "_source": ["product_id", "product_title", "product_brand"],
        "query": {
            "bool": {
                "should": [
                    lexical,
                    {
                        "knn": {
                            "product_embedding": {
                                "vector": query_vector.astype(float).tolist(),
                                "k": k,
                                "boost": semantic_boost,
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        },
    }


def semantic_index_ready(index: str = SEMANTIC_INDEX) -> bool:
    response = requests.get(f"{BASE_URL}/{index}/_count", timeout=10)
    if not response.ok:
        return False
    return int(response.json().get("count", 0)) > 0


def smoke_hybrid_queries(test: pd.DataFrame, query_vectors: dict[str, np.ndarray], index: str, limit: int = 10) -> pd.DataFrame:
    rows = []
    if not semantic_index_ready(index):
        return pd.DataFrame(rows)
    sample = test[["query_id", "query"]].drop_duplicates("query_id").head(limit)
    for row in sample.itertuples(index=False):
        body = hybrid_candidate_body(str(row.query), query_vectors[str(row.query_id)], size=10, k=100, semantic_boost=1.0)
        hits = check(requests.post(f"{BASE_URL}/{index}/_search", json=body, timeout=30)).json()["hits"]["hits"]
        for rank, hit in enumerate(hits, start=1):
            source = hit.get("_source", {})
            rows.append(
                {
                    "query_id": row.query_id,
                    "query": row.query,
                    "rank": rank,
                    "product_id": hit["_id"],
                    "score": hit["_score"],
                    "product_title": source.get("product_title"),
                    "product_brand": source.get("product_brand"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-encode", action="store_true", help="Fail if semantic caches are unavailable instead of encoding missing products.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--semantic-index", default=SEMANTIC_INDEX)
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, names_v3 = load_v3_features()
    all_examples = pd.concat([train, test], ignore_index=True)
    names_v4 = names_v3 + ["semantic_cosine"]

    if args.skip_encode and not (FULL_EMBEDDINGS.exists() or JUDGED_EMBEDDINGS.exists()):
        raise SystemExit("semantic embedding cache is missing; run without --skip-encode to create it")

    query_vectors = query_embedding_map(all_examples, batch_size=args.batch_size)
    product_vectors = judged_product_embedding_map(all_examples, batch_size=args.batch_size)
    train_v4 = add_semantic_feature(train, query_vectors, product_vectors)
    test_v4 = add_semantic_feature(test, query_vectors, product_vectors)

    choice, tuning = select_model(train_v4, names_v4, "v4_semantic")
    valid = train_v4[train_v4.is_validation].copy()
    valid["v4"] = choice["model"].predict(valid[names_v4].to_numpy(np.float32), output_margin=True)
    final = fit_final(train_v4, names_v4, choice)

    test_v4 = add_v3_predictions(test_v4, names_v3)
    test_v4["v4"] = final.predict(test_v4[names_v4].to_numpy(np.float32), output_margin=True)

    test_v3_ndcg = per_query_ndcg(test_v4, "v3")
    test_v4_ndcg = per_query_ndcg(test_v4, "v4")
    validation_v4 = per_query_ndcg(valid, "v4")
    cheap, cheap_rows = fit_cheap_score(valid)
    smoke = smoke_hybrid_queries(test_v4, query_vectors, args.semantic_index, limit=10)

    metrics = {
        "model_name": MODEL_NAME,
        "feature_count_v3": len(names_v3),
        "feature_count_v4": len(names_v4),
        "semantic_feature": "semantic_cosine = dot(E5 query embedding, E5 product embedding)",
        "fit_queries": int(train_v4.loc[~train_v4.is_validation, "query_id"].nunique()),
        "validation_queries": int(valid.query_id.nunique()),
        "selected_config": choice["config"],
        "selected_trees": int(choice["trees"]),
        "validation_v4_official_nDCG": float(validation_v4.mean()),
        "test_v3_official_nDCG": float(test_v3_ndcg.mean()),
        "test_v4_official_nDCG": float(test_v4_ndcg.mean()),
        "test_v4_vs_v3": paired(test_v3_ndcg, test_v4_ndcg),
        "cheap_score_selected": cheap,
        "semantic_index": args.semantic_index,
        "semantic_index_ready": semantic_index_ready(args.semantic_index),
        "notes": [
            "v4 is evaluated as an external reranker because OpenSearch LTR templates stringify dynamic vector parameters.",
            "hybrid_candidate_body uses bool.should lexical+kNN and is not RRF.",
            "cheap_score coefficients are fitted on validation queries to approximate query-normalized v4 rank.",
        ],
    }

    importance = pd.DataFrame({"feature": names_v4, "gain": final.feature_importances_}).sort_values("gain", ascending=False)
    tuning.to_csv(ARTIFACT_DIR / "validation_tuning.csv", index=False)
    cheap_rows.to_csv(ARTIFACT_DIR / "cheap_score_regression.csv", index=False)
    importance.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
    test_v4[["query_id", "query", "product_id", "esci_label", "v3", "v4", "semantic_cosine", "lexical_v2"]].to_csv(
        ARTIFACT_DIR / "test_predictions.csv",
        index=False,
    )
    if not smoke.empty:
        smoke.to_csv(ARTIFACT_DIR / "hybrid_smoke_top10.csv", index=False)
    final.get_booster().feature_names = names_v4
    final.save_model(ARTIFACT_DIR / "xgboost_model_v4_external_semantic.json")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(importance.head(20).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
