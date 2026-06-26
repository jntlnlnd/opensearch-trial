"""Train, validate, deploy, and parity-check an XGBoost ESCI LTR model."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

from ltr_features import EPSILON, create_model_payload, feature_set


BASE_URL = "http://localhost:9200"
INDEX = "amazon-jp-v2"
FEATURESET = "esci_jp_features_v2"
MODEL = "esci_jp_xgb_v2"
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "esci-data/shopping_queries_dataset"
ARTIFACT_DIR = HERE / "artifacts/ltr_xgboost"
GAINS = {"I": 0.0, "S": 0.01, "C": 0.1, "E": 1.0}
TRAIN_LABELS = {"I": 0, "S": 1, "C": 2, "E": 3}


def check(response: requests.Response) -> requests.Response:
    if not response.ok:
        raise RuntimeError(f"OpenSearch {response.status_code}: {response.text[:2000]}")
    return response


def register_featureset() -> None:
    current = requests.get(f"{BASE_URL}/_ltr/_featureset/{FEATURESET}")
    if current.status_code == 200:
        return
    check(requests.post(f"{BASE_URL}/_ltr/_featureset/{FEATURESET}", json=feature_set(FEATURESET)))


def is_validation_query(query_id: object) -> bool:
    return int(hashlib.sha1(str(query_id).encode()).hexdigest(), 16) % 5 == 0


def load_examples() -> pd.DataFrame:
    columns = ["example_id", "query_id", "query", "product_id", "esci_label", "split"]
    frame = pd.read_parquet(
        DATA_DIR / "shopping_queries_dataset_examples.parquet",
        columns=columns + ["product_locale", "small_version"],
        filters=[("product_locale", "=", "jp"), ("small_version", "=", 1)],
    )[columns]
    frame["is_validation"] = frame.query_id.map(is_validation_query)
    return frame.sort_values(["split", "query_id", "example_id"]).reset_index(drop=True)


def _search_body(group: pd.DataFrame) -> dict:
    ids = group.product_id.astype(str).tolist()
    return {
        "size": len(ids),
        "track_total_hits": True,
        "_source": False,
        "query": {
            "bool": {
                "must": [{"ids": {"values": ids}}],
                "filter": [
                    {
                        "sltr": {
                            "_name": "logged_features",
                            "featureset": FEATURESET,
                            "params": {"keywords": str(group["query"].iloc[0])},
                        }
                    }
                ],
            }
        },
        "ext": {"ltr_log": {"log_specs": {"name": "features", "named_query": "logged_features"}}},
    }


def _parse_response(group: pd.DataFrame, response: dict, feature_names: list[str]) -> list[dict]:
    if "error" in response:
        raise RuntimeError(json.dumps(response["error"], ensure_ascii=False)[:2000])
    hits = response["hits"]["hits"]
    expected = set(group.product_id.astype(str))
    actual = {hit["_id"] for hit in hits}
    if actual != expected:
        raise AssertionError(f"query_id={group.query_id.iloc[0]} missing={sorted(expected-actual)[:5]}")
    rows = []
    for hit in hits:
        values = {item["name"]: item.get("value", EPSILON) for item in hit["fields"]["_ltrlog"][0]["features"]}
        row = {"query_id": group.query_id.iloc[0], "product_id": hit["_id"]}
        row.update({name: float(values[name]) for name in feature_names})
        rows.append(row)
    return rows


def collect_features(examples: pd.DataFrame, cache: Path, batch_queries: int = 50) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    names = [item["name"] for item in feature_set(FEATURESET)["featureset"]["features"]]
    groups = list(examples.groupby("query_id", sort=True))
    rows: list[dict] = []
    started = time.time()
    for start in range(0, len(groups), batch_queries):
        batch = groups[start : start + batch_queries]
        lines = []
        for _, group in batch:
            lines.extend([json.dumps({"index": INDEX}), json.dumps(_search_body(group), ensure_ascii=False)])
        response = check(
            requests.post(
                f"{BASE_URL}/_msearch",
                data="\n".join(lines) + "\n",
                headers={"Content-Type": "application/x-ndjson"},
                timeout=180,
            )
        ).json()
        for (_, group), result in zip(batch, response["responses"], strict=True):
            rows.extend(_parse_response(group, result, names))
        if start == 0 or (start // batch_queries + 1) % 20 == 0:
            print(f"feature logging {min(start+len(batch), len(groups)):,}/{len(groups):,} queries")
    logged = pd.DataFrame(rows)
    result = examples.merge(logged, on=["query_id", "product_id"], how="left", validate="one_to_one")
    if result[names].isna().any().any():
        raise AssertionError("feature logging produced missing rows")
    result.to_parquet(cache, index=False)
    print(f"feature logging completed: {len(result):,} rows, {time.time()-started:.1f}s")
    return result


def ndcg(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    gains = labels[order]
    discount = np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains / discount))
    ideal = float(np.sum(np.sort(labels)[::-1] / discount))
    return dcg / ideal if ideal else 0.0


def mean_ndcg(frame: pd.DataFrame, score_column: str) -> float:
    return float(per_query_ndcg(frame, score_column).mean())


def per_query_ndcg(frame: pd.DataFrame, score_column: str) -> pd.Series:
    return frame.groupby("query_id", sort=True).apply(
        lambda group: ndcg(
            group.sort_values([score_column, "product_id"], ascending=[False, True]).esci_label.map(GAINS).to_numpy(),
            np.arange(len(group), 0, -1, dtype=float),
        ),
        include_groups=False,
    )


def paired_summary(baseline: pd.Series, treatment: pd.Series, seed: int = 42) -> dict:
    delta = (treatment - baseline).to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.empty(5000)
    for i in range(len(boot)):
        boot[i] = rng.choice(delta, size=len(delta), replace=True).mean()
    return {
        "delta": float(delta.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "wins": int((delta > 1e-12).sum()),
        "ties": int((np.abs(delta) <= 1e-12).sum()),
        "losses": int((delta < -1e-12).sum()),
    }


def render_keywords(value: object, keywords: str) -> object:
    if isinstance(value, dict):
        return {key: render_keywords(item, keywords) for key, item in value.items()}
    if isinstance(value, list):
        return [render_keywords(item, keywords) for item in value]
    return keywords if value == "{{keywords}}" else value


def matrix(frame: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort_values(["query_id", "example_id"]).reset_index(drop=True)
    return (
        ordered[names].to_numpy(dtype=np.float32),
        ordered.esci_label.map(TRAIN_LABELS).to_numpy(dtype=np.int32),
        ordered.groupby("query_id", sort=False).size().to_numpy(dtype=np.int32),
    )


def train_and_select(train: pd.DataFrame, names: list[str]) -> tuple[xgb.XGBRanker, pd.DataFrame, int]:
    fit = train[~train.is_validation].copy().sort_values(["query_id", "example_id"])
    valid = train[train.is_validation].copy().sort_values(["query_id", "example_id"])
    x_fit, y_fit, g_fit = matrix(fit, names)
    x_val, y_val, g_val = matrix(valid, names)
    configs = [
        ("pairwise_d3", "rank:pairwise", 3),
        ("pairwise_d5", "rank:pairwise", 5),
        ("ndcg_d3", "rank:ndcg", 3),
        ("ndcg_d5", "rank:ndcg", 5),
    ]
    records = []
    candidates = []
    for name, objective, depth in configs:
        model = xgb.XGBRanker(
            objective=objective,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=depth,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=42,
            early_stopping_rounds=40,
            eval_metric="ndcg",
        )
        model.fit(x_fit, y_fit, group=g_fit, eval_set=[(x_val, y_val)], eval_group=[g_val], verbose=False)
        column = f"prediction_{name}"
        valid[column] = model.predict(x_val, output_margin=True)
        score = mean_ndcg(valid, column)
        iteration = int(model.best_iteration) + 1
        records.append({"config": name, "objective": objective, "max_depth": depth, "trees": iteration, "validation_official_nDCG": score})
        candidates.append((score, name, model, iteration))
        print(records[-1])
    results = pd.DataFrame(records).sort_values("validation_official_nDCG", ascending=False)
    _, best_name, _, best_trees = max(candidates, key=lambda item: item[0])
    best_cfg = next(config for config in configs if config[0] == best_name)
    all_train = train.sort_values(["query_id", "example_id"]).reset_index(drop=True)
    x_all, y_all, g_all = matrix(all_train, names)
    final = xgb.XGBRanker(
        objective=best_cfg[1], n_estimators=best_trees, learning_rate=0.05,
        max_depth=best_cfg[2], min_child_weight=2, subsample=0.9,
        colsample_bytree=0.9, reg_lambda=1.0, tree_method="hist", random_state=42,
    )
    final.fit(x_all, y_all, group=g_all, verbose=False)
    return final, results, best_trees


def deploy_and_check(model: xgb.XGBRanker, test: pd.DataFrame, names: list[str]) -> tuple[float, pd.DataFrame]:
    booster = model.get_booster()
    # Training uses a dense ndarray for speed; restore semantic names before
    # JSON export because the LTR plugin resolves splits against feature names.
    booster.feature_names = names
    payload, offset = create_model_payload(MODEL, FEATURESET, booster)
    requests.delete(f"{BASE_URL}/_ltr/_model/{MODEL}")
    check(requests.post(f"{BASE_URL}/_ltr/_featureset/{FEATURESET}/_createmodel", json=payload))
    sample = test.sort_values(["query_id", "example_id"]).groupby("query_id", sort=True).head(50)
    sample = sample[sample.query_id.isin(sample.query_id.drop_duplicates().head(3))].copy()
    expected = model.predict(sample[names].to_numpy(dtype=np.float32), output_margin=True) + offset
    actual = {}
    for _, group in sample.groupby("query_id", sort=True):
        ids = group.product_id.astype(str).tolist()
        body = {
            "size": len(ids), "query": {"ids": {"values": ids}},
            "rescore": {"window_size": len(ids), "query": {
                "rescore_query": {"sltr": {"model": MODEL, "params": {"keywords": group["query"].iloc[0]}}},
                "query_weight": 0.0, "rescore_query_weight": 1.0,
            }},
        }
        result = check(requests.post(f"{BASE_URL}/{INDEX}/_search", json=body)).json()
        actual.update({hit["_id"]: hit["_score"] for hit in result["hits"]["hits"]})
    sample["python_score"] = expected
    sample["opensearch_score"] = sample.product_id.map(actual)
    sample["abs_diff"] = (sample.python_score - sample.opensearch_score).abs()
    return offset, sample


def end_to_end_test(test: pd.DataFrame, batch_queries: int = 40) -> pd.DataFrame:
    """Compare first-stage content_low against top-100 LTR rescore on the corpus."""
    dense = feature_set(FEATURESET)["featureset"]["features"][-1]["template"]
    lexical_template = dense["bool"]["should"][0]
    groups = list(test.groupby("query_id", sort=True))
    rows = []
    for start in range(0, len(groups), batch_queries):
        batch = groups[start : start + batch_queries]
        lines = []
        metadata = []
        for qid, group in batch:
            query = str(group["query"].iloc[0])
            lexical = render_keywords(lexical_template, query)
            baseline = {"size": 100, "_source": False, "query": lexical}
            reranked = {
                **baseline,
                "rescore": {"window_size": 100, "query": {
                    "rescore_query": {"sltr": {"model": MODEL, "params": {"keywords": query}}},
                    "query_weight": 0.0, "rescore_query_weight": 1.0,
                }},
            }
            for variant, body in (("baseline", baseline), ("xgboost", reranked)):
                lines.extend([json.dumps({"index": INDEX}), json.dumps(body, ensure_ascii=False)])
                metadata.append((qid, variant, group))
        response = check(requests.post(
            f"{BASE_URL}/_msearch", data="\n".join(lines) + "\n",
            headers={"Content-Type": "application/x-ndjson"}, timeout=180,
        )).json()["responses"]
        for (qid, variant, group), result in zip(metadata, response, strict=True):
            if "error" in result:
                raise RuntimeError(json.dumps(result["error"], ensure_ascii=False)[:2000])
            run = [hit["_id"] for hit in result["hits"]["hits"]]
            qrels = dict(zip(group.product_id.astype(str), group.esci_label.map(GAINS)))
            labels = np.array([qrels.get(pid, 0.0) for pid in run], dtype=float)
            discounts = np.log2(np.arange(2, len(labels) + 2))
            dcg = float(np.sum(labels / discounts))
            ideal_labels = np.sort(np.fromiter(qrels.values(), dtype=float))[::-1]
            ideal_discount = np.log2(np.arange(2, len(ideal_labels) + 2))
            ideal = float(np.sum(ideal_labels / ideal_discount))
            top10 = labels[:10]
            dcg10 = float(np.sum(top10 / np.log2(np.arange(2, len(top10) + 2))))
            ideal10_labels = ideal_labels[:10]
            ideal10 = float(np.sum(ideal10_labels / np.log2(np.arange(2, len(ideal10_labels) + 2))))
            positive = {pid for pid, gain in qrels.items() if gain > 0}
            rows.append({
                "query_id": qid, "variant": variant,
                "nDCG_at_100_unjudged0": dcg / ideal if ideal else 0.0,
                "nDCG_at_10_unjudged0": dcg10 / ideal10 if ideal10 else 0.0,
                "recall_at_100_judged_positive": len(set(run) & positive) / len(positive),
                "judged_at_10": len(set(run[:10]) & set(qrels)) / 10.0,
                "hits": len(run),
            })
        if start == 0 or (start // batch_queries + 1) % 20 == 0:
            print(f"end-to-end {min(start+len(batch), len(groups)):,}/{len(groups):,} queries")
    return pd.DataFrame(rows)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    register_featureset()
    examples = load_examples()
    train = collect_features(examples[examples.split == "train"].copy(), ARTIFACT_DIR / "train_features_v2.parquet")
    test = collect_features(examples[examples.split == "test"].copy(), ARTIFACT_DIR / "test_features_v2.parquet")
    # Cached features are reusable across split-policy changes; derive the
    # validation flag from query_id on every run.
    train["is_validation"] = train.query_id.map(is_validation_query)
    names = [item["name"] for item in feature_set(FEATURESET)["featureset"]["features"]]
    valid = train[train.is_validation].copy()
    valid["baseline"] = valid["lexical_v2"]
    baseline_validation = mean_ndcg(valid, "baseline")
    model, tuning, trees = train_and_select(train, names)
    test = test.sort_values(["query_id", "example_id"]).reset_index(drop=True)
    test["baseline"] = test["lexical_v2"]
    test["xgboost"] = model.predict(test[names].to_numpy(dtype=np.float32), output_margin=True)
    metrics = {
        "feature_count": len(names), "train_queries": int(train.query_id.nunique()),
        "fit_queries": int(train.loc[~train.is_validation, "query_id"].nunique()),
        "validation_queries": int(valid.query_id.nunique()), "test_queries": int(test.query_id.nunique()),
        "validation_baseline_official_nDCG": baseline_validation,
        "test_baseline_official_nDCG": mean_ndcg(test, "baseline"),
        "test_xgboost_official_nDCG": mean_ndcg(test, "xgboost"), "selected_trees": trees,
    }
    candidate_baseline = per_query_ndcg(test, "baseline")
    candidate_xgboost = per_query_ndcg(test, "xgboost")
    metrics["candidate_paired"] = paired_summary(candidate_baseline, candidate_xgboost)
    offset, parity = deploy_and_check(model, test, names)
    metrics["model_score_offset"] = offset
    metrics["parity_max_abs_diff"] = float(parity.abs_diff.max())
    metrics["parity_rank_equal"] = bool(
        parity.sort_values(["query_id", "python_score"], ascending=[True, False]).product_id.tolist()
        == parity.sort_values(["query_id", "opensearch_score"], ascending=[True, False]).product_id.tolist()
    )
    end_to_end = end_to_end_test(test)
    e2e = end_to_end.pivot(index="query_id", columns="variant", values="nDCG_at_100_unjudged0")
    metrics["end_to_end_baseline_nDCG_at_100_unjudged0"] = float(e2e.baseline.mean())
    metrics["end_to_end_xgboost_nDCG_at_100_unjudged0"] = float(e2e.xgboost.mean())
    metrics["end_to_end_paired"] = paired_summary(e2e.baseline, e2e.xgboost)
    for column in ["nDCG_at_10_unjudged0", "recall_at_100_judged_positive", "judged_at_10"]:
        summary = end_to_end.pivot(index="query_id", columns="variant", values=column)
        metrics[f"end_to_end_baseline_{column}"] = float(summary.baseline.mean())
        metrics[f"end_to_end_xgboost_{column}"] = float(summary.xgboost.mean())
    importance = pd.DataFrame({"feature": names, "gain": [model.feature_importances_[i] for i in range(len(names))]}).sort_values("gain", ascending=False)
    tuning.to_csv(ARTIFACT_DIR / "validation_tuning.csv", index=False)
    importance.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
    parity.to_csv(ARTIFACT_DIR / "parity_check.csv", index=False)
    end_to_end.to_csv(ARTIFACT_DIR / "end_to_end_test.csv", index=False)
    model.save_model(ARTIFACT_DIR / "xgboost_model.json")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
