"""Train and validate v3 structured/query-intent features for OpenSearch LTR."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

from ltr_features import EPSILON, create_model_payload, feature_set_v3, query_params


BASE_URL = "http://localhost:9200"
INDEX = "amazon-jp-v2"
FEATURESET = "esci_jp_features_v3"
MODEL = "esci_jp_xgb_v3"
V2_MODEL = "esci_jp_xgb_v2"
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "esci-data/shopping_queries_dataset"
ARTIFACT_DIR = HERE / "artifacts/ltr_xgboost_v3"
GAINS = {"I": 0.0, "S": 0.01, "C": 0.1, "E": 1.0}
LABELS = {"I": 0, "S": 1, "C": 2, "E": 3}


def check(response: requests.Response) -> requests.Response:
    if not response.ok:
        raise RuntimeError(f"OpenSearch {response.status_code}: {response.text[:2000]}")
    return response


def validation_query(query_id: object) -> bool:
    return int(hashlib.sha1(str(query_id).encode()).hexdigest(), 16) % 5 == 0


def feature_names() -> list[str]:
    return [item["name"] for item in feature_set_v3(FEATURESET)["featureset"]["features"]]


def register_featureset() -> None:
    if requests.get(f"{BASE_URL}/_ltr/_featureset/{FEATURESET}").status_code != 200:
        check(requests.post(f"{BASE_URL}/_ltr/_featureset/{FEATURESET}", json=feature_set_v3(FEATURESET)))


def load_small_examples() -> pd.DataFrame:
    columns = ["example_id", "query_id", "query", "product_id", "esci_label", "split"]
    frame = pd.read_parquet(
        DATA_DIR / "shopping_queries_dataset_examples.parquet",
        columns=columns + ["product_locale", "small_version"],
        filters=[("product_locale", "=", "jp"), ("small_version", "=", 1)],
    )[columns]
    frame["is_validation"] = frame.query_id.map(validation_query)
    return frame.sort_values(["split", "query_id", "example_id"]).reset_index(drop=True)


def search_body(group: pd.DataFrame) -> dict:
    ids = group.product_id.astype(str).tolist()
    return {
        "size": len(ids), "track_total_hits": True, "_source": False,
        "query": {"bool": {
            "must": [{"ids": {"values": ids}}],
            "filter": [{"sltr": {
                "_name": "logged_features", "featureset": FEATURESET,
                "params": query_params(str(group["query"].iloc[0])),
            }}],
        }},
        "ext": {"ltr_log": {"log_specs": {"name": "features", "named_query": "logged_features"}}},
    }


def collect_features(examples: pd.DataFrame, cache: Path, batch_queries: int = 40) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    names = feature_names()
    groups = list(examples.groupby("query_id", sort=True))
    rows = []
    started = time.time()
    for start in range(0, len(groups), batch_queries):
        batch = groups[start:start + batch_queries]
        lines = []
        for _, group in batch:
            lines.extend([json.dumps({"index": INDEX}), json.dumps(search_body(group), ensure_ascii=False)])
        responses = check(requests.post(
            f"{BASE_URL}/_msearch", data="\n".join(lines) + "\n",
            headers={"Content-Type": "application/x-ndjson"}, timeout=180,
        )).json()["responses"]
        for (_, group), response in zip(batch, responses, strict=True):
            if "error" in response:
                raise RuntimeError(json.dumps(response["error"], ensure_ascii=False)[:2000])
            expected = set(group.product_id.astype(str))
            hits = response["hits"]["hits"]
            if {hit["_id"] for hit in hits} != expected:
                raise AssertionError(f"feature rows missing for query_id={group.query_id.iloc[0]}")
            for hit in hits:
                logged = {item["name"]: item.get("value", EPSILON) for item in hit["fields"]["_ltrlog"][0]["features"]}
                rows.append({"query_id": group.query_id.iloc[0], "product_id": hit["_id"], **{name: float(logged[name]) for name in names}})
        if start == 0 or (start // batch_queries + 1) % 25 == 0:
            print(f"v3 feature logging {min(start + len(batch), len(groups)):,}/{len(groups):,}", flush=True)
    result = examples.merge(pd.DataFrame(rows), on=["query_id", "product_id"], validate="one_to_one")
    if len(result) != len(examples) or result[names].isna().any().any():
        raise AssertionError("v3 feature matrix is incomplete")
    result.to_parquet(cache, index=False)
    print(f"v3 feature logging completed: {len(result):,} rows / {time.time()-started:.1f}s", flush=True)
    return result


def ranked_ndcg(group: pd.DataFrame, score: str) -> float:
    ranked = group.sort_values([score, "product_id"], ascending=[False, True])
    gains = ranked.esci_label.map(GAINS).to_numpy()
    discount = np.log2(np.arange(2, len(gains) + 2))
    ideal = np.sort(gains)[::-1]
    denominator = float(np.sum(ideal / discount))
    return float(np.sum(gains / discount) / denominator) if denominator else 0.0


def per_query_ndcg(frame: pd.DataFrame, score: str) -> pd.Series:
    return frame.groupby("query_id", sort=True).apply(lambda group: ranked_ndcg(group, score), include_groups=False)


def paired(baseline: pd.Series, treatment: pd.Series, seed: int = 42) -> dict:
    delta = (treatment - baseline).to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(5000)])
    return {
        "delta": float(delta.mean()), "ci95_low": float(np.quantile(boot, .025)),
        "ci95_high": float(np.quantile(boot, .975)), "wins": int((delta > 1e-12).sum()),
        "ties": int((np.abs(delta) <= 1e-12).sum()), "losses": int((delta < -1e-12).sum()),
    }


def matrix(frame: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort_values(["query_id", "example_id"])
    return (
        ordered[names].to_numpy(np.float32), ordered.esci_label.map(LABELS).to_numpy(np.int32),
        ordered.groupby("query_id", sort=False).size().to_numpy(np.int32),
    )


CONFIGS = [
    ("pairwise_d3", "rank:pairwise", 3), ("pairwise_d5", "rank:pairwise", 5),
    ("ndcg_d3", "rank:ndcg", 3), ("ndcg_d5", "rank:ndcg", 5),
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
            objective=objective, n_estimators=500, learning_rate=.05, max_depth=depth,
            min_child_weight=2, subsample=.9, colsample_bytree=.9, reg_lambda=1,
            tree_method="hist", random_state=42, early_stopping_rounds=40, eval_metric="ndcg",
        )
        model.fit(x_fit, y_fit, group=g_fit, eval_set=[(x_valid, y_valid)], eval_group=[g_valid], verbose=False)
        column = f"prediction_{feature_version}_{config}"
        valid[column] = model.predict(x_valid, output_margin=True)
        score = float(per_query_ndcg(valid, column).mean())
        trees = int(model.best_iteration) + 1
        record = {"feature_version": feature_version, "config": config, "objective": objective, "max_depth": depth, "trees": trees, "validation_official_nDCG": score}
        rows.append(record); print(record, flush=True)
        candidate = {"score": score, "config": config, "objective": objective, "depth": depth, "trees": trees, "model": model, "valid": valid, "column": column}
        if best is None or score > best["score"]:
            best = candidate
    assert best is not None
    return best, pd.DataFrame(rows)


def fit_final(train: pd.DataFrame, names: list[str], choice: dict) -> xgb.XGBRanker:
    x_all, y_all, g_all = matrix(train, names)
    model = xgb.XGBRanker(
        objective=choice["objective"], n_estimators=choice["trees"], learning_rate=.05,
        max_depth=choice["depth"], min_child_weight=2, subsample=.9,
        colsample_bytree=.9, reg_lambda=1, tree_method="hist", random_state=42,
    )
    model.fit(x_all, y_all, group=g_all, verbose=False)
    return model


def deploy(model: xgb.XGBRanker, names: list[str], sample: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    booster = model.get_booster(); booster.feature_names = names
    payload, offset = create_model_payload(MODEL, FEATURESET, booster)
    requests.delete(f"{BASE_URL}/_ltr/_model/{MODEL}")
    check(requests.post(f"{BASE_URL}/_ltr/_featureset/{FEATURESET}/_createmodel", json=payload))
    sample = sample.sort_values(["query_id", "example_id"])
    sample = sample[sample.query_id.isin(sample.query_id.drop_duplicates().head(3))].copy()
    sample["python_score"] = model.predict(sample[names].to_numpy(np.float32), output_margin=True) + offset
    actual = {}
    for _, group in sample.groupby("query_id", sort=True):
        ids = group.product_id.astype(str).tolist()
        body = {"size": len(ids), "query": {"ids": {"values": ids}}, "rescore": {
            "window_size": len(ids), "query": {
                "rescore_query": {"sltr": {"model": MODEL, "params": query_params(group["query"].iloc[0])}},
                "query_weight": 0.0, "rescore_query_weight": 1.0,
            },
        }}
        hits = check(requests.post(f"{BASE_URL}/{INDEX}/_search", json=body)).json()["hits"]["hits"]
        actual.update({hit["_id"]: hit["_score"] for hit in hits})
    sample["opensearch_score"] = sample.product_id.map(actual)
    sample["abs_diff"] = (sample.python_score - sample.opensearch_score).abs()
    return offset, sample


def render(value: object, params: dict) -> object:
    if isinstance(value, dict): return {key: render(item, params) for key, item in value.items()}
    if isinstance(value, list): return [render(item, params) for item in value]
    if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
        return params[value[2:-2]]
    return value


def end_to_end(examples: pd.DataFrame, include_v2: bool = True, batch_queries: int = 30) -> pd.DataFrame:
    features = feature_set_v3(FEATURESET)["featureset"]["features"]
    lexical_dense = next(item["template"] for item in features if item["name"] == "lexical_v2")
    lexical_template = lexical_dense["bool"]["should"][0]
    groups = list(examples.groupby("query_id", sort=True)); rows = []
    for start in range(0, len(groups), batch_queries):
        batch = groups[start:start + batch_queries]; lines = []; metadata = []
        for qid, group in batch:
            params = query_params(group["query"].iloc[0]); lexical = render(lexical_template, params)
            base = {"size": 100, "_source": False, "query": lexical}
            variants = [("baseline", base)]
            if include_v2:
                variants.append(("v2", {**base, "rescore": {"window_size": 100, "query": {"rescore_query": {"sltr": {"model": V2_MODEL, "params": {"keywords": params["keywords"]}}}, "query_weight": 0.0, "rescore_query_weight": 1.0}}}))
            variants.append(("v3", {**base, "rescore": {"window_size": 100, "query": {"rescore_query": {"sltr": {"model": MODEL, "params": params}}, "query_weight": 0.0, "rescore_query_weight": 1.0}}}))
            for variant, body in variants:
                lines.extend([json.dumps({"index": INDEX}), json.dumps(body, ensure_ascii=False)]); metadata.append((qid, variant, group))
        responses = check(requests.post(f"{BASE_URL}/_msearch", data="\n".join(lines) + "\n", headers={"Content-Type": "application/x-ndjson"}, timeout=180)).json()["responses"]
        for (qid, variant, group), response in zip(metadata, responses, strict=True):
            if "error" in response: raise RuntimeError(json.dumps(response["error"], ensure_ascii=False)[:2000])
            run = [hit["_id"] for hit in response["hits"]["hits"]]
            qrels = dict(zip(group.product_id.astype(str), group.esci_label.map(GAINS)))
            gains = np.array([qrels.get(pid, 0.0) for pid in run]); ideal = np.sort(np.array(list(qrels.values())))[::-1]
            def metric(k: int) -> float:
                actual = gains[:k]; target = ideal[:k]
                dcg = float(np.sum(actual / np.log2(np.arange(2, len(actual) + 2))))
                denom = float(np.sum(target / np.log2(np.arange(2, len(target) + 2))))
                return dcg / denom if denom else 0.0
            positive = {pid for pid, gain in qrels.items() if gain > 0}
            rows.append({"query_id": qid, "variant": variant, "nDCG_at_10_unjudged0": metric(10), "nDCG_at_100_unjudged0": metric(100), "recall_at_100_judged_positive": len(set(run) & positive) / len(positive), "judged_at_10": len(set(run[:10]) & set(qrels)) / 10})
        if start == 0 or (start // batch_queries + 1) % 25 == 0:
            print(f"end-to-end {min(start + len(batch), len(groups)):,}/{len(groups):,}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True); register_featureset()
    examples = load_small_examples()
    train = collect_features(examples[examples.split == "train"].copy(), ARTIFACT_DIR / "train_features_v3.parquet")
    test = collect_features(examples[examples.split == "test"].copy(), ARTIFACT_DIR / "test_features_v3.parquet")
    train["is_validation"] = train.query_id.map(validation_query)
    all_names = feature_names(); v2_names = all_names[:11]
    v2_choice, v2_tuning = select_model(train, v2_names, "v2_features")
    v3_choice, v3_tuning = select_model(train, all_names, "v3_features")
    tuning = pd.concat([v2_tuning, v3_tuning], ignore_index=True)
    valid = train[train.is_validation].copy()
    valid["v2"] = v2_choice["model"].predict(valid[v2_names].to_numpy(np.float32), output_margin=True)
    valid["v3"] = v3_choice["model"].predict(valid[all_names].to_numpy(np.float32), output_margin=True)
    final = fit_final(train, all_names, v3_choice)
    v2_booster = xgb.Booster(); v2_booster.load_model(HERE / "artifacts/ltr_xgboost/xgboost_model.json")
    test["v2"] = v2_booster.predict(xgb.DMatrix(test[v2_names], feature_names=v2_names), output_margin=True)
    test["v3"] = final.predict(test[all_names].to_numpy(np.float32), output_margin=True)
    offset, parity = deploy(final, all_names, test)
    e2e = end_to_end(test)
    validation_v2 = per_query_ndcg(valid, "v2"); validation_v3 = per_query_ndcg(valid, "v3")
    test_v2 = per_query_ndcg(test, "v2"); test_v3 = per_query_ndcg(test, "v3")
    metrics = {
        "feature_count_v2": len(v2_names), "feature_count_v3": len(all_names),
        "fit_queries": int(train.loc[~train.is_validation, "query_id"].nunique()), "validation_queries": int(valid.query_id.nunique()),
        "validation_v2_official_nDCG": float(validation_v2.mean()), "validation_v3_official_nDCG": float(validation_v3.mean()),
        "validation_v3_vs_v2": paired(validation_v2, validation_v3),
        "selected_config": v3_choice["config"], "selected_trees": v3_choice["trees"],
        "test_v2_official_nDCG": float(test_v2.mean()), "test_v3_official_nDCG": float(test_v3.mean()), "test_v3_vs_v2": paired(test_v2, test_v3),
        "model_score_offset": offset, "parity_max_abs_diff": float(parity.abs_diff.max()),
        "parity_rank_equal": bool(parity.sort_values(["query_id", "python_score"], ascending=[True, False]).product_id.tolist() == parity.sort_values(["query_id", "opensearch_score"], ascending=[True, False]).product_id.tolist()),
    }
    for column in ["nDCG_at_10_unjudged0", "nDCG_at_100_unjudged0", "recall_at_100_judged_positive", "judged_at_10"]:
        table = e2e.pivot(index="query_id", columns="variant", values=column)
        for variant in ["baseline", "v2", "v3"]: metrics[f"end_to_end_{variant}_{column}"] = float(table[variant].mean())
        if column.startswith("nDCG"): metrics[f"end_to_end_v3_vs_v2_{column}"] = paired(table.v2, table.v3)
    importance = pd.DataFrame({"feature": all_names, "gain": final.feature_importances_}).sort_values("gain", ascending=False)
    tuning.to_csv(ARTIFACT_DIR / "validation_tuning.csv", index=False); importance.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
    parity.to_csv(ARTIFACT_DIR / "parity_check.csv", index=False); e2e.to_csv(ARTIFACT_DIR / "end_to_end_test.csv", index=False)
    final.get_booster().feature_names = all_names; final.save_model(ARTIFACT_DIR / "xgboost_model.json")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2)); print(importance.head(20).to_string(index=False))


DIAGNOSTIC_QUERY_TYPES = {
    82972: "属性・否定条件", 92648: "typo/未知ブランド", 51965: "ブランド＋カテゴリ",
    119872: "明確カテゴリ", 123957: "日本語複合語", 125063: "広いカテゴリ",
    129196: "複合意図・カテゴリ境界", 58451: "長文・語彙差", 120267: "ブランド＋適合",
    127228: "作品＋商品タイプ", 122801: "曖昧語", 124760: "作品名",
    123338: "用途・課題解決", 129890: "カテゴリ＋成分", 54841: "型番＋アクセサリ",
    117598: "用途＋カテゴリ＋対象", 122259: "多義語", 116: "記号ブランド",
    114017: "作品名＋型番", 124889: "専門カテゴリ", 125672: "固有商品名",
    126907: "作品タイトル", 129244: "人物・作品entity", 127927: "短いニッチカテゴリ",
    121931: "広い素材・食品", 102534: "ブランド表記揺れ", 122508: "ブランド＋交換品",
    123455: "高級ブランド＋カテゴリ", 7680: "非常に長いquery", 41495: "英語の広いカテゴリ",
}


def diagnostic_main() -> None:
    columns = ["example_id", "query_id", "query", "product_id", "esci_label", "split", "small_version"]
    examples = pd.read_parquet(
        DATA_DIR / "shopping_queries_dataset_examples.parquet", columns=columns + ["product_locale"],
        filters=[("product_locale", "=", "jp")],
    )[columns]
    examples = examples[examples.query_id.isin(DIAGNOSTIC_QUERY_TYPES)].sort_values(["query_id", "example_id"])
    result = end_to_end(examples, include_v2=True, batch_queries=10)
    wide = result.pivot(index="query_id", columns="variant", values="nDCG_at_10_unjudged0").reset_index()
    metadata = examples.groupby("query_id", as_index=False).agg(query=("query", "first"), small_version=("small_version", "first"))
    wide = metadata.merge(wide, on="query_id")
    wide["query_type"] = wide.query_id.map(DIAGNOSTIC_QUERY_TYPES)
    wide["training_status"] = np.where(wide.small_version == 1, "small-version学習集合内", "学習対象外")
    wide["delta_v3_v2"] = wide.v3 - wide.v2
    summaries = {}
    for status, group in wide.groupby("training_status"):
        summaries[status] = {
            "queries": len(group), "mean_v2": float(group.v2.mean()), "mean_v3": float(group.v3.mean()),
            "paired": paired(group.set_index("query_id").v2, group.set_index("query_id").v3),
        }
    wide.to_csv(ARTIFACT_DIR / "diagnostic_30.csv", index=False)
    (ARTIFACT_DIR / "diagnostic_30_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(wide.sort_values("delta_v3_v2", ascending=False).to_string(index=False))


if __name__ == "__main__":
    diagnostic_main() if "--diagnostic" in sys.argv else main()
