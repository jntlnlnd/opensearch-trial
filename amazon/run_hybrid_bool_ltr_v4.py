"""Evaluate one-request bool.should lexical+kNN candidate generation + LTR v3.

This is the "B plan" semantic experiment:

    bool.should(lexical_v2, kNN * semantic_boost)
      -> OpenSearch raw _score selects the rescore window
      -> OpenSearch LTR plugin XGBoost v3 reranks that window

Semantic similarity is not used as an XGBoost feature.  It only changes the
candidate set that reaches the existing LTR plugin model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from ltr_features import feature_set_v3, query_params
from run_ltr_xgboost_v3 import BASE_URL, GAINS, MODEL, paired, render, validation_query
from semantic_embeddings import SEMANTIC_INDEX, encode_queries


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "esci-data/shopping_queries_dataset"
ARTIFACT_DIR = HERE / "artifacts/hybrid_bool_ltr_v4"


def check(response: requests.Response) -> requests.Response:
    if not response.ok:
        raise RuntimeError(f"OpenSearch {response.status_code}: {response.text[:3000]}")
    return response


def load_examples() -> pd.DataFrame:
    columns = ["example_id", "query_id", "query", "product_id", "esci_label", "split", "small_version"]
    examples = pd.read_parquet(
        DATA_DIR / "shopping_queries_dataset_examples.parquet",
        columns=columns + ["product_locale"],
        filters=[("product_locale", "=", "jp")],
    )[columns]
    examples["is_validation"] = examples.query_id.map(validation_query)
    return examples.sort_values(["split", "query_id", "example_id"]).reset_index(drop=True)


def lexical_query(query: str) -> dict[str, Any]:
    features = feature_set_v3("esci_jp_features_v3")["featureset"]["features"]
    lexical_dense = next(item["template"] for item in features if item["name"] == "lexical_v2")
    lexical_template = lexical_dense["bool"]["should"][0]
    return render(lexical_template, query_params(query))


def search_body(
    query: str,
    query_vector: np.ndarray | None,
    boost: float,
    size: int,
    window_size: int,
    knn_k: int,
    use_ltr: bool,
) -> dict[str, Any]:
    lexical = lexical_query(query)
    if query_vector is None or boost <= 0:
        query_clause = lexical
    else:
        query_clause = {
            "bool": {
                "should": [
                    lexical,
                    {
                        "knn": {
                            "product_embedding": {
                                "vector": query_vector.astype(float).tolist(),
                                "k": knn_k,
                                "boost": boost,
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }
    body: dict[str, Any] = {
        "size": size,
        "_source": False,
        "query": query_clause,
    }
    if use_ltr:
        body["rescore"] = {
            "window_size": window_size,
            "query": {
                "rescore_query": {
                    "sltr": {
                        "model": MODEL,
                        "params": query_params(query),
                    }
                },
                "query_weight": 0.0,
                "rescore_query_weight": 1.0,
            },
        }
    return body


def metric_for_run(run: list[str], qrels: dict[str, float], k: int) -> float:
    actual_ids = run[:k]
    gains = np.array([qrels.get(pid, 0.0) for pid in actual_ids], dtype=np.float64)
    ideal = np.sort(np.array(list(qrels.values()), dtype=np.float64))[::-1][:k]
    dcg = float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))
    denom = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / denom if denom else 0.0


def recall_for_run(run: list[str], qrels: dict[str, float], k: int) -> float:
    positives = {pid for pid, gain in qrels.items() if gain > 0}
    if not positives:
        return 0.0
    return len(set(run[:k]) & positives) / len(positives)


def judged_fraction(run: list[str], qrels: dict[str, float], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(run[:k]) & set(qrels)) / k


def query_embedding_map(
    groups: list[tuple[Any, pd.DataFrame]],
    batch_size: int,
    device: str,
    max_seq_length: int,
) -> dict[str, np.ndarray]:
    query_table = pd.DataFrame(
        [{"query_id": qid, "query": group["query"].iloc[0]} for qid, group in groups]
    ).sort_values("query_id")
    vectors = encode_queries(
        query_table["query"].astype(str).tolist(),
        batch_size=batch_size,
        device=device,
        max_seq_length=max_seq_length,
    )
    return dict(zip(query_table.query_id.astype(str), vectors, strict=True))


def evaluate(
    examples: pd.DataFrame,
    boosts: list[float],
    index: str,
    size: int,
    window_size: int,
    knn_k: int,
    batch_queries: int,
    query_batch_size: int,
    device: str,
    query_max_seq_length: int,
    include_prerescore: bool,
) -> pd.DataFrame:
    groups = list(examples.groupby("query_id", sort=True))
    vectors = query_embedding_map(groups, query_batch_size, device, query_max_seq_length)
    rows = []
    started = time.time()
    for start in range(0, len(groups), batch_queries):
        batch = groups[start:start + batch_queries]
        lines: list[str] = []
        metadata: list[tuple[Any, str, float, pd.DataFrame]] = []
        for qid, group in batch:
            query = str(group["query"].iloc[0])
            vector = vectors[str(qid)]
            variants: list[tuple[str, float, bool]] = []
            for boost in boosts:
                variants.append(("ltr", boost, True))
                if include_prerescore:
                    variants.append(("candidate", boost, False))
            for stage, boost, use_ltr in variants:
                lines.append(json.dumps({"index": index}))
                lines.append(json.dumps(
                    search_body(query, vector if boost > 0 else None, boost, size, window_size, knn_k, use_ltr),
                    ensure_ascii=False,
                ))
                metadata.append((qid, stage, boost, group))
        responses = check(
            requests.post(
                f"{BASE_URL}/_msearch",
                data="\n".join(lines) + "\n",
                headers={"Content-Type": "application/x-ndjson"},
                timeout=240,
            )
        ).json()["responses"]
        for (qid, stage, boost, group), response in zip(metadata, responses, strict=True):
            if "error" in response:
                raise RuntimeError(json.dumps(response["error"], ensure_ascii=False)[:3000])
            run = [hit["_id"] for hit in response["hits"]["hits"]]
            qrels = dict(zip(group.product_id.astype(str), group.esci_label.map(GAINS), strict=True))
            rows.append(
                {
                    "query_id": qid,
                    "query": group["query"].iloc[0],
                    "stage": stage,
                    "semantic_boost": boost,
                    "nDCG_at_10_unjudged0": metric_for_run(run, qrels, 10),
                    "nDCG_at_100_unjudged0": metric_for_run(run, qrels, min(100, size)),
                    "recall_at_100_judged_positive": recall_for_run(run, qrels, min(100, size)),
                    "recall_at_window_judged_positive": recall_for_run(run, qrels, min(window_size, size)),
                    "judged_at_10": judged_fraction(run, qrels, 10),
                    "hits": len(run),
                    "top_product_ids": " ".join(run[:10]),
                }
            )
        done = min(start + len(batch), len(groups))
        if start == 0 or done == len(groups) or (start // batch_queries + 1) % 10 == 0:
            print(f"evaluated {done:,}/{len(groups):,} queries in {time.time() - started:.1f}s", flush=True)
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, baseline_boost: float = 0.0) -> tuple[pd.DataFrame, dict[str, Any]]:
    metric_columns = [
        "nDCG_at_10_unjudged0",
        "nDCG_at_100_unjudged0",
        "recall_at_100_judged_positive",
        "recall_at_window_judged_positive",
        "judged_at_10",
    ]
    summary = (
        results
        .groupby(["stage", "semantic_boost"], as_index=False)[metric_columns]
        .mean()
        .sort_values(["stage", "semantic_boost"])
    )
    payload: dict[str, Any] = {
        "summary": summary.to_dict(orient="records"),
    }
    ltr = results[results.stage == "ltr"]
    baseline = ltr[ltr.semantic_boost == baseline_boost].set_index("query_id")
    if not baseline.empty:
        comparisons = {}
        for boost, group in ltr.groupby("semantic_boost"):
            if boost == baseline_boost:
                continue
            current = group.set_index("query_id")
            common = baseline.index.intersection(current.index)
            comparisons[str(boost)] = {
                metric: paired(baseline.loc[common, metric], current.loc[common, metric])
                for metric in metric_columns
            }
        payload["ltr_vs_baseline"] = comparisons
    return summary, payload


def sample_deltas(results: pd.DataFrame, best_boost: float, output_dir: Path) -> None:
    ltr = results[results.stage == "ltr"].copy()
    base = ltr[ltr.semantic_boost == 0.0][["query_id", "query", "nDCG_at_10_unjudged0", "top_product_ids"]]
    best = ltr[ltr.semantic_boost == best_boost][["query_id", "nDCG_at_10_unjudged0", "top_product_ids"]]
    if base.empty or best.empty:
        return
    joined = base.merge(best, on="query_id", suffixes=("_baseline", "_hybrid"))
    joined["delta_nDCG_at_10"] = joined.nDCG_at_10_unjudged0_hybrid - joined.nDCG_at_10_unjudged0_baseline
    joined.sort_values("delta_nDCG_at_10", ascending=False).head(50).to_csv(output_dir / "example_improvements.csv", index=False)
    joined.sort_values("delta_nDCG_at_10", ascending=True).head(50).to_csv(output_dir / "example_regressions.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=SEMANTIC_INDEX)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--small-only", action="store_true", help="Evaluate only small_version=1 examples.")
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--boosts", default="0,0.05,0.1,0.2,0.3,0.5,0.8,1.0")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--knn-k", type=int, default=100)
    parser.add_argument("--batch-queries", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--query-max-seq-length", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--include-prerescore", action="store_true")
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    examples = load_examples()
    if args.small_only:
        examples = examples[examples.small_version == 1].copy()
    if args.split == "validation":
        examples = examples[(examples.split == "train") & examples.is_validation].copy()
    elif args.split == "test":
        examples = examples[examples.split == "test"].copy()
    if args.limit_queries:
        keep = examples.query_id.drop_duplicates().head(args.limit_queries)
        examples = examples[examples.query_id.isin(keep)].copy()
    boosts = [float(item) for item in args.boosts.split(",") if item != ""]
    print(
        json.dumps(
            {
                "index": args.index,
                "split": args.split,
                "queries": int(examples.query_id.nunique()),
                "rows": int(len(examples)),
                "boosts": boosts,
                "size": args.size,
                "window_size": args.window_size,
                "knn_k": args.knn_k,
                "query_max_seq_length": args.query_max_seq_length,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    results = evaluate(
        examples=examples,
        boosts=boosts,
        index=args.index,
        size=args.size,
        window_size=args.window_size,
        knn_k=args.knn_k,
        batch_queries=args.batch_queries,
        query_batch_size=args.query_batch_size,
        device=args.device,
        query_max_seq_length=args.query_max_seq_length,
        include_prerescore=args.include_prerescore,
    )
    suffix = f"_{args.output_suffix}" if args.output_suffix else f"_{args.split}"
    result_path = ARTIFACT_DIR / f"per_query_results{suffix}.csv"
    summary_path = ARTIFACT_DIR / f"metrics_by_boost{suffix}.csv"
    metrics_path = ARTIFACT_DIR / f"metrics_summary{suffix}.json"
    results.to_csv(result_path, index=False)
    summary, payload = summarize(results)
    summary.to_csv(summary_path, index=False)
    best_rows = summary[summary.stage == "ltr"].sort_values("nDCG_at_10_unjudged0", ascending=False)
    best_boost = float(best_rows.iloc[0].semantic_boost) if not best_rows.empty else 0.0
    payload.update(
        {
            "best_ltr_boost_by_nDCG_at_10": best_boost,
            "result_path": str(result_path),
            "summary_path": str(summary_path),
        }
    )
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    sample_deltas(results, best_boost, ARTIFACT_DIR)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
