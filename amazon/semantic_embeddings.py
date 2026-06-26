"""Utilities for ESCI semantic retrieval experiments.

This module intentionally keeps the semantic index separate from amazon-jp-v2.
It can be run in a small pilot mode first and resumed for full-corpus ingestion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
import torch


BASE_URL = "http://localhost:9200"
SOURCE_INDEX = "amazon-jp-v2"
SEMANTIC_INDEX = "amazon-jp-semantic-v1"
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "esci-data/shopping_queries_dataset"
ARTIFACT_DIR = HERE / "artifacts/semantic_hybrid"
MODEL_NAME = "intfloat/multilingual-e5-base"
DIMENSION = 768
MAX_SEQ_LENGTH = 128
PRODUCT_BULLET_CHARS = 120


def check(response: requests.Response) -> requests.Response:
    if not response.ok:
        raise RuntimeError(f"OpenSearch {response.status_code}: {response.text[:2000]}")
    return response


def product_passage(row: pd.Series, bullet_chars: int = PRODUCT_BULLET_CHARS) -> str:
    parts = []
    if pd.notna(row.get("product_title")):
        parts.append(str(row.product_title))
    if pd.notna(row.get("product_brand")):
        parts.append(f"brand: {row.product_brand}")
    if pd.notna(row.get("product_color")):
        parts.append(f"color: {row.product_color}")
    if pd.notna(row.get("product_bullet_point")):
        bullet = str(row.product_bullet_point).replace("\n", " ")[:bullet_chars]
        if bullet:
            parts.append(f"bullet: {bullet}")
    return "passage: " + " | ".join(parts)


def load_products(limit: int | None = None) -> pd.DataFrame:
    columns = [
        "product_id",
        "product_title",
        "product_description",
        "product_bullet_point",
        "product_brand",
        "product_color",
        "product_locale",
    ]
    products = pd.read_parquet(
        DATA_DIR / "shopping_queries_dataset_products.parquet",
        columns=columns,
        filters=[("product_locale", "=", "jp")],
    ).sort_values("product_id")
    if limit:
        products = products.head(limit)
    return products.reset_index(drop=True)


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS device was requested but torch.backends.mps.is_available() is false")
    if device not in {"cpu", "mps"}:
        raise ValueError(f"unsupported device: {device}")
    return device


def load_model(device: str = "auto", max_seq_length: int = MAX_SEQ_LENGTH) -> SentenceTransformer:
    resolved = resolve_device(device)
    print(f"loading {MODEL_NAME} on device={resolved}, max_seq_length={max_seq_length}", flush=True)
    model = SentenceTransformer(MODEL_NAME, device=resolved)
    model.max_seq_length = max_seq_length
    return model


def embedding_paths(limit: int | None) -> tuple[Path, Path]:
    suffix = f"pilot_{limit}" if limit else "full"
    return ARTIFACT_DIR / f"product_embeddings_{suffix}.npy", ARTIFACT_DIR / f"product_ids_{suffix}.json"


def _resumable_paths(limit: int | None) -> tuple[Path, Path]:
    emb_path, _ = embedding_paths(limit)
    return emb_path.with_suffix(".tmp.npy"), emb_path.with_suffix(".progress.json")


def _ids_digest(product_ids: list[str]) -> str:
    joined = "\n".join(product_ids).encode()
    return hashlib.sha1(joined).hexdigest()


def encode_products(
    limit: int | None = None,
    batch_size: int = 64,
    device: str = "auto",
    encode_chunk_size: int = 2048,
    max_seq_length: int = MAX_SEQ_LENGTH,
    bullet_chars: int = PRODUCT_BULLET_CHARS,
) -> tuple[Path, Path]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    emb_path, ids_path = embedding_paths(limit)
    if emb_path.exists() and ids_path.exists():
        print(f"embedding cache exists: {emb_path}")
        return emb_path, ids_path
    products = load_products(limit)
    product_ids = products.product_id.astype(str).tolist()
    product_ids_digest = _ids_digest(product_ids)
    config = {"dimension": DIMENSION, "max_seq_length": max_seq_length, "bullet_chars": bullet_chars}
    ids_path.write_text(json.dumps(product_ids, ensure_ascii=False))
    model = load_model(device, max_seq_length=max_seq_length)
    texts = [product_passage(row, bullet_chars=bullet_chars) for _, row in products.iterrows()]
    tmp_path, progress_path = _resumable_paths(limit)
    start = 0
    if tmp_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("product_ids_digest") == product_ids_digest and progress.get("config") == config:
            start = int(progress.get("completed", 0))
            embeddings = np.lib.format.open_memmap(tmp_path, mode="r+", dtype="float32", shape=(len(products), DIMENSION))
            print(f"resuming embedding cache: {tmp_path} from {start:,}/{len(products):,}")
        else:
            tmp_path.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)
            embeddings = np.lib.format.open_memmap(tmp_path, mode="w+", dtype="float32", shape=(len(products), DIMENSION))
    else:
        embeddings = np.lib.format.open_memmap(tmp_path, mode="w+", dtype="float32", shape=(len(products), DIMENSION))
    progress_path.write_text(json.dumps({"completed": start, "config": config, "product_ids_digest": product_ids_digest}, ensure_ascii=False))
    started = time.time()
    encode_chunk_size = max(batch_size, encode_chunk_size)
    for chunk_start in range(start, len(products), encode_chunk_size):
        chunk_end = min(chunk_start + encode_chunk_size, len(products))
        chunk = model.encode(
            texts[chunk_start:chunk_end],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")
        embeddings[chunk_start:chunk_end] = chunk
        embeddings.flush()
        progress_path.write_text(json.dumps({"completed": chunk_end, "config": config, "product_ids_digest": product_ids_digest}, ensure_ascii=False))
        if chunk_start == start or chunk_end == len(products) or (chunk_end // encode_chunk_size) % 5 == 0:
            elapsed = time.time() - started
            rate = (chunk_end - start) / elapsed if elapsed else 0.0
            remaining = (len(products) - chunk_end) / rate if rate else math.inf
            print(f"encoded {chunk_end:,}/{len(products):,} products; eta {remaining/60:.1f} min", flush=True)
    del embeddings
    tmp_path.replace(emb_path)
    progress_path.unlink(missing_ok=True)
    print(f"encoded {len(products):,} products in {time.time() - started:.1f}s -> {emb_path}")
    return emb_path, ids_path


def source_settings() -> dict:
    settings = check(requests.get(f"{BASE_URL}/{SOURCE_INDEX}/_settings?include_defaults=false")).json()[SOURCE_INDEX]["settings"]["index"]
    return {
        "index": {
            "knn": True,
            "number_of_shards": "1",
            "number_of_replicas": "0",
            "analysis": settings["analysis"],
        }
    }


def semantic_mapping() -> dict:
    source = check(requests.get(f"{BASE_URL}/{SOURCE_INDEX}/_mapping")).json()[SOURCE_INDEX]["mappings"]
    mapping = json.loads(json.dumps(source))
    mapping["properties"]["product_embedding"] = {
        "type": "knn_vector",
        "dimension": DIMENSION,
        "method": {
            "name": "hnsw",
            "space_type": "cosinesimil",
            "engine": "lucene",
            "parameters": {"ef_construction": 128, "m": 24},
        },
    }
    return mapping


def create_index(index: str = SEMANTIC_INDEX, recreate: bool = False) -> None:
    exists = requests.head(f"{BASE_URL}/{index}").status_code == 200
    if exists and recreate:
        check(requests.delete(f"{BASE_URL}/{index}"))
        exists = False
    if exists:
        print(f"{index} already exists")
        return
    payload = {"settings": source_settings(), "mappings": semantic_mapping()}
    check(requests.put(f"{BASE_URL}/{index}", json=payload))
    print(f"created {index}")


def bulk_lines(products: pd.DataFrame, embeddings: np.ndarray, index: str) -> Iterable[str]:
    for row, vector in zip(products.itertuples(index=False), embeddings, strict=True):
        product_id = str(row.product_id)
        yield json.dumps({"index": {"_index": index, "_id": product_id}})
        doc = {
            "product_id": product_id,
            "product_title": None if pd.isna(row.product_title) else row.product_title,
            "product_description": None if pd.isna(row.product_description) else row.product_description,
            "product_bullet_point": None if pd.isna(row.product_bullet_point) else row.product_bullet_point,
            "product_brand": None if pd.isna(row.product_brand) else row.product_brand,
            "product_color": None if pd.isna(row.product_color) else row.product_color,
            "product_locale": row.product_locale,
            "product_embedding": vector.astype(float).tolist(),
        }
        yield json.dumps(doc, ensure_ascii=False)


def ingest(limit: int | None = None, index: str = SEMANTIC_INDEX, batch_docs: int = 500, device: str = "auto") -> None:
    emb_path, ids_path = embedding_paths(limit)
    if not emb_path.exists():
        encode_products(limit, device=device)
    products = load_products(limit)
    embeddings = np.load(emb_path)
    if len(products) != len(embeddings):
        raise AssertionError("product and embedding counts differ")
    total = len(products)
    started = time.time()
    for start in range(0, total, batch_docs):
        end = min(start + batch_docs, total)
        lines = list(bulk_lines(products.iloc[start:end], embeddings[start:end], index))
        response = check(
            requests.post(
                f"{BASE_URL}/_bulk",
                data="\n".join(lines) + "\n",
                headers={"Content-Type": "application/x-ndjson"},
                timeout=180,
            )
        ).json()
        if response.get("errors"):
            first_error = next(item for item in response["items"] if item["index"].get("error"))
            raise RuntimeError(json.dumps(first_error, ensure_ascii=False)[:2000])
        if start == 0 or end == total or (start // batch_docs + 1) % 20 == 0:
            print(f"ingested {end:,}/{total:,} docs", flush=True)
    check(requests.post(f"{BASE_URL}/{index}/_refresh"))
    print(f"ingested {total:,} docs in {time.time() - started:.1f}s")


def encode_queries(
    queries: list[str],
    batch_size: int = 64,
    device: str = "auto",
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> np.ndarray:
    model = load_model(device, max_seq_length=max_seq_length)
    texts = [f"query: {query}" for query in queries]
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True).astype("float32")


def smoke(index: str = SEMANTIC_INDEX, device: str = "auto", max_seq_length: int = MAX_SEQ_LENGTH) -> None:
    vector = encode_queries(["自転車スピーカー"], device=device, max_seq_length=max_seq_length)[0].astype(float).tolist()
    body = {
        "size": 5,
        "_source": ["product_id", "product_title", "product_brand"],
        "query": {"knn": {"product_embedding": {"vector": vector, "k": 5}}},
    }
    result = check(requests.post(f"{BASE_URL}/{index}/_search", json=body, timeout=30)).json()
    print(json.dumps(result["hits"]["hits"], ensure_ascii=False, indent=2)[:4000])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["encode", "create-index", "ingest", "smoke", "all"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--index", default=SEMANTIC_INDEX)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-chunk-size", type=int, default=2048)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--bullet-chars", type=int, default=PRODUCT_BULLET_CHARS)
    parser.add_argument("--bulk-size", type=int, default=500)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    if args.command == "encode":
        encode_products(args.limit, args.batch_size, args.device, args.encode_chunk_size, args.max_seq_length, args.bullet_chars)
    elif args.command == "create-index":
        create_index(args.index, args.recreate)
    elif args.command == "ingest":
        ingest(args.limit, args.index, args.bulk_size, args.device)
    elif args.command == "smoke":
        smoke(args.index, args.device, args.max_seq_length)
    elif args.command == "all":
        encode_products(args.limit, args.batch_size, args.device, args.encode_chunk_size, args.max_seq_length, args.bullet_chars)
        create_index(args.index, args.recreate)
        ingest(args.limit, args.index, args.bulk_size, args.device)
        smoke(args.index, args.device, args.max_seq_length)


if __name__ == "__main__":
    main()
