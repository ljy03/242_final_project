from __future__ import annotations

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def guess_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="CD: Generate transformer embeddings for IMDb reviews.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="outputs/clean_reviews.csv",
        help="Path to Toby's cleaned CSV (default: outputs/clean_reviews.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )

    # columns (optional)
    parser.add_argument("--text_col", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--split_col", type=str, default=None)

    # embedding settings
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformer model name",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings")

    # optional PCA
    parser.add_argument("--pca_dim", type=int, default=0, help="0 = no PCA; e.g. 50/100")

    args = parser.parse_args()

    root = Path(__file__).resolve().parent  # project root (same level as main.py)
    input_csv = (root / args.input_csv).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Cannot find {input_csv}. Run Toby's pipeline first to generate outputs/clean_reviews.csv."
        )

    # ---------- Load CSV (KEEP ROW ORDER!) ----------
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("clean_reviews.csv is empty.")

    text_col = args.text_col or guess_col(df, ["clean_text", "text", "review", "content"])
    label_col = args.label_col or guess_col(df, ["label", "sentiment", "target", "y"])
    split_col = args.split_col or guess_col(df, ["split", "set", "fold"])

    if text_col is None:
        raise ValueError(f"Cannot infer text column. Available columns: {list(df.columns)}. "
                         f"Please pass --text_col.")

    texts = df[text_col].fillna("").astype(str).tolist()

    # ---------- Embedding ----------
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model_name)
    try:
        model.max_seq_length = args.max_length
    except Exception:
        pass

    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=args.normalize,
    ).astype(np.float32)

    # sanity checks
    if emb.shape[0] != len(df):
        raise RuntimeError(f"Embedding rows {emb.shape[0]} != CSV rows {len(df)} (alignment broken).")
    if np.isnan(emb).any():
        raise RuntimeError("Embeddings contain NaN.")

    emb_path = output_dir / "bert_embeddings.npy"
    np.save(emb_path, emb)

    # ---------- Optional PCA ----------
    pca_info = None
    if args.pca_dim and args.pca_dim > 0:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=args.pca_dim, random_state=42)
        emb_pca = pca.fit_transform(emb).astype(np.float32)

        pca_path = output_dir / f"pca_embeddings_{args.pca_dim}d.npy"
        np.save(pca_path, emb_pca)

        pca_info = {
            "pca_dim": int(args.pca_dim),
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "pca_output_file": pca_path.name,
        }

    # ---------- Meta for reproducibility ----------
    meta = {
        "input_csv": str(input_csv.relative_to(root)),
        "num_rows": int(len(df)),
        "text_col": text_col,
        "label_col": label_col,
        "split_col": split_col,
        "split_value_counts": df[split_col].value_counts(dropna=False).to_dict() if split_col else None,
        "model_name": args.model_name,
        "embedding_dim": int(emb.shape[1]),
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "normalize": bool(args.normalize),
        "output_embedding_file": emb_path.name,
        "dtype": str(emb.dtype),
        "alignment_rule": "row i in bert_embeddings.npy corresponds to row i in outputs/clean_reviews.csv",
        "optional_pca": pca_info,
        "runtime_sec": round(time.time() - t0, 2),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    meta_path = output_dir / "embedding_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n✅ Done!")
    print(f"Saved embeddings: {emb_path}")
    print(f"Saved metadata  : {meta_path}")
    if pca_info:
        print(f"Saved PCA       : {output_dir / pca_info['pca_output_file']}")
        print(f"PCA var sum     : {pca_info['explained_variance_ratio_sum']:.4f}")


if __name__ == "__main__":
    main()