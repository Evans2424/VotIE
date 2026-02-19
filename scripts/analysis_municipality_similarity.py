"""
Municipality Heterogeneity Analysis

Computes intra-municipality vs. inter-municipality document similarity
using TF-IDF cosine similarity to quantify document heterogeneity across
Portuguese municipal meeting minutes.

Usage:
    python scripts/analysis_municipality_similarity.py
    python scripts/analysis_municipality_similarity.py --embeddings  # adds multilingual embedding similarity
    python scripts/analysis_municipality_similarity.py --output results/municipality_similarity.json
"""

import argparse
import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats


DATA_DIR = "data/citilink_spans"
SPLITS = ["train", "dev", "test"]


# ---------------------------------------------------------------------------
# Municipality code → human-readable name (from document_id prefix)
# ---------------------------------------------------------------------------

def load_documents():
    """Load all splits and group segments into documents."""
    doc_texts = defaultdict(list)   # doc_id -> list of segment texts
    doc_muni = {}                   # doc_id -> municipality code

    for split in SPLITS:
        with open(f"{DATA_DIR}/{split}.jsonl") as f:
            for line in f:
                d = json.loads(line)
                doc_id = d["document_id"]
                muni = d["municipality"]
                doc_texts[doc_id].append(d["text"])
                doc_muni[doc_id] = muni

    # Concatenate segments → one string per document
    documents = {doc_id: " ".join(segs) for doc_id, segs in doc_texts.items()}
    return documents, doc_muni


# ---------------------------------------------------------------------------
# TF-IDF similarity
# ---------------------------------------------------------------------------

def compute_tfidf_similarity(documents, doc_muni):
    """Return intra and inter similarity lists (TF-IDF cosine)."""
    doc_ids = sorted(documents.keys())
    texts = [documents[d] for d in doc_ids]
    munis = [doc_muni[d] for d in doc_ids]

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)

    intra, inter = [], []
    n = len(doc_ids)
    for i, j in combinations(range(n), 2):
        sim = sim_matrix[i, j]
        if munis[i] == munis[j]:
            intra.append(sim)
        else:
            inter.append(sim)

    return np.array(intra), np.array(inter), sim_matrix, munis, doc_ids


# ---------------------------------------------------------------------------
# Optional embedding-based similarity
# ---------------------------------------------------------------------------

def compute_embedding_similarity(documents, doc_muni):
    """Return intra and inter similarity lists using multilingual embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  [skip] sentence-transformers not installed. Run: pip install sentence-transformers")
        return None, None

    print("  Loading multilingual sentence encoder (this may take a moment)...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    doc_ids = sorted(documents.keys())
    texts = [documents[d] for d in doc_ids]
    munis = [doc_muni[d] for d in doc_ids]

    # Truncate each document to first 512 words to keep inference fast
    texts_trunc = [" ".join(t.split()[:512]) for t in texts]
    embeddings = model.encode(texts_trunc, batch_size=16, show_progress_bar=True)
    sim_matrix = cosine_similarity(embeddings)

    intra, inter = [], []
    for i, j in combinations(range(len(doc_ids)), 2):
        sim = sim_matrix[i, j]
        if munis[i] == munis[j]:
            intra.append(sim)
        else:
            inter.append(sim)

    return np.array(intra), np.array(inter)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def describe(arr, label):
    return (
        f"  {label}: n={len(arr):5d} | "
        f"mean={arr.mean():.4f} | "
        f"std={arr.std():.4f} | "
        f"median={np.median(arr):.4f} | "
        f"[{arr.min():.4f}, {arr.max():.4f}]"
    )


def _cohens_d(a, b):
    """Cohen's d with properly weighted pooled SD (handles unequal n)."""
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1 - 1) * a.std(ddof=1)**2 + (n2 - 1) * b.std(ddof=1)**2) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / pooled_std)


def compute_stats(intra, inter):
    """Return a dict of statistics for intra/inter similarity arrays."""
    t_stat, p_val = stats.ttest_ind(intra, inter, equal_var=False)
    cohens_d = _cohens_d(intra, inter)
    return {
        "intra": {
            "n": len(intra),
            "mean": float(intra.mean()),
            "std": float(intra.std()),
            "median": float(np.median(intra)),
            "min": float(intra.min()),
            "max": float(intra.max()),
        },
        "inter": {
            "n": len(inter),
            "mean": float(inter.mean()),
            "std": float(inter.std()),
            "median": float(np.median(inter)),
            "min": float(inter.min()),
            "max": float(inter.max()),
        },
        "difference_intra_minus_inter": float(intra.mean() - inter.mean()),
        "welch_t": float(t_stat),
        "welch_p": float(p_val),
        "cohens_d": cohens_d,
    }


def report_stats(intra, inter, method_name):
    print(f"\n{'='*65}")
    print(f"  {method_name} Cosine Similarity")
    print(f"{'='*65}")
    print(describe(intra, "Intra-municipality "))
    print(describe(inter, "Inter-municipality "))

    t_stat, p_val = stats.ttest_ind(intra, inter, equal_var=False)
    cohens_d = _cohens_d(intra, inter)

    print(f"\n  Difference (intra − inter): {intra.mean() - inter.mean():+.4f}")
    print(f"  Welch's t-test: t={t_stat:.3f}, p={p_val:.2e}")
    print(f"  Cohen's d: {cohens_d:.3f}  (large effect > 0.8)")


def build_municipality_matrix(sim_matrix, munis, doc_ids):
    """Compute municipality-level mean similarity matrix, return dict."""
    unique_munis = sorted(set(munis))
    n = len(unique_munis)
    muni_to_idx = {m: i for i, m in enumerate(unique_munis)}

    sums = np.zeros((n, n))
    counts = np.zeros((n, n))
    for i, j in combinations(range(len(doc_ids)), 2):
        mi = muni_to_idx[munis[i]]
        mj = muni_to_idx[munis[j]]
        sums[mi, mj] += sim_matrix[i, j]
        sums[mj, mi] += sim_matrix[i, j]
        counts[mi, mj] += 1
        counts[mj, mi] += 1
    for i, m in enumerate(unique_munis):
        idxs = [k for k, mu in enumerate(munis) if mu == m]
        pairs = list(combinations(idxs, 2))
        if pairs:
            sums[i, i] = sum(sim_matrix[a, b] for a, b in pairs)
            counts[i, i] = len(pairs)

    mean_matrix = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return unique_munis, mean_matrix


def print_municipality_matrix(unique_munis, mean_matrix, method_name):
    n = len(unique_munis)
    print(f"\n{'='*65}")
    print(f"  {method_name} Mean Cosine Similarity Matrix (municipality level)")
    print(f"{'='*65}")
    header = "        " + "  ".join(f"{m:>6}" for m in unique_munis)
    print(header)
    for i, m in enumerate(unique_munis):
        row = f"  {m:>4}  " + "  ".join(
            f"{'*' if i==j else ''}{mean_matrix[i,j]:.3f}{'*' if i==j else ' '}"
            for j in range(n)
        )
        print(row)
    print("  (* = intra-municipality diagonal)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", action="store_true",
                        help="Also compute multilingual embedding-based similarity")
    parser.add_argument("--output", default="results/municipality_similarity.json",
                        help="Path to save results JSON (default: results/municipality_similarity.json)")
    args = parser.parse_args()

    print("Loading documents...")
    documents, doc_muni = load_documents()

    muni_counts = defaultdict(int)
    for m in doc_muni.values():
        muni_counts[m] += 1
    print(f"  {len(documents)} documents across {len(muni_counts)} municipalities")
    for m, c in sorted(muni_counts.items()):
        print(f"    {m}: {c} documents")

    output = {
        "dataset": {
            "num_documents": len(documents),
            "num_municipalities": len(muni_counts),
            "documents_per_municipality": {m: c for m, c in sorted(muni_counts.items())},
        }
    }

    # --- TF-IDF ---
    print("\nComputing TF-IDF representations...")
    intra_tfidf, inter_tfidf, sim_matrix, munis, doc_ids = compute_tfidf_similarity(documents, doc_muni)
    report_stats(intra_tfidf, inter_tfidf, "TF-IDF")
    unique_munis, mean_matrix = build_municipality_matrix(sim_matrix, munis, doc_ids)
    print_municipality_matrix(unique_munis, mean_matrix, "TF-IDF")

    output["tfidf"] = {
        "stats": compute_stats(intra_tfidf, inter_tfidf),
        "municipality_mean_similarity_matrix": {
            "municipalities": unique_munis,
            "matrix": mean_matrix.tolist(),
        },
    }

    # --- Embeddings (optional) ---
    if args.embeddings:
        print("\nComputing multilingual embedding representations...")
        intra_emb, inter_emb = compute_embedding_similarity(documents, doc_muni)
        if intra_emb is not None:
            report_stats(intra_emb, inter_emb, "Multilingual Embedding")
            output["embeddings"] = {"stats": compute_stats(intra_emb, inter_emb)}

    # --- Save ---
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print("\nDone.")


if __name__ == "__main__":
    main()
