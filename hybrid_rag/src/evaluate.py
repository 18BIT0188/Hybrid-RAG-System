# src/evaluate.py
import argparse
import csv
import json
import time
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from src.retrieve import retrieve as retrieve_hybrid
from src.generate import answer as generate_answer


# ---------------------------
# Text normalization for F1
# ---------------------------
_ARTICLES = {"a", "an", "the"}


def _normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks).strip()


def _f1_score(pred: str, gold: str) -> float:
    p = _normalize_answer(pred).split()
    g = _normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    common = defaultdict(int)
    for t in g:
        common[t] += 1

    num_same = 0
    for t in p:
        if common[t] > 0:
            num_same += 1
            common[t] -= 1

    if num_same == 0:
        return 0.0

    precision = num_same / len(p)
    recall = num_same / len(g)
    return (2 * precision * recall) / (precision + recall)


# ---------------------------
# URL-level ranking helpers
# ---------------------------
def _rank_urls_from_chunk_ranking(
    chunks: List[Dict], ranked_chunk_ids: List[int], top_k_urls: int
) -> List[str]:
    """
    Convert ranked chunk indices -> ranked unique URLs
    Rule: URL appears at the position of its first (earliest) chunk.
    """
    out = []
    seen = set()

    for idx in ranked_chunk_ids:
        if idx < 0 or idx >= len(chunks):
            continue
        url = chunks[idx].get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= top_k_urls:
            break

    return out


def _mrr_url(gt_urls: List[str], ranked_urls: List[str]) -> Tuple[float, int]:
    """
    URL-level MRR for a single query:
    if first correct URL at rank r => 1/r else 0
    Returns (mrr_contrib, rank_found_or_0)
    """
    gt = set(gt_urls or [])
    for i, url in enumerate(ranked_urls, start=1):
        if url in gt:
            return 1.0 / i, i
    return 0.0, 0


def _recall_at_k_url(gt_urls: List[str], ranked_urls: List[str], k: int) -> float:
    gt = set(gt_urls or [])
    if not gt:
        return 0.0
    top = set((ranked_urls or [])[:k])
    return len(gt.intersection(top)) / len(gt)


# ---------------------------
# Retrieval modes for ablation
# ---------------------------
def _get_ranked_chunk_ids(ret: Dict, mode: str, top_k_chunks: int) -> List[int]:
    """
    mode: 'dense' | 'sparse' | 'rrf'
    ret['dense'] = [(chunk_idx, score), ...]
    ret['sparse'] = [(chunk_idx, score), ...]
    ret['rrf']   = [(chunk_idx, rrf_score), ...]
    """
    if mode == "dense":
        return [i for i, _ in ret.get("dense", [])[:top_k_chunks]]
    if mode == "sparse":
        return [i for i, _ in ret.get("sparse", [])[:top_k_chunks]]
    return [i for i, _ in ret.get("rrf", [])[:top_k_chunks]]


def evaluate_questions(
    questions_path: str = "eval/questions_100.json",
    out_csv: str = "eval/results.csv",
    out_metrics_json: str = "eval/metrics.json",
    top_k_retriever: int = 10,
    top_k_urls_eval: int = 10,
    top_n_context: int = 5,
):
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    rows = []
    summary = {}

    for mode in ["dense", "sparse", "rrf"]:
        mrrs = []
        recalls = []
        f1s = []
        latencies = []
        failures = defaultdict(int)

        for q in questions:
            qid = q.get("id", "")
            query = q.get("question", "")
            gt_urls = q.get("source_urls", []) or []
            gt_ans = q.get("ground_truth", "") or ""
            cat = q.get("category", "unknown")

            # defaults so variables are always defined
            pred_ans = "I don't know"
            ranked_urls = []
            rank_found = 0
            mrr_c = 0.0
            recall_k = 0.0
            f1 = 0.0
            fail = "unknown"

            t0 = time.perf_counter()

            # Retrieval
            try:
                ret = retrieve_hybrid(query, top_k=top_k_retriever)
                chunks = ret.get("chunks", [])

                ranked_chunk_ids = _get_ranked_chunk_ids(
                    ret, mode=mode, top_k_chunks=top_k_retriever
                )
                ranked_urls = _rank_urls_from_chunk_ranking(
                    chunks, ranked_chunk_ids, top_k_urls=top_k_urls_eval
                )

                # Build mode-specific ordering for generator
                fake_rrf = [(idx, 0.0) for idx in ranked_chunk_ids[:top_n_context]]
                ret_for_gen = dict(ret)
                ret_for_gen["rrf"] = fake_rrf

                # Generation
                try:
                    # Expected signature from your compatibility wrapper:
                    # answer(query, top_k_dense=..., top_k_sparse=..., top_n=...)
                    # If your wrapper supports ret_for_gen, adapt there.
                    pred_ans, _top_used = generate_answer(query, ret_for_gen, top_n=top_n_context)
                except TypeError:
                    # Fallback for wrapper signature without ret object
                    try:
                        pred_ans, _top_used = generate_answer(query, top_n=top_n_context)
                    except Exception:
                        pred_ans = "I don't know"
                        fail = "generation_exception"
                except Exception:
                    pred_ans = "I don't know"
                    fail = "generation_exception"

            except Exception:
                # retrieval failure; keep defaults
                fail = "retrieval_exception"

            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            # Metrics
            mrr_c, rank_found = _mrr_url(gt_urls, ranked_urls)
            recall_k = _recall_at_k_url(gt_urls, ranked_urls, k=top_k_urls_eval)

            if gt_ans.strip().lower() == "i don't know":
                f1 = 1.0 if pred_ans.strip().lower() == "i don't know" else 0.0
            else:
                f1 = _f1_score(pred_ans, gt_ans)

            # Failure labeling (if no explicit exception label)
            if fail not in ("retrieval_exception", "generation_exception"):
                if rank_found == 0:
                    fail = "retrieval_fail"
                elif gt_ans.strip().lower() != "i don't know" and f1 < 0.2:
                    fail = "generation_or_grounding_fail"
                else:
                    fail = "ok"

            failures[fail] += 1
            mrrs.append(mrr_c)
            recalls.append(recall_k)
            f1s.append(f1)
            latencies.append(latency_ms)

            rows.append({
                "mode": mode,
                "id": qid,
                "category": cat,
                "question": query,
                "gt_urls": "|".join(gt_urls),
                "ranked_urls_topk": "|".join(ranked_urls),
                "rank_first_correct_url": rank_found,
                "mrr_url": mrr_c,
                "recall_url_at_k": recall_k,
                "ground_truth": gt_ans,
                "pred_answer": pred_ans,
                "answer_f1": f1,
                "latency_ms": latency_ms,
                "failure_label": fail,
            })

        # safe aggregates
        n = max(1, len(mrrs))
        summary[mode] = {
            "MRR_URL": sum(mrrs) / n,
            f"Recall_URL@{top_k_urls_eval}": sum(recalls) / n,
            "Answer_F1": sum(f1s) / n,
            "Latency_ms_avg": sum(latencies) / n,
            "Failure_counts": dict(failures),
            "Num_questions": len(mrrs),
        }

    # ---------------------------
    # SAFE CSV WRITING
    # ---------------------------
    fieldnames = [
        "mode", "id", "category", "question", "gt_urls",
        "ranked_urls_topk", "rank_first_correct_url",
        "mrr_url", "recall_url_at_k", "ground_truth",
        "pred_answer", "answer_f1", "latency_ms", "failure_label"
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)
        else:
            print("WARNING: No evaluation rows generated; wrote header only.")

    # ---------------------------
    # METRICS JSON
    # ---------------------------
    with open(out_metrics_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved results: {out_csv}")
    print(f"Saved metrics: {out_metrics_json}")
    print(f"Total evaluation rows: {len(rows)}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions_100.json")
    parser.add_argument("--out-csv", default="eval/results.csv")
    parser.add_argument("--out-metrics", default="eval/metrics.json")
    parser.add_argument("--top-k", type=int, default=10, help="top-k chunks for retrieval")
    parser.add_argument("--top-k-urls", type=int, default=10, help="top-k URLs for URL-level metrics")
    parser.add_argument("--top-n", type=int, default=5, help="top-n chunks for generation context")
    args = parser.parse_args()

    evaluate_questions(
        questions_path=args.questions,
        out_csv=args.out_csv,
        out_metrics_json=args.out_metrics,
        top_k_retriever=args.top_k,
        top_k_urls_eval=args.top_k_urls,
        top_n_context=args.top_n,
    )


if __name__ == "__main__":
    main()
