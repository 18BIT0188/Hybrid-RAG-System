# src/pipeline_eval.py
import argparse
from pathlib import Path

from src.question_gen import generate_questions
from src.evaluate import evaluate_questions
from src.report import build_pdf_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-questions", action="store_true", help="Generate eval/questions_100.json")
    parser.add_argument("--eval", action="store_true", help="Run evaluation and write CSV/JSON")
    parser.add_argument("--report", action="store_true", help="Generate PDF report")
    parser.add_argument("--k", type=int, default=10, help="Top-K per retriever")
    parser.add_argument("--urls_k", type=int, default=10, help="Top-K URLs for MRR/Recall")
    parser.add_argument("--n", type=int, default=5, help="Top-N chunks to feed generator")
    args = parser.parse_args()

    Path("eval").mkdir(exist_ok=True)

    if args.gen_questions or not Path("eval/questions_100.json").exists():
        generate_questions(n_total=100, include_unanswerable=10)

    if args.eval or not Path("eval/results.csv").exists():
        evaluate_questions(
            questions_path="eval/questions_100.json",
            out_csv="eval/results.csv",
            out_metrics_json="eval/metrics.json",
            top_k_retriever=args.k,
            top_k_urls_eval=args.urls_k,
            top_n_context=args.n,
        )

    if args.report or not Path("eval/report.pdf").exists():
        build_pdf_report(
            results_csv="eval/results.csv",
            metrics_json="eval/metrics.json",
            out_pdf="eval/report.pdf",
        )

if __name__ == "__main__":
    main()
