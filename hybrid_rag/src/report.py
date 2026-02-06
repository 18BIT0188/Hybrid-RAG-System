# src/report.py
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def _save_bar_chart(metrics_json_path: str, out_png: str):
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    modes = list(m.keys())
    mrr = [m[x]["MRR_URL"] for x in modes]
    rec = [m[x][next(k for k in m[x].keys() if k.startswith("Recall_URL@"))] for x in modes]
    f1  = [m[x]["Answer_F1"] for x in modes]

    fig = plt.figure()
    x = range(len(modes))
    plt.bar([i - 0.25 for i in x], mrr, width=0.25, label="MRR_URL")
    plt.bar([i for i in x], rec, width=0.25, label="Recall_URL@K")
    plt.bar([i + 0.25 for i in x], f1, width=0.25, label="Answer_F1")
    plt.xticks(list(x), modes)
    plt.ylabel("Score")
    plt.title("Ablation: Dense vs Sparse vs Hybrid (RRF)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def _save_latency_hist(df: pd.DataFrame, out_png: str):
    fig = plt.figure()
    for mode in df["mode"].unique():
        subset = df[df["mode"] == mode]["latency_ms"]
        plt.hist(subset, bins=20, alpha=0.5, label=mode)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title("Latency distribution by mode")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def build_pdf_report(
    results_csv="eval/results.csv",
    metrics_json="eval/metrics.json",
    out_pdf="eval/report.pdf",
):
    Path("eval").mkdir(exist_ok=True)
    df = pd.read_csv(results_csv)
    chart1 = "eval/ablation_scores.png"
    chart2 = "eval/latency_hist.png"

    _save_bar_chart(metrics_json, chart1)
    _save_latency_hist(df, chart2)

    with open(metrics_json, "r", encoding="utf-8") as f:
        m = json.load(f)

    c = canvas.Canvas(out_pdf, pagesize=letter)
    width, height = letter

    # Title page
    y = height - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Hybrid RAG Evaluation Report")
    y -= 25
    c.setFont("Helvetica", 11)
    c.drawString(50, y, "Metrics: URL-level MRR + URL Recall@K + Answer F1 + Ablation + Error analysis")
    y -= 40

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Overall Summary (by retrieval mode)")
    y -= 20
    c.setFont("Helvetica", 10)

    for mode, stats in m.items():
        c.drawString(55, y, f"- {mode}: MRR_URL={stats['MRR_URL']:.4f}, "
                           f"Recall={list(stats.keys())[1]}={list(stats.values())[1]:.4f}, "
                           f"F1={stats['Answer_F1']:.4f}, Lat(ms)={stats['Latency_ms_avg']:.1f}")
        y -= 14

    y -= 10
    c.drawImage(chart1, 50, y - 220, width=520, height=220, preserveAspectRatio=True, mask='auto')
    y -= 250
    c.drawImage(chart2, 50, y - 220, width=520, height=220, preserveAspectRatio=True, mask='auto')
    c.showPage()

    # Error analysis page
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 60, "Error Analysis (Examples)")
    y = height - 90
    c.setFont("Helvetica", 10)

    # Show a few failures for hybrid
    fail_df = df[(df["mode"] == "rrf") & (df["failure_label"] != "ok")].head(6)
    if fail_df.empty:
        c.drawString(50, y, "No failures found in top sample. (Good sign!)")
    else:
        for _, row in fail_df.iterrows():
            text = (
                f"ID={row['id']} | Cat={row['category']} | Fail={row['failure_label']} | "
                f"Rank={row['rank_first_correct_url']} | F1={row['answer_f1']:.3f}\n"
                f"Q: {row['question']}\n"
                f"GT: {str(row['ground_truth'])[:180]}\n"
                f"Pred: {str(row['pred_answer'])[:180]}\n"
            )
            for line in text.split("\n"):
                c.drawString(50, y, line)
                y -= 12
            y -= 8
            if y < 80:
                c.showPage()
                y = height - 60
                c.setFont("Helvetica", 10)

    c.save()
    print(f"Saved PDF report: {out_pdf}")

if __name__ == "__main__":
    build_pdf_report()
