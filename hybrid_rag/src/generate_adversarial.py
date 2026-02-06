# src/generate_adversarial.py
# Generates adversarial questions:
# - paraphrased
# - negated
# - multi-hop
# - unanswerable

import argparse
import json
import os
import random

def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def paraphrase(q):
    swaps = {
        "What is": "Can you explain",
        "How does": "In what way does",
        "Why is": "What makes",
        "Explain": "Describe",
    }
    for k, v in swaps.items():
        if q.startswith(k):
            return q.replace(k, v, 1)
    return "Explain in different words: " + q

def negate(q):
    return q.replace(" is ", " is not ", 1)

def unanswerable(q):
    return "What is the phone number related to " + q.rstrip("?") + "?"

def multihop(q1, q2):
    return f"What connection exists between {q1.rstrip('?')} and {q2.rstrip('?')}?"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions_100.json")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--out", default="eval/questions_adversarial.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    qs = load_questions(args.questions)
    random.shuffle(qs)

    out = []
    qid = 1

    while len(out) < args.count:
        base = random.choice(qs)

        # paraphrase
        out.append({
            "id": f"adv{qid:03d}",
            "question": paraphrase(base["question"]),
            "ground_truth": base["ground_truth"],
            "source_urls": base["source_urls"],
            "category": "paraphrase"
        })
        qid += 1

        if len(out) >= args.count:
            break

        # negation
        out.append({
            "id": f"adv{qid:03d}",
            "question": negate(base["question"]),
            "ground_truth": base["ground_truth"],
            "source_urls": base["source_urls"],
            "category": "negation"
        })
        qid += 1

        if len(out) >= args.count:
            break

        # unanswerable
        out.append({
            "id": f"adv{qid:03d}",
            "question": unanswerable(base["question"]),
            "ground_truth": "",
            "source_urls": [],
            "category": "unanswerable"
        })
        qid += 1

        if len(out) >= args.count:
            break

        # multi-hop
        other = random.choice(qs)
        out.append({
            "id": f"adv{qid:03d}",
            "question": multihop(base["question"], other["question"]),
            "ground_truth": base["ground_truth"],
            "source_urls": base["source_urls"],
            "category": "multi-hop"
        })
        qid += 1

    out = out[:args.count]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(out)} adversarial questions -> {args.out}")

if __name__ == "__main__":
    main()
