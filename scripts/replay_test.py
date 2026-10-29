"""
Прогон тестовой части через api.
Запуск (при открытом SSH-туннеле на 8000):
    uv run python scripts/replay_test.py --data data/transactions.csv --url http://localhost:8000
"""
import argparse
import json
import urllib.error
import urllib.request

import pandas as pd

COLS = ["step", "customer", "age", "gender", "merchant", "category", "amount"]


def make_payload(rows: list[dict]):
    return {"rows": rows}


def post(url: str, payload) -> object:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        print("HTTP status:", e.code)
        print("Response:", e.read().decode("utf-8", errors="replace"))
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/transactions.csv")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--train-size", type=float, default=0.8)
    ap.add_argument("--chunk", type=int, default=1000)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df.merchant = df.merchant.str.strip("'")
    df.customer = df.customer.str.strip("'")
    df.age = df.age.str.strip("'")
    df.gender = df.gender.str.strip("'")
    df.category = df.category.str.strip("'")
    test_start = int(df["step"].max() * args.train_size)
    test = df[df["step"] >= test_start].sort_values("step", kind="stable")
    print(f"test: {len(test):,} строк, step {test['step'].min()}-{test['step'].max()}")

    for i in range(0, len(test), args.chunk):
        rows = test.iloc[i:i + args.chunk][COLS].to_dict("records")
        post(f"{args.url}/v1/predict/batch", make_payload(rows))
        print(f"{min(i + args.chunk, len(test)):,}/{len(test):,}")


if __name__ == "__main__":
    main()