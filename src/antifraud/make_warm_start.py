import argparse
import logging

import pandas as pd

from antifraud import db
from antifraud.service.preprocessing import FeatureBuilder, warm_start

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("warm_start")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", required=True,
        help="CSV с step, customer, age, gender, merchant, category, amount"
    )
    parser.add_argument("--chunk", type=int, default=20_000)
    args = parser.parse_args()

    db.init()

    history = pd.read_csv(args.source)
    required = ["step", "customer", "merchant", "age", "gender", "category", "amount"]
    missing = required - set(history.columns)
    
    if missing:
        raise ValueError(f"в источнике нет колонок: {missing}")

    logger.info("Загружено %d строк истории", len(history))

    builder = FeatureBuilder()
    warm_start(builder, history, chunk=args.chunk)

    logger.info("Прогрев завершён")


if __name__ == "__main__":
    main()