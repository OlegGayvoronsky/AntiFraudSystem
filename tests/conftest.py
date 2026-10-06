import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from antifraud.service.app import app

BATCH_PATH = Path(__file__).parent / "history.csv"


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def good_row():
    return {
      "step": 143,
      "customer": "C1216867117",
      "age": "4",
      "gender": "F",
      "merchant": "M1823072687",
      "category": "es_transportation",
      "amount": 49.77,
    }


def _load_raw_transactions(path: Path) -> pd.DataFrame:
    trans = pd.read_json(path)
 
    return trans
 
 
@pytest.fixture
def raw_transactions() -> pd.DataFrame:
    if not BATCH_PATH.exists():
        pytest.skip(f"нет тестовых данных: {BATCH_PATH}")
    
    return _load_raw_transactions(BATCH_PATH)
