import numpy as np
import pandas as pd

from splitbureau.realdata import PARTIES, experiments


def test_uci_columns_split_into_three_parties(tmp_path) -> None:
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({c: rng.integers(-2, 9, n) for cols in PARTIES.values() for c in cols})
    df["LIMIT_BAL"] = rng.integers(10_000, 500_000, n)
    df["AGE"] = rng.integers(21, 70, n)
    df["SEX"] = rng.integers(1, 3, n)
    df["default payment next month"] = (df["PAY_0"] + rng.normal(0, 2, n) > 2).astype(int)
    f = tmp_path / "uci.csv"
    df.to_csv(f, index=False)

    res = experiments(f, epochs=3, log=lambda *_: None)
    assert (res["train_rows"], res["test_rows"]) == (300, 100)
    assert set(res["models"]) == {"bank_alone", "split_no_defence", "split_full_defence", "pooled"}
    leak = res["models"]["split_no_defence"]["attacker_r2_per_feature"]
    assert len(leak) == len(PARTIES["bureau"]) + len(PARTIES["payments"]) and "bureau.PAY_0" in leak
    assert all(0.0 <= m["roc_auc"] <= 1.0 for m in res["models"].values())
