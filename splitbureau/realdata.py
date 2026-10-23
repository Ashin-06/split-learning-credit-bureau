"""Split learning on real credit data: UCI "Default of Credit Card Clients".

30,000 credit-card holders at a Taiwanese bank, April to September 2005 (Yeh & Lien 2009,
UCI Machine Learning Repository, CC BY 4.0), 22.1% of whom defaulted the next month. All the
columns come from one bank, so the three "parties" are made by splitting them:
- bank (label holder): credit limit, education, marital status, age;
- bureau: repayment status for each of the six months (PAY_0, PAY_2 ... PAY_6);
- payments: the six monthly bill amounts and six payment amounts.
Sex is left out of every model. Amounts are log-scaled (sign * log1p). The parties are
already aligned row by row, so PSI is skipped.

Rows are split 75/25 into train and test (seed 42, as in the synthetic benchmark), features
are standardised on the training rows, and every model uses the synthetic benchmark's
settings unchanged (160 epochs, batch 64, learning rate 0.025, the same defence). Nothing
was tuned on this data.

    python -m splitbureau realdata --data "default of credit card clients.xls"
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from .inversion import evaluate_cut_layer_leakage
from .splitnn import CutLayerDefenseConfig, SplitCreditNet, train_bank_only_baseline, train_centralized_oracle

AMOUNTS = ["LIMIT_BAL"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
PARTIES = {
    "bank": ["LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE"],
    "bureau": ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
    "payments": AMOUNTS[1:],
}
NO_DEFENCE = CutLayerDefenseConfig(enabled=False, clip_norm=100.0, decorr_lambda=0.0, noise_sigma=0.0)
FULL_DEFENCE = CutLayerDefenseConfig(enabled=True, clip_norm=0.95, decorr_lambda=0.55, noise_sigma=0.45, bottleneck_rank=1)


def load(path: str | Path) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    import pandas as pd  # only needed for the real data (.xls needs xlrd)

    df = pd.read_excel(path, header=1) if str(path).endswith(".xls") else pd.read_csv(path)
    col = lambda c: np.sign(df[c].to_numpy(float)) * np.log1p(np.abs(df[c].to_numpy(float))) if c in AMOUNTS else df[c].to_numpy(float)
    return ({p: np.column_stack([col(c) for c in cols]) for p, cols in PARTIES.items()},
            df["default payment next month"].to_numpy(dtype=np.float64))


def split(parts: Dict[str, np.ndarray], y: np.ndarray, test_fraction: float = 0.25, seed: int = 42):
    perm = np.random.default_rng(seed).permutation(len(y))
    n_test = int(round(len(y) * test_fraction))
    te, tr = perm[:n_test], perm[n_test:]
    std = {}
    for p, x in parts.items():
        mu, sd = x[tr].mean(axis=0), x[tr].std(axis=0) + 1e-8
        std[p] = ((x[tr] - mu) / sd, (x[te] - mu) / sd)
    return std, y[tr], y[te]


def experiments(path: str | Path, epochs: int = 160, seed: int = 42, log=print) -> dict:
    parts, y = load(path)
    std, y_tr, y_te = split(parts, y, seed=seed)
    (b_tr, b_te), (u_tr, u_te), (t_tr, t_te) = std["bank"], std["bureau"], std["payments"]
    dims = tuple(len(c) for c in PARTIES.values())
    keep = lambda m: {"roc_auc": m.roc_auc, "pr_auc": m.pr_auc, "brier": m.brier_score}

    res = {"dataset": "UCI Default of Credit Card Clients (30,000 rows, CC BY 4.0)",
           "parties": PARTIES, "train_rows": len(y_tr), "test_rows": len(y_te),
           "test_default_rate": round(float(y_te.mean()), 4), "epochs": epochs, "models": {}}

    bank, _ = train_bank_only_baseline(b_tr, y_tr, b_te, y_te, epochs=epochs, seed=seed)
    res["models"]["bank_alone"] = keep(bank)
    log(f"bank alone      ROC-AUC {bank.roc_auc}")
    for name, cfg in (("split_no_defence", NO_DEFENCE), ("split_full_defence", FULL_DEFENCE)):
        net = SplitCreditNet(embed_dim=8, hidden_dim=16, top_hidden_dim=16, defense_config=cfg, seed=seed, input_dims=dims)
        net.fit(b_tr, u_tr, t_tr, y_tr, epochs=epochs, batch_size=64, lr=0.025)
        m = net.evaluate(b_te, u_te, t_te, y_te, model_name=name)
        inv = evaluate_cut_layer_leakage(net, b_tr, u_tr, t_tr, b_te, u_te, t_te, model_name=name,
                                         feature_names=(PARTIES["bureau"], PARTIES["payments"]))
        res["models"][name] = {**keep(m), "attacker_r2": inv.overall_mean_r2, "attacker_r2_bureau": inv.party_upi_r2,
                               "attacker_r2_payments": inv.party_telecom_r2, "attacker_r2_per_feature": {k.replace("upi.", "bureau.").replace("telecom.", "payments."): v
                                                           for k, v in inv.per_feature_r2.items()}}
        log(f"{name:15s} ROC-AUC {m.roc_auc}  attacker R2 {inv.overall_mean_r2}")
    pooled, _ = train_centralized_oracle(b_tr, u_tr, t_tr, y_tr, b_te, u_te, t_te, y_te, epochs=epochs, seed=seed)
    res["models"]["pooled"] = keep(pooled)
    log(f"pooled          ROC-AUC {pooled.roc_auc}")
    return res


def main(data: str, out: str = "results/real_data.json", epochs: int = 160) -> dict:
    res = experiments(data, epochs)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    return res
