# Model card: split-learning default model

## What it is

A three-party vertical split neural network that predicts whether a borrower defaults, trained
without pooling raw features:

- each party (bank, payments app, telecom; or bank, "bureau", "payments" in the real-data test)
  runs a bottom network: its features → 16 hidden units (tanh) → an 8-dimensional embedding;
- the bank's top network combines the three embeddings: one hidden layer plus a linear skip;
- training is plain mini-batch SGD written in NumPy.

The **defended** variant protects the two non-bank parties' embeddings with a rank-1 bottleneck
(each party effectively sends one number spread over 8 dimensions), L2 clipping to norm 0.95, a
cross-covariance penalty (λ 0.55) and Gaussian noise (σ 0.45). The noise is a fixed setting. It is
**not calibrated to a differential-privacy budget**, and no ε is claimed.

## Intended use

Research and demonstration: how much a vertical split model gains over the bank's own data, and how
much the cut layer leaks. It is **not** a credit-decision model. The model the API serves is trained
on synthetic data (720 generated customers, seed 42, 140 epochs, or whatever a data scientist trains
through `/train`), and its score, tier and decision are illustrative.

## Data

**Synthetic** (`data.py`). Bank (balances, salary regularity, bounces, utilisation, plus the label),
UPI app (spend diversity, late-night betting share, P2P velocity, BNPL rollovers) and telecom (SIM
tenure, recharge regularity, on-time electricity payments, handset tier). Defaults are a
deterministic function of the generated features plus a little noise, so absolute AUCs are much
higher than any real bureau would see.

**Real-data check** (`python -m splitbureau realdata`, results in `results/real_data.json`). UCI's
Default of Credit Card Clients: 30,000 card holders at a Taiwanese bank in 2005, 22.1% of whom
defaulted the next month (CC BY 4.0).
- All columns come from one bank, so they are split three ways: the bank keeps `LIMIT_BAL`,
  `EDUCATION`, `MARRIAGE`, `AGE` and the label; a "bureau" holds six months of repayment status
  (`PAY_0`, `PAY_2`…`PAY_6`); a "payments" party holds six bill amounts and six payment amounts.
  Sex is left out. The rows are already aligned, so PSI is skipped.
- 75/25 train/test split: 22,500 / 7,500 rows, test default rate 21.81%. Features standardised on the
  training rows, amounts log-scaled. Every model setting is the synthetic benchmark's, unchanged;
  nothing was tuned on this data.

## Metrics

Attacker R² is the better of two decoders (ridge regression on a quadratic basis, and a 64-64 MLP)
fitted from embedding to features, held-out, averaged over the non-bank parties' features. The
attacker is assumed to know the raw features of the training customers (a generous assumption).

**UCI credit default data** (7,500 test rows, 160 epochs):

| Model | ROC-AUC | PR-AUC | Brier | Attacker R² |
|---|---|---|---|---|
| Bank's own 4 features | 0.6234 | 0.2995 | 0.1654 | – |
| Split network, no defence | 0.7692 | 0.5427 | 0.1352 | 0.9466 (bureau 0.987, payments 0.9063) |
| Split network, full defence | 0.7347 | 0.4912 | 0.1421 | 0.1111 (bureau 0.1654, payments 0.0568) |
| All 22 features pooled in one place | 0.7659 | 0.5397 | 0.1361 | – |

**Synthetic benchmark** (`python -m splitbureau bench --ablation`: 900 shared customers, 160 epochs,
seed 42):

| Model | ROC-AUC | PR-AUC | Attacker R² |
|---|---|---|---|
| Bank's own 4 features | 0.703 | 0.617 | – |
| Split network, no defence | 0.976 | 0.969 | 0.954 |
| Split network, full defence | 0.867 | 0.841 | 0.134 |
| All 12 features pooled in one place | 0.986 | 0.981 | – |

What the numbers say:
- On the real data the other parties' features are what matter: 0.62 for the bank alone, 0.77 for the
  split network, the same as pooling everything. The 0.003 difference is under one standard error
  (about 0.007).
- The defence costs 0.035 ROC-AUC on the real data (0.11 on synthetic) and keeps three quarters of
  what the other parties add.
- **What leaks is what predicts.** With the defence, the latest repayment status is still recovered
  at R² 0.41 (`PAY_0` 0.4147). Bill amounts, which hardly predict default, leak almost nothing
  (R² ≤ 0.01). The one number each party sends is mostly a risk score.
- Most of the protection comes from the rank-1 bottleneck. On synthetic data clipping and the
  penalty add nothing measurable; the noise buys a further drop in R² at a real cost in accuracy.

## Known limitations and failure modes

- **Simulated parties.** The real-data test splits one bank's columns three ways; real bureau or
  payments data would be noisier and only partly overlapping.
- **Low R² means these two decoders fail.** It is not a proof that nothing leaks; a stronger
  attacker, or one that also uses the gradients, could recover more. Label leakage through the
  gradients the bank sends back is not addressed.
- **Not validated on any lender's own population**, and not monitored for drift.
- **Fairness has not been evaluated.** Sex is excluded from the real-data run, but age and marital
  status are features. The synthetic features include late-night betting share and handset price
  tier, alternative data that raises discrimination and consent issues a lender would need to clear
  with regulators and fair-lending review before use.
- The score bands (credit score 900 − 600 × p, tiers at p = 0.20 / 0.45 / 0.70) are illustrative
  cut-offs, not calibrated to any loss target.
