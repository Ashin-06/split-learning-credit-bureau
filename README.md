# Split-Learning Credit Bureau (`splitbureau`)

Three institutions hold different facts about the same customers: a bank (balances, salary regularity, bounces, utilisation, plus the default label), a UPI payments app (spend diversity, late-night betting share, P2P velocity, BNPL rollovers), and a telecom/utility provider (SIM tenure, recharge regularity, on-time electricity payments, handset tier). This project shows how they could train one default model without pooling raw data:

1. **Private set intersection** (`psi.py`) finds the customers all three share without exchanging the ID lists.
2. **Split learning** (`splitnn.py`): each party runs a small bottom network on its own features and sends only the resulting embedding. The bank's top network combines the embeddings and sends gradients back.
3. **A cut-layer defence**: a low-rank bottleneck, norm clipping, a cross-covariance penalty and Gaussian noise. It limits how much a curious bank can reconstruct of the other parties' features from their embeddings. `inversion.py` measures this by attacking it.

Everything runs in one process on **synthetic data** (`data.py`), with all three parties simulated. Defaults are a deterministic function of the generated features plus a little noise, so the absolute AUCs are much higher than any real bureau would see. See "Real data" for the same models on real card-holder data.

> **The PSI is a reference implementation.** It uses libsodium's Ed25519 group arithmetic, but the protocol code is hand-written and unreviewed, and it is secure only against parties who follow it (semi-honest).

## How it works

**PSI.**
- Each ID is hashed onto Ed25519: a SHA-512 try-and-increment candidate, multiplied by the cofactor 8 so it lies in the prime-order subgroup.
- Each party blinds its points with a fresh random scalar, and the bank compares the double-blinded tokens.
- A token match means an ID match. The tests check that hashed points really are in the prime-order subgroup.

What it reveals: both sides learn the intersection size, and the bank learns which of its rows are in it. In the three-party flow the bank runs two pairwise PSIs, so it learns its overlap with each party separately, not just the three-way overlap.

**Split network.**
- Bottom networks: 4 features → 16 hidden units (tanh) → 8-dimensional embedding. The top network is one hidden layer plus a linear skip.
- Training is plain mini-batch SGD written in NumPy.
- The defence for the UPI and telecom parties has four parts:
  - a rank-1 bottleneck, so each party effectively sends one number spread over 8 dimensions;
  - L2 clipping to norm 0.95;
  - a cross-covariance penalty between features and embedding;
  - Gaussian noise with σ = 0.45.
- The noise is a fixed setting. It is **not calibrated to a differential-privacy budget**, and no ε is claimed.

**Attack.** The bank is assumed to know the raw features of the training customers (a generous assumption) and fits two decoders from embedding to features: ridge regression on a quadratic basis, and a 64-64 MLP. Leakage is the better decoder's held-out R², averaged over the UPI and telecom features.

## Results

`python -m splitbureau bench --ablation` (900 shared customers, 160 epochs, seed 42):

| Model | ROC-AUC | PR-AUC | Attacker R² |
|---|---|---|---|
| Bank's own 4 features | 0.703 | 0.617 | – |
| Split network, no defence | 0.976 | 0.969 | 0.954 |
| Split network, full defence | 0.867 | 0.841 | 0.134 |
| All 12 features pooled in one place | 0.986 | 0.981 | – |

PSI found exactly the 900 shared customers among 1,080 / 1,140 / 1,110 records, in about 5 seconds.

| Defence ablation | ROC-AUC | Attacker R² |
|---|---|---|
| none | 0.976 | 0.954 |
| rank-1 bottleneck only | 0.932 | 0.233 |
| bottleneck + clipping + penalty | 0.933 | 0.229 |
| bottleneck + noise | 0.865 | 0.137 |
| full defence | 0.867 | 0.134 |

What this shows:
- An undefended 8-dimensional embedding of 4 features is nearly invertible (R² 0.95).
- Most of the protection comes from sending a single number per party instead of a rich embedding. Clipping and the penalty add nothing measurable here, and the noise buys a further drop in R² at a real cost in accuracy.
- Low R² means these two decoders fail. It is not a proof that nothing leaks; a stronger attacker, or one that also uses the gradients, could recover more.

## Real data

`python -m splitbureau realdata --data "default of credit card clients.xls"` trains the same models on UCI's Default of Credit Card Clients data: 30,000 card holders at a Taiwanese bank in 2005, 22.1% of whom defaulted the next month (CC BY 4.0). Results are in `results/real_data.json`.
- **How the parties are formed.** All the columns come from one bank, so they are split three ways. The bank keeps the credit limit, education, marital status and age, plus the label. A "bureau" holds six months of repayment status. A "payments" party holds six monthly bill amounts and six payment amounts. Sex is left out. The rows are already aligned, so PSI is skipped.
- **How it is tested.** A 75/25 train/test split (7,500 test rows), features standardised on the training rows, amounts log-scaled. Every model setting is the synthetic benchmark's, unchanged; nothing was tuned on this data.

| Model | ROC-AUC | PR-AUC | Attacker R² |
|---|---|---|---|
| Bank's own 4 features | 0.623 | 0.300 | – |
| Split network, no defence | 0.769 | 0.543 | 0.947 |
| Split network, full defence | 0.735 | 0.491 | 0.111 |
| All 22 features pooled in one place | 0.766 | 0.540 | – |

What the real data shows:
- **The other parties' data is what matters.** On its own, the bank's limit and demographics give 0.62. Adding repayment history and amounts through the split network gives 0.77, the same as pooling everything; the 0.003 difference is under one standard error (about 0.007).
- **The defence costs less here than on synthetic data**: 0.035 ROC-AUC against 0.11. It keeps three quarters of what the other parties add.
- **What leaks is what predicts.** Without the defence the attacker rebuilds every feature (R² 0.84–0.99). With it, R² falls to 0.11 overall, but the latest repayment status (PAY_0) is still recovered at R² 0.41. The one number each party sends is mostly a risk score, and PAY_0 is the strongest risk signal. Bill amounts, which hardly predict default, leak almost nothing (R² ≤ 0.01).

## Run it

```bash
pip install -r requirements-dev.lock && pip install -e .
python -m pytest -q                            # 34 tests
python -m splitbureau bench --ablation         # about 40 s
python -m splitbureau realdata --data FILE     # UCI credit default data, about 2.5 minutes (pip install -e ".[realdata]")
python -m splitbureau serve                    # API + console at http://127.0.0.1:8017
```

Endpoints: `POST /psi/align`, `POST /train`, `POST /score`, `GET /bench`, plus `GET /healthz`, `GET /readyz` and `GET /metrics`. The server plays all three parties, so `/score` and `/train` do receive every party's raw features. The score, tier and decision it returns are illustrative only.

**Settings and access.** The service reads `SPLITBUREAU_*` environment variables (see `docs/runbook.md`). Callers send an API key in `X-API-Key` (the console has a field for it) or `Authorization: Bearer`. Each key has a role: `lender` scores, `data_scientist` also trains and runs PSI, `monitor` reads metrics, `admin` can do everything including the benchmark. `python -m splitbureau.ops keygen` makes a key and the hash to configure. Without keys the service runs open in development and refuses to listen on anything but loopback; `SPLITBUREAU_ENV=production` refuses to start without hashed keys and a database.

**State.** Trained models and a log of every score (user, customer reference, model version, probability, decision; no features) are kept in SQLite or Postgres (`SPLITBUREAU_DATABASE_URL`). An empty database gets the default model, trained on synthetic data, on first use. `/train` stores a new version and `/score` always uses the newest, so a restart serves the same model and replicas share it.

**Docker.**

```bash
docker build -t split-learning-credit-bureau:1.0.0 .
docker run -d -p 8017:8017 -v splitbureau-data:/data -e SPLITBUREAU_API_KEYS='{"sha256:...": {"user": "ops", "role": "admin"}}' split-learning-credit-bureau:1.0.0
```

The image runs in production mode as a non-root user. Run without keys it exits with a configuration error.

## Limits

- **Simulated parties.** There is no network transport and no separate deployments. The real-data test splits one bank's columns three ways; real bureau or payments data would be noisier and only partly overlapping.
- **No formal privacy guarantee.** There's no differential-privacy accounting and only two attack decoders. Label leakage through the gradients the bank sends back is not addressed.
- **Semi-honest PSI only.** A malicious party could send malformed tokens or lie about its set, and hashing is not constant-time.
- **Features have fairness and legal problems.** Late-night betting share, handset price tier and similar alternative data raise discrimination and consent issues that a real lender would need to clear with regulators and fair-lending review before use.
- **One service plays every party.** There is no transport between parties and no per-party authentication; TLS is left to the ingress.

## Production readiness

In place (1.0.0):
- Settings checked at startup; production refuses to start without hashed API keys and a database, and never listens publicly without keys.
- API keys with roles, per-client rate limit, request body cap, security headers, CORS only for listed origins, validation errors that don't echo input.
- Models versioned in a database (no pickle) and a score log recording user, model version and decision; SQLite or Postgres, with migrations.
- JSON logs with request IDs, `/healthz`, `/readyz`, Prometheus `/metrics`.
- Pinned dependency locks, a non-root container image with a health check, CI running ruff, bandit, pip-audit and the tests.
- `SECURITY.md`, `docs/threat-model.md`, `docs/runbook.md`, `docs/model-card.md`.

Still needed before real use:
- A cryptographer's review of the PSI protocol (production mode logs a warning until then), and a malicious-secure protocol if parties can't be trusted to follow it.
- A formal privacy analysis of the cut-layer defence (differential-privacy accounting, stronger attacks, gradient label leakage).
- Separate deployments per party with authenticated transport, instead of one service holding everyone's data.
- An independent security review and penetration test.
- Validation on the bank's own data, plus fairness and lawful-use review of every feature.
- RBI and DPDP Act 2023 sign-off (consent for sharing data between the bank, payments app and telecom, purpose limitation, credit-information rules).
- Real integrations: the parties' data pipelines and the lender's decision system.
- HSM/KMS-backed key storage if signing or encryption keys are added, and managed secrets for the API key file.
- An on-call team, SLOs and alerting wired to the metrics above.
