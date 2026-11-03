# Split-Learning Credit Bureau

Three organisations hold different facts about the same customers: a bank (balances, salary regularity, bounced payments, credit use and whether the customer defaulted), a UPI payments app (spending patterns, peer-to-peer transfers, buy-now-pay-later rollovers) and a telecom provider (SIM age, recharge regularity, on-time bill payments). This project shows how they can train one default-prediction model without pooling their raw data.

1. Private set intersection finds the customers they have in common without exchanging customer lists.
2. Split learning: each party runs a small network on its own data and sends only the output. The bank's network combines these outputs and sends gradients back.
3. A defence at the point where data is shared limits how much the bank can reconstruct of the other parties' data, and an attack measures how well it works.

All three parties run in one process. The main benchmark uses synthetic data; the same models were also tested on real card-holder data.

## Quick start

```bash
python -m pip install -r requirements-dev.lock && python -m pip install -e .
python -m pytest -q                            # 35 tests
python -m splitbureau bench --ablation         # about 40 s
python -m splitbureau realdata --data FILE     # UCI credit default data, about 2.5 minutes
python -m splitbureau serve                    # API and console on http://127.0.0.1:8017
```

To run it in a container:

```bash
docker build -t split-learning-credit-bureau .
docker run -d -p 8017:8017 -v splitbureau-data:/data \
  -e SPLITBUREAU_API_KEYS='{"sha256:...": {"user": "ops", "role": "admin"}}' \
  -e SPLITBUREAU_ACCEPT_UNREVIEWED_CRYPTO=yes split-learning-credit-bureau
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /psi/align` | find shared customers |
| `POST /train` | train and store a new model version |
| `POST /score` | score a customer with the newest model |
| `GET /bench` | the benchmark |
| `GET /healthz`, `/readyz`, `/metrics` | health, readiness and Prometheus metrics |

The server plays all three parties, so `/score` and `/train` receive every party's raw data, and the scores it returns are illustrative. API keys carry a role: `lender` scores, `data_scientist` also trains and runs the intersection, `monitor` reads metrics, `admin` does everything. Create one with `python -m splitbureau.ops keygen`. Models and a log of every score (no features) are kept in SQLite or Postgres.

## Results

| Data | Bank alone | Split model, no defence | Split model, with defence | All data pooled |
|---|---|---|---|---|
| Synthetic, ROC-AUC | 0.703 | 0.976 | 0.867 | 0.986 |
| Real card holders (UCI), ROC-AUC | 0.623 | 0.769 | 0.735 | 0.766 |

Without the defence, an attacker can rebuild the other parties' data almost exactly (R² 0.95). With it, R² falls to 0.11–0.13, although on real data the most predictive feature, the latest repayment status, is still partly recovered (R² 0.41). How the protocol, network and attack work, with the full tables, is in [docs/design.md](docs/design.md).

## Configuration

| Variable | Purpose |
|---|---|
| `SPLITBUREAU_ENV=production` | refuse to start without hashed API keys and a database |
| `SPLITBUREAU_ACCEPT_UNREVIEWED_CRYPTO=yes` | required in production: acknowledges that the PSI protocol is hand-written and unreviewed |
| `SPLITBUREAU_API_KEYS` / `SPLITBUREAU_KEYS_FILE` | hashed API keys with roles |
| `SPLITBUREAU_DATABASE_URL` | `sqlite:///...` or `postgresql://...` |

## Limitations

- All parties run in one service with no network transport between them; a real setup needs a separate deployment per party with authenticated connections.
- The private set intersection is written in this project and has not been reviewed by a cryptographer, and it only protects against parties that follow the protocol.
- The defence has no formal privacy guarantee, only two attacks were tried, and leakage through gradients is not covered.
- The real-data test splits one bank's columns three ways; real bureau and payments data would be noisier and only partly overlap.
- Some of the alternative-data features raise fairness, consent and legal questions that a lender would need to clear before use.
- The code has not had an independent security review.

See [SECURITY.md](SECURITY.md), [docs/threat-model.md](docs/threat-model.md) and [docs/model-card.md](docs/model-card.md).

## License

MIT. See [LICENSE](LICENSE).
