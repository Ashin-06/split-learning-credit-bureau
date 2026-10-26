# Threat model

What the bureau service protects, from whom, and what is still open. Written for the 1.0.0 code;
it has not been reviewed by anyone independent.

## The deployment this describes

One service plays all three institutions (bank, payments app, telecom). That is the simulation
the code implements: `/score` and `/train` receive every party's raw features, and `/psi/align`
receives every party's identifier list. In a real deployment each party would run its own bottom
network and PSI client and send only blinded tokens and cut-layer embeddings; nothing here does
that yet. Read every "the bank learns" row below as "the operator of this service already knows".

## What it protects

| Asset | Where it lives |
|---|---|
| Customer identifiers sent to `/psi/align` | request memory only; not stored or logged |
| Raw features sent to `/score` and `/train` | request memory only; not stored or logged |
| Trained models (weights, standardisation statistics, config, metrics) | `models` table, NumPy `.npz` in base64 |
| Score log (who scored which customer reference, with which model version, what came back) | `score_log` table |
| API keys | callers hold the key; the service holds only SHA-256 hashes |

## Trust boundaries

```
lender / data scientist ──(TLS at your ingress)──► bureau service ──► database (models, score log)
                                                     (plays bank, payments app and telecom)
```

1. **Caller → service.** Callers are authenticated by API key; the key, not any header, decides
   user and role.
2. **Service → database.** The database holds model weights and the score log, not raw features
   or identifiers.
3. **Between parties** (bank ↔ payments app ↔ telecom). Simulated inside one process. The PSI and
   cut-layer rows below describe what would cross this line in a real deployment.

## Threats and what stops them

| Threat | Mitigation | Residual risk |
|---|---|---|
| A party learns identifiers outside the intersection | Ed25519 DH-PSI with fresh random scalars per run; points hashed into the prime-order subgroup | **Unreviewed cryptography.** Hand-written protocol code, semi-honest only, non-constant-time hashing. Both sides learn the intersection size; the bank learns its overlap with each party separately. A malicious party can lie about its set or send malformed tokens. Production logs a warning; do not use it on real identifiers before a cryptographer's review. |
| The bank reconstructs another party's features from its embeddings | Rank-1 bottleneck, L2 clipping, cross-covariance penalty, Gaussian noise (σ 0.45) | **No formal guarantee** (no DP accounting). Two decoders tested: R² 0.95 → 0.13 on synthetic data, 0.95 → 0.11 on the UCI data, but the latest repayment status (PAY_0) is still recovered at R² 0.41. A stronger attacker, or one using gradients, may recover more. |
| Labels leak to the other parties through the gradients the bank sends back | — | Not addressed. |
| Someone replays the cut-layer noise | The served model draws noise from OS entropy | Training and the benchmarks use a seeded generator (for reproducible numbers), and the seed is stored with the model. In a real deployment each party must use its own unseeded generator. |
| The service operator sees all raw features | — | Inherent in the single-process simulation. Treat the service as holding all three parties' data. |
| Stolen API key | Keys are per client and revocable; rate limit per key; the score log records the user | A stolen key works until removed from the key list and the service restarts |
| Brute-forcing keys | 256-bit random keys; rate limit per address for unknown keys | Rate limit is per process |
| A model in the database is swapped or edited | Weights load with `allow_pickle=False` (a row cannot run code); every version records who trained it | Anyone who can write the `models` table can change the scores the service gives. Restrict write access to the service's own database user. |
| Score log exposes customers | Only the caller's customer reference, model version, probability and decision are stored; no features | Use internal references, not PANs or phone numbers, as `customer_id` |
| CPU exhaustion through training or the benchmark | `/train` bounded (≤ 2,000 rows, ≤ 200 epochs) and one run at a time per process (429 otherwise); `/bench` needs the admin role; body cap; PSI lists ≤ 2,000 IDs of ≤ 64 characters | Per process; several replicas multiply it |
| Error messages leak data | Validation errors carry no input; unhandled errors return only a request ID | — |
| Browser attacks on the console | Security headers, CSP, `frame-ancestors 'none'`, no CORS unless origins are listed | The console's CSP allows its own inline script; don't serve it on a shared origin |
| Misconfigured production start | `SPLITBUREAU_ENV=production` refuses to start without hashed keys and a database; no public bind without keys | Production mode can't check that keys are *good*, only that they exist and have the right shape |

## Not covered

- TLS. Terminate it at your ingress or service mesh; the service speaks plain HTTP.
- Separate deployments per party, transport between them, and per-party authentication.
- Fairness and lawful use of the features (late-night betting share, handset tier and similar
  alternative data). See `docs/model-card.md`.
- An insider with database write access.
