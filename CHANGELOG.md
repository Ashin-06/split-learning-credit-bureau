# Changelog

All notable changes to this project. Versions follow [semantic versioning](https://semver.org/).

## 1.0.0 — 2026-10-26

### Added
- `pyproject.toml` with a `splitbureau` console command, and pinned `requirements.lock` /
  `requirements-dev.lock` (generated with `uv pip compile --universal`).
- Container image: slim Python 3.12 base, non-root user, health check on `/healthz`.
- CI on every push: ruff, bandit, pip-audit, pytest with a coverage report.
- Settings from `SPLITBUREAU_*` environment variables, checked at startup.
  `SPLITBUREAU_ENV=production` refuses to start without hashed API keys and
  `SPLITBUREAU_DATABASE_URL`.
- Roles on API keys: `lender` scores, `data_scientist` also trains and runs PSI, `monitor` scrapes
  metrics, `admin` does all of that plus the benchmark. Keys can be stored as `sha256:<hex>`;
  `python -m splitbureau.ops keygen` makes new ones.
- Per-client rate limit, request body cap (chunked bodies included), security headers, CORS only
  for listed origins.
- JSON logs with request IDs; `/healthz`, `/readyz` and Prometheus `/metrics`
  (`splitbureau_decisions_total` counts score decisions).
- Model registry and score log in a database: SQLite by default, Postgres through
  `SPLITBUREAU_DATABASE_URL`, numbered SQL migrations (`python -m splitbureau migrate`). Models are
  stored as NumPy `.npz` read without pickle; each score records the model version and the user.
- A warning, logged at startup in production and returned by `/psi/align`, that the PSI is
  unreviewed.
- `SECURITY.md`, `docs/threat-model.md`, `docs/runbook.md`, `docs/model-card.md`.

### Changed
- `/score` and `/train` no longer keep the model in process memory. A restart serves the same model,
  `/train` adds a version, and replicas sharing the database pick it up. Responses carry
  `model_version`.
- The served model draws its cut-layer noise from OS entropy instead of the training seed, which is
  stored with the model and would let anyone replay the noise.
- `SPLITBUREAU_API_KEY` (one shared key) is replaced by `SPLITBUREAU_API_KEYS` with roles. Without
  keys the service refuses to listen on anything but loopback.
- `/health` is replaced by `/healthz` (liveness) and `/readyz` (database reachable).
- `/bench` needs the admin role. Validation errors no longer echo the submitted input. `/train`
  rejects negative seeds.

### Unchanged
- PSI, split network, defence, attack and benchmark code, and every benchmark and real-data number
  in the README. `results/real_data.json` was not re-run: the only change in `realdata.py` turns two
  lambdas into functions for the linter, which does not change what it computes.
