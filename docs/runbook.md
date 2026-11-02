# Runbook

How to deploy, configure and operate the bureau service. Commands assume the repository root.

## Deploy

```bash
docker build -t split-learning-credit-bureau:1.0.0 .
python -m splitbureau.ops keygen          # one API key per client; keep only the hash
docker run -d --name splitbureau -p 8017:8017 -v splitbureau-data:/data \
  -e SPLITBUREAU_ACCEPT_UNREVIEWED_CRYPTO=yes \
  -e SPLITBUREAU_KEYS_FILE=/run/secrets/splitbureau-keys.json \
  split-learning-credit-bureau:1.0.0
curl -fsS http://127.0.0.1:8017/readyz
```

The image runs in production mode as a non-root user and keeps its SQLite database in the `/data`
volume. On an empty database it trains the default model (synthetic data, a few seconds) before it
takes traffic. It refuses to start, and logs why, when anything required is missing. In production
it also logs a warning that the PSI is unreviewed; that is expected. Terminate TLS in front of it.

For more than one replica, use Postgres (`SPLITBUREAU_DATABASE_URL=postgresql://user:pass@host/db`)
and put a shared rate limiter at the ingress. Replicas pick up a newly trained model on their next
score.

Without Docker: `pip install -r requirements.lock && pip install -e .`, then `splitbureau serve`.

## Configure

| Variable | Required in production | Meaning |
|---|---|---|
| `SPLITBUREAU_ENV` | — | `development` (default) or `production` |
| `SPLITBUREAU_ACCEPT_UNREVIEWED_CRYPTO` | `yes` | acknowledges that the PSI protocol has not been reviewed by a cryptographer; production refuses to start without it |
| `SPLITBUREAU_API_KEYS` / `SPLITBUREAU_KEYS_FILE` | yes | JSON `{"sha256:<hex>": {"user": "...", "role": "..."}}`. Roles: `lender` (`/score`), `data_scientist` (`/score`, `/train`, `/psi/align`), `monitor` (`/metrics`), `admin` (all, plus `/bench`) |
| `SPLITBUREAU_DATABASE_URL` | yes | `sqlite:////data/splitbureau.db` or `postgresql://...` (default in development: in memory) |
| `SPLITBUREAU_HOST`, `SPLITBUREAU_PORT` | — | default `127.0.0.1:8017`; any non-loopback address needs API keys |
| `SPLITBUREAU_RATE_LIMIT` | — | requests per minute per client (default 120) |
| `SPLITBUREAU_MAX_BODY_BYTES` | — | default 1,000,000 |
| `SPLITBUREAU_CORS_ORIGINS` | — | comma-separated origins allowed to call from a browser; none by default |
| `SPLITBUREAU_LOG_LEVEL` | — | default `INFO` |
| `FORWARDED_ALLOW_IPS` | — | addresses of proxies whose `X-Forwarded-For` is trusted (uvicorn) |

The service has no secrets of its own beyond the API keys.

## Watch

- `GET /healthz`: the process is up (used by the container health check).
- `GET /readyz`: database reachable; 503 otherwise.
- `GET /metrics` (role `monitor`): `http_requests_total`, `http_request_duration_seconds`,
  `errors_total`, `splitbureau_decisions_total{decision="APPROVE_STANDARD_APR|APPROVE_WITH_RISK_SPREAD|CAPPED_LIMIT_MANUAL_REVIEW|DECLINE_CREDIT_LINE"}`.
- Logs are one JSON object per line on stdout, each with a `request_id` that is also returned in the
  `X-Request-ID` response header. Training logs `model trained` with the version and user.

Suggested alerts: `/readyz` failing for 2 minutes; `errors_total{kind="unhandled"}` increasing; 5xx
share above 1% for 10 minutes; a sudden shift in the mix of `splitbureau_decisions_total` (a
different model, or different inputs); 429s on `/train` (someone retrying training in a loop).

## Models

- `POST /train` (role `data_scientist`) trains on synthetic data with the given settings and stores
  the defended model as the next version. The response has the test metrics and attack results.
- `/score` always uses the newest version; the response and the score log carry `model_version`.
- **Roll back**: there is no endpoint. Delete the newer rows from `models`
  (`DELETE FROM models WHERE version > N`) and the next score loads version N. The score log keeps
  the version each score was made with.

## Rotate keys

**API keys.** Add the new hash to the key file, restart, give the client the new key, then remove
the old hash and restart again.

## Back up and restore

SQLite (in the `/data` volume):

```bash
docker exec splitbureau python -c "import sqlite3; s=sqlite3.connect('/data/splitbureau.db'); d=sqlite3.connect('/data/backup.db'); s.backup(d)"
docker cp splitbureau:/data/backup.db ./splitbureau-$(date +%F).db
```

Restore: stop the container, copy the backup to `/data/splitbureau.db` in the volume, start it, check
`/readyz` and one `/score` (its `model_version` should be the newest version in the backup).

Postgres: `pg_dump` / `pg_restore` as usual. Migrations are applied automatically on start, or with
`python -m splitbureau migrate`.

The score log names customers by the caller's reference: store backups like the database itself.

## Incidents

1. **Leaked API key.** Remove its hash from the key file and restart. Look for that user in the
   logs and in `score_log.user_name` / `models.created_by`. Issue a new key.
2. **Unexpected model.** Check `models` for versions you didn't train (`created_by`), roll back as
   above, and find out who has write access to the database.
3. **PSI used on real identifiers.** Stop it: the protocol is unreviewed. Treat the identifier
   lists as disclosed to the service operator and follow the data-breach procedure (DPDP Act
   notification duties).
4. **Database unavailable.** `/readyz` returns 503 and scores fail with 500. Nothing is queued;
   callers retry.
