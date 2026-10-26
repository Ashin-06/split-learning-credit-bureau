"""Settings, auth, limits, headers, metrics, health, persistence and migrations."""
import hashlib
import importlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from splitbureau import db, ops
from splitbureau.data import generate_vertical_credit_dataset
from splitbureau.registry import PARTS, STATS, Registry
from splitbureau.splitnn import CutLayerDefenseConfig, SplitCreditNet

KEY = "splitbureau_" + "x" * 40
HASHED = {f"sha256:{hashlib.sha256(KEY.encode()).hexdigest()}": {"user": "meera", "role": "lender"}}
KEYS = {"k-lend": {"user": "meera", "role": "lender"}, "k-ds": {"user": "asha", "role": "data_scientist"},
        "k-mon": {"user": "prom", "role": "monitor"}, "k-admin": {"user": "root", "role": "admin"}}


def prod_env(**over):
    env = {"SPLITBUREAU_ENV": "production", "SPLITBUREAU_API_KEYS": json.dumps(HASHED), "SPLITBUREAU_DATABASE_URL": "sqlite://"}
    env.update(over)
    return {k: v for k, v in env.items() if v is not None}


def boot(monkeypatch, **env):
    for k in [k for k in os.environ if k.startswith("SPLITBUREAU_")]:
        monkeypatch.delenv(k)
    for k, v in env.items():
        monkeypatch.setenv(f"SPLITBUREAU_{k}", v)
    import splitbureau.api as api

    return importlib.reload(api)


def test_production_settings_accepted():
    s = ops.load_settings(prod_env())
    assert s.production and not s.demo and not s.ephemeral


@pytest.mark.parametrize("change, message", [
    ({"SPLITBUREAU_API_KEYS": None}, "API keys"),
    ({"SPLITBUREAU_DATABASE_URL": None}, "DATABASE_URL"),
    ({"SPLITBUREAU_API_KEYS": json.dumps({KEY: {"user": "meera", "role": "lender"}})}, "sha256"),
    ({"SPLITBUREAU_API_KEYS": json.dumps({"sha256:" + "0" * 64: {"user": "x", "role": "root"}})}, "roles"),
    ({"SPLITBUREAU_API_KEYS": "not json"}, "JSON"),
    ({"SPLITBUREAU_CORS_ORIGINS": "*"}, "CORS"),
    ({"SPLITBUREAU_ENV": "prod"}, "development or production"),
    ({"SPLITBUREAU_DATABASE_URL": "mysql://x"}, "postgresql"),
    ({"SPLITBUREAU_PORT": "http"}, "integers"),
])
def test_production_refuses_to_start(change, message):
    with pytest.raises(ops.ConfigError, match=message):
        ops.load_settings(prod_env(**change))


def test_no_public_bind_without_keys():
    with pytest.raises(ops.ConfigError, match="without API keys"):
        ops.load_settings({"SPLITBUREAU_HOST": "0.0.0.0"})
    assert ops.load_settings({"SPLITBUREAU_HOST": "0.0.0.0", "SPLITBUREAU_API_KEYS": json.dumps(HASHED)}).host == "0.0.0.0"
    dev = ops.load_settings({})
    assert dev.demo and dev.database_url == "sqlite://" and dev.port == 8017


def test_serve_exits_with_a_readable_error():
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPLITBUREAU_")}
    r = subprocess.run([sys.executable, "-m", "splitbureau", "serve", "--host", "0.0.0.0"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "configuration error" in r.stderr and "without API keys" in r.stderr


def test_roles_decide_what_a_key_may_do(monkeypatch):
    api = boot(monkeypatch, API_KEYS=json.dumps(KEYS))
    c = TestClient(api.app)
    assert c.post("/score", json={}).status_code == 401
    assert c.post("/score", json={}, headers={"x-api-key": "wrong"}).status_code == 401
    assert c.post("/score", json={}, headers={"x-api-key": "k-lend"}).status_code == 200
    assert c.post("/psi/align", json={}, headers={"x-api-key": "k-lend"}).status_code == 403
    assert c.post("/train", json={}, headers={"authorization": "Bearer k-lend"}).status_code == 403
    assert c.post("/psi/align", json={}, headers={"x-api-key": "k-ds"}).status_code == 200
    assert c.get("/bench", headers={"x-api-key": "k-ds"}).status_code == 403  # minutes of CPU: admin only
    assert c.post("/score", json={}, headers={"x-api-key": "k-mon"}).status_code == 403
    assert c.get("/metrics", headers={"x-api-key": "k-lend"}).status_code == 403
    m = c.get("/metrics", headers={"x-api-key": "k-mon"})
    assert m.status_code == 200 and 'http_requests_total{method="POST",route="/score",status="200"}' in m.text
    assert "splitbureau_decisions_total" in m.text and "errors_total" in m.text
    assert "http_request_duration_seconds_bucket" in m.text


def test_input_limits(monkeypatch):
    api = boot(monkeypatch, API_KEYS=json.dumps(KEYS))
    c, hdr = TestClient(api.app), {"x-api-key": "k-admin"}
    assert c.post("/score", json={"savings_balance": -1}, headers=hdr).status_code == 422
    assert c.post("/score", json={"savings_balance": "NaN"}, headers=hdr).status_code == 422
    big = {"bank_ids": [f"ID{i}" for i in range(api.MAX_PSI_IDS + 1)], "upi_ids": ["a"]}
    assert c.post("/psi/align", json=big, headers=hdr).status_code == 422
    assert c.post("/psi/align", json={"bank_ids": ["x" * 65], "upi_ids": ["a"]}, headers=hdr).status_code == 422
    assert c.post("/train", json={"epochs": 1000}, headers=hdr).status_code == 422
    assert c.post("/train", json={"seed": -1}, headers=hdr).status_code == 422


def test_rate_limit(monkeypatch):
    api = boot(monkeypatch, RATE_LIMIT="3")
    c = TestClient(api.app)
    codes = [c.post("/psi/align", json={}).status_code for _ in range(5)]
    assert codes[:3] == [200] * 3 and codes[3:] == [429, 429]
    r = c.post("/psi/align", json={})
    assert int(r.headers["retry-after"]) >= 1
    assert c.get("/healthz").status_code == 200  # probes are never limited


def test_body_size_cap(monkeypatch):
    api = boot(monkeypatch, MAX_BODY_BYTES="1000")
    c = TestClient(api.app)
    assert c.post("/psi/align", json={"bank_ids": ["a" * 60] * 20, "upi_ids": ["a"]}).status_code == 413

    def chunks():  # no content-length: the cap still holds
        for _ in range(20):
            yield b"a" * 100

    assert c.post("/psi/align", content=chunks(), headers={"content-type": "application/json"}).status_code == 413
    assert c.post("/psi/align", json={"bank_ids": ["a"], "upi_ids": ["a"]}).status_code == 200


def test_headers_and_request_ids(monkeypatch):
    c = TestClient(boot(monkeypatch).app)
    r = c.post("/psi/align", json={}, headers={"x-request-id": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    assert r.headers["content-security-policy"] == ops.API_CSP and r.headers["cache-control"] == "no-store"
    assert "unsafe-inline" in c.get("/").headers["content-security-policy"]
    assert len(c.get("/healthz", headers={"x-request-id": "bad id\n"}).headers["x-request-id"]) == 32
    assert "access-control-allow-origin" not in c.get("/healthz", headers={"origin": "https://evil.example"}).headers


def test_cors_only_for_listed_origins(monkeypatch):
    c = TestClient(boot(monkeypatch, CORS_ORIGINS="https://console.bank.example").app)
    ok = c.get("/healthz", headers={"origin": "https://console.bank.example"})
    assert ok.headers["access-control-allow-origin"] == "https://console.bank.example"
    assert "access-control-allow-origin" not in c.get("/healthz", headers={"origin": "https://evil.example"}).headers


def test_errors_do_not_leak(monkeypatch):
    api = boot(monkeypatch)

    @api.app.get("/boom")
    def boom():
        raise RuntimeError("secret detail: customer PAN_COMMON_00042")

    c = TestClient(api.app)
    r = c.get("/boom")
    assert r.status_code == 500 and "PAN_COMMON" not in r.text and r.json()["request_id"] == r.headers["x-request-id"]
    bad = c.post("/score", json={"savings_balance": "4111111111111111"})
    assert bad.status_code == 422 and "4111" not in bad.text
    assert c.post("/score", content=b"{not json", headers={"content-type": "application/json"}).status_code == 422


def test_health_and_readiness(monkeypatch):
    api = boot(monkeypatch)
    c = TestClient(api.app)
    assert c.get("/healthz").json() == {"status": "ok"}
    r = c.get("/readyz")
    assert r.status_code == 200 and r.json()["checks"] == {"database": True}
    api.store.dispose()
    monkeypatch.setattr(api.store, "connect", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    r = c.get("/readyz")
    assert r.status_code == 503 and r.json()["checks"]["database"] is False and "db down" not in r.text


def test_json_logs_carry_request_ids(capsys):
    ops.setup_logging("INFO")
    token = ops.request_id.set("rid-1")
    ops.log.info("hello", extra={"fields": {"n": 1}})
    ops.request_id.reset(token)
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["msg"] == "hello" and line["request_id"] == "rid-1" and line["n"] == 1 and line["level"] == "INFO"


def test_production_warns_that_the_psi_is_unreviewed(monkeypatch, tmp_path, capsys):
    env = {k.removeprefix("SPLITBUREAU_"): v for k, v in prod_env(SPLITBUREAU_DATABASE_URL=f"sqlite:///{tmp_path / 'p.db'}").items()}
    api = boot(monkeypatch, **env)
    warnings = [json.loads(line) for line in capsys.readouterr().out.splitlines() if "UNREVIEWED" in line]
    assert warnings and warnings[0]["level"] == "WARNING" and "cryptographer" in warnings[0]["msg"]
    assert api.registry.newest_version() == 1  # the model is trained before the first request
    r = TestClient(api.app).post("/psi/align", json={}, headers={"x-api-key": KEY})
    assert r.status_code == 403  # lender
    assert "reviewed" in api.PSI_WARNING


def _url(tmp_path):
    return os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path / 'state.db'}"


def test_model_round_trips_through_the_database(tmp_path):
    ds = generate_vertical_credit_dataset(n_samples=200, seed=1)
    m = SplitCreditNet(defense_config=CutLayerDefenseConfig(enabled=False), seed=1)
    m.fit(ds.x_bank_train, ds.x_upi_train, ds.x_telecom_train, ds.y_train, epochs=3)
    reg = Registry(db.connect(_url(tmp_path)))
    v = reg.save(m, {k: getattr(ds, k) for k in STATS}, {"roc_auc": 0.5}, "test")
    assert reg.newest_version() >= v
    back = Registry(db.connect(_url(tmp_path))).load(v)  # a new process, same database
    assert back.version == v
    np.testing.assert_array_equal(back.model.predict_proba(ds.x_bank_test, ds.x_upi_test, ds.x_telecom_test),
                                  m.predict_proba(ds.x_bank_test, ds.x_upi_test, ds.x_telecom_test))
    np.testing.assert_array_equal(back.stats["upi_std"], ds.upi_std)
    defended = SplitCreditNet(defense_config=CutLayerDefenseConfig(enabled=True), seed=1)
    loaded = reg.load(reg.save(defended, {k: getattr(ds, k) for k in STATS}, {}, "test")).model
    for p in PARTS:
        for k, v in vars(getattr(defended, p)).items():
            if isinstance(v, np.ndarray):
                np.testing.assert_array_equal(getattr(getattr(loaded, p), k), v)
    assert loaded.party_upi.W2_a is not None


def test_served_model_and_score_log_survive_a_restart(monkeypatch, tmp_path):
    url = _url(tmp_path)
    keys = json.dumps(KEYS)
    api = boot(monkeypatch, DATABASE_URL=url, API_KEYS=keys)
    c = TestClient(api.app)
    s1 = c.post("/score", json={"customer_id": "CUST-r1"}, headers={"x-api-key": "k-lend"}).json()
    version = s1["model_version"]
    weights = api.current_model().model.top_net.W_top1.copy()

    api = boot(monkeypatch, DATABASE_URL=url, API_KEYS=keys)  # restart: same database
    assert api.registry.newest_version() == version  # no retraining
    np.testing.assert_array_equal(api.current_model().model.top_net.W_top1, weights)
    c = TestClient(api.app)
    t = c.post("/train", json={"n_samples": 200, "epochs": 20}, headers={"x-api-key": "k-ds"}).json()
    assert t["model_version"] == version + 1
    assert c.post("/score", json={"customer_id": "CUST-r1"}, headers={"x-api-key": "k-lend"}).json()["model_version"] == version + 1
    with api.store.connect() as conn:
        rows = conn.execute(text("SELECT user_name, model_version, decision FROM score_log WHERE customer_ref = 'CUST-r1' "
                                 "ORDER BY model_version")).all()
        trained_by = conn.execute(text("SELECT created_by FROM models WHERE version = :v"), {"v": version + 1}).scalar()
    assert [tuple(r[:2]) for r in rows] == [("meera", version), ("meera", version + 1)]
    assert rows[0][2] == s1["recommended_decision"] and trained_by == "asha"
    with api.store.begin() as conn:
        conn.execute(text("DELETE FROM score_log WHERE customer_ref = 'CUST-r1'"))


def test_migrations_apply_once(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    eng = db.engine(url)
    assert db.migrate(eng) == ["001_init.sql"]
    assert db.migrate(eng) == []
    with eng.connect() as c:
        assert [r[0] for r in c.execute(text("SELECT version FROM schema_migrations"))] == [1]
        tables = {r[0] for r in c.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))}
    assert {"models", "score_log"} <= tables
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("SPLITBUREAU_")}, "SPLITBUREAU_DATABASE_URL": url}
    out = subprocess.run([sys.executable, "-m", "splitbureau", "migrate"], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "up to date" in out.stdout
