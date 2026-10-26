-- Trained models. config and metrics are JSON; weights = base64 of a NumPy .npz
-- (read without pickle). The newest version is the one served.
CREATE TABLE models (
    version BIGINT PRIMARY KEY,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    config TEXT NOT NULL,
    metrics TEXT NOT NULL,
    weights TEXT NOT NULL
);

-- One row per score served: who asked, about which customer reference, with which model
-- and what came back. Raw features are not stored.
CREATE TABLE score_log (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    user_name TEXT NOT NULL,
    customer_ref TEXT NOT NULL,
    model_version BIGINT NOT NULL,
    probability TEXT NOT NULL,
    decision TEXT NOT NULL
);
CREATE INDEX score_log_created ON score_log (created_at);
