"""Trained models and the score log, kept in the database (see db.py).

A model is stored as its JSON config plus its arrays in a NumPy .npz, base64 in a TEXT
column. The .npz is read with allow_pickle=False, so a row cannot carry code. Every
/train adds a version; /score uses the newest one, so replicas sharing the database
serve the same model.
"""
from __future__ import annotations

import base64
import dataclasses
import io
import json
import threading
import uuid
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from .splitnn import CutLayerDefenseConfig, SplitCreditNet

PARTS = ("party_bank", "party_upi", "party_telecom", "top_net")
STATS = ("bank_mean", "bank_std", "upi_mean", "upi_std", "telecom_mean", "telecom_std")


@dataclasses.dataclass
class Served:
    version: int
    model: SplitCreditNet
    stats: dict[str, np.ndarray]  # standardisation of raw features, from the training rows


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Registry:
    def __init__(self, engine: Engine):
        self.db = engine
        self._lock = threading.Lock()  # ponytail: per process; other replicas are caught by the primary key

    def save(self, model: SplitCreditNet, stats: dict, metrics: dict, user: str) -> int:
        """Store a trained model as the next version and return that version."""
        arrays = {f"{p}.{k}": v for p in PARTS for k, v in vars(getattr(model, p)).items() if isinstance(v, np.ndarray)}
        arrays.update({f"stats.{k}": np.asarray(stats[k]) for k in STATS})
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        config = {"embed_dim": model.embed_dim, "hidden_dim": model.hidden_dim, "top_hidden_dim": model.top_hidden_dim,
                  "seed": model.seed, "defense": dataclasses.asdict(model.defense_config),
                  "input_dims": [getattr(model, p).input_dim for p in PARTS[:3]]}
        row = {"t": _now(), "u": user, "c": json.dumps(config), "m": json.dumps(metrics),
               "w": base64.b64encode(buf.getvalue()).decode()}
        for attempt in range(3):
            try:
                with self._lock, self.db.begin() as c:
                    row["v"] = (c.execute(text("SELECT MAX(version) FROM models")).scalar() or 0) + 1
                    c.execute(text("INSERT INTO models (version, created_at, created_by, config, metrics, weights) "
                                   "VALUES (:v, :t, :u, :c, :m, :w)"), row)
                    return row["v"]
            except IntegrityError:  # another replica took the same version; number again
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    def newest_version(self) -> int:
        with self.db.connect() as c:
            return c.execute(text("SELECT MAX(version) FROM models")).scalar() or 0

    def load(self, version: int) -> Served:
        with self.db.connect() as c:
            row = c.execute(text("SELECT config, weights FROM models WHERE version = :v"), {"v": version}).one()
        cfg = json.loads(row[0])
        model = SplitCreditNet(embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"], top_hidden_dim=cfg["top_hidden_dim"],
                               defense_config=CutLayerDefenseConfig(**cfg["defense"]), seed=cfg["seed"],
                               input_dims=tuple(cfg["input_dims"]))
        stats = {}
        with np.load(io.BytesIO(base64.b64decode(row[1])), allow_pickle=False) as z:
            for name in z.files:
                part, attr = name.split(".", 1)
                if part == "stats":
                    stats[attr] = z[name]
                else:
                    setattr(getattr(model, part), attr, z[name])
        for p in PARTS[:3]:
            # Cut-layer noise while serving comes from OS entropy, not the training seed, which
            # is in the config above and would let anyone replay the noise.
            getattr(model, p).rng = np.random.default_rng()
        return Served(version, model, stats)

    def log_score(self, user: str, customer_ref: str, version: int, probability: float, decision: str) -> None:
        with self.db.begin() as c:
            c.execute(text("INSERT INTO score_log (id, created_at, user_name, customer_ref, model_version, probability, decision) "
                           "VALUES (:i, :t, :u, :r, :v, :p, :d)"),
                      {"i": uuid.uuid4().hex, "t": _now(), "u": user, "r": customer_ref, "v": version,
                       "p": f"{probability:.4f}", "d": decision})
