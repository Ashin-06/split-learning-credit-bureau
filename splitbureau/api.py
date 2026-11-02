"""
FastAPI Service for the Vertically Partitioned DH-PSI & Split-Learning Credit Bureau.

Endpoints:
  - GET  /          : Interactive HTML Console (web/console.html)
  - POST /psi/align : Run 2-party or 3-party elliptic-curve DH-PSI alignment     (permission: align)
  - POST /train     : Train Vertical SplitNN with configurable Cut-Layer Privacy Defense  (train)
  - POST /score     : Score a borrower via 3-party Cut-Layer embeddings          (score)
  - GET  /bench     : Run & return full benchmark suite metrics                  (bench)
  - GET  /healthz, /readyz, /metrics (see ops.py)

This is a single-process demo: the server plays all three institutions, so /score and /train
receive every party's raw features. In a real deployment each party runs its own bottom network
and sends only its cut-layer embedding.

Settings come from SPLITBUREAU_* environment variables (ops.py). Trained models and a log of
every score are kept in the database (registry.py), so a restart serves the same model.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import db, ops
from .bench import BenchmarkSummary, run_full_benchmark
from .data import generate_vertical_credit_dataset
from .inversion import evaluate_cut_layer_leakage
from .psi import align_three_parties, align_two_parties
from .registry import STATS, Registry, Served
from .splitnn import (
    CutLayerDefenseConfig,
    SplitCreditNet,
    train_bank_only_baseline,
)

log = logging.getLogger("splitbureau.api")
PSI_WARNING = ("the PSI protocol is hand-written and has not been reviewed by a cryptographer, and is only "
               "secure against parties that follow it; do not use it to match real customer identifiers")

settings = ops.load_settings()
ops.setup_logging(settings.log_level)
# before touching the database, so a refused start leaves no trace
if settings.production and os.environ.get("SPLITBUREAU_ACCEPT_UNREVIEWED_CRYPTO", "").strip().lower() != "yes":
    raise ops.ConfigError(PSI_WARNING + "; set SPLITBUREAU_ACCEPT_UNREVIEWED_CRYPTO=yes to run anyway")
store = db.connect(settings.database_url)
registry = Registry(store)
if settings.production:
    log.warning("UNREVIEWED CRYPTOGRAPHY: " + PSI_WARNING)

app = FastAPI(
    title="SplitBureau: DH-PSI & Vertical Split-Learning Credit Bureau",
    version="1.0.0",
    description=(
        "Vertically Partitioned Elliptic-Curve Diffie-Hellman PSI & 3-Party Split Neural "
        "Network Credit Bureau with Cut-Layer Feature-Inversion Defense."
    ),
)
ops.install(app, settings, {"database": lambda: store.connect().close()})

_CONSOLE_HTML_PATH = Path(__file__).resolve().parent.parent / "web" / "console.html"
MAX_PSI_IDS = 2_000
_TRAIN_LOCK = threading.Lock()  # one training run at a time per process
_MODEL_LOCK = threading.Lock()
_served: Optional[Served] = None
_cached_bench: Optional[BenchmarkSummary] = None


def _bootstrap() -> None:
    """First start with an empty database: train the default defended model and store it."""
    ds = generate_vertical_credit_dataset(n_samples=720, seed=42)
    model = SplitCreditNet(embed_dim=8, hidden_dim=16, top_hidden_dim=16, seed=42, defense_config=CutLayerDefenseConfig(
        enabled=True, clip_norm=0.95, decorr_lambda=0.55, noise_sigma=0.45, bottleneck_rank=1))
    model.fit(ds.x_bank_train, ds.x_upi_train, ds.x_telecom_train, ds.y_train, epochs=140, batch_size=64, lr=0.025)
    metrics = model.evaluate(ds.x_bank_test, ds.x_upi_test, ds.x_telecom_test, ds.y_test, model_name="Defended_Vertical_SplitNN")
    registry.save(model, {k: getattr(ds, k) for k in STATS}, dataclasses.asdict(metrics), "bootstrap")


def current_model() -> Served:
    """The newest stored model; picks up versions trained by other replicas."""
    global _served
    with _MODEL_LOCK:
        newest = registry.newest_version()
        if newest == 0:
            _bootstrap()
            newest = registry.newest_version()
        if _served is None or _served.version != newest:
            _served = registry.load(newest)
        return _served


if settings.production:
    current_model()  # train or load before taking traffic


class PSIAlignRequest(BaseModel):
    bank_ids: List[str] = Field(
        max_length=MAX_PSI_IDS,
        default_factory=lambda: [
            "PAN_COMMON_00001",
            "PAN_COMMON_00002",
            "PAN_BANK_ONLY_991",
            "PAN_COMMON_00003",
            "PAN_BANK_ONLY_992",
        ]
    )
    upi_ids: List[str] = Field(
        max_length=MAX_PSI_IDS,
        default_factory=lambda: [
            "PAN_UPI_ONLY_501",
            "PAN_COMMON_00003",
            "PAN_COMMON_00001",
            "PAN_UPI_ONLY_502",
            "PAN_COMMON_00002",
        ]
    )
    telecom_ids: Optional[List[str]] = Field(
        max_length=MAX_PSI_IDS,
        default_factory=lambda: [
            "PAN_COMMON_00002",
            "PAN_TEL_ONLY_701",
            "PAN_COMMON_00001",
            "PAN_COMMON_00003",
        ]
    )


class TrainSplitNNRequest(BaseModel):
    n_samples: int = Field(default=800, ge=200, le=2000)
    epochs: int = Field(default=140, ge=20, le=200)
    clip_norm: float = Field(default=0.95, ge=0.2, le=20.0)
    decorr_lambda: float = Field(default=0.55, ge=0.0, le=5.0)
    noise_sigma: float = Field(default=0.45, ge=0.0, le=2.0)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class ScoreBorrowerRequest(BaseModel):
    customer_id: str = Field("PAN_COMMON_00042", max_length=64)
    # Party 1 (Bank) features
    savings_balance: float = Field(default=2.4, ge=0.0, le=1000.0, allow_inf_nan=False, description="Savings balance in INR Lakhs")
    salary_regularity: float = Field(default=0.72, ge=0.0, le=1.0)
    bounce_count: float = Field(default=1.0, ge=0.0, le=20.0)
    credit_utilization: float = Field(default=0.58, ge=0.0, le=1.0)
    # Party 2 (UPI Super-App) features
    merchant_spend_diversity: float = Field(default=0.45, ge=0.0, le=1.0)
    late_night_betting_share: float = Field(default=0.38, ge=0.0, le=1.0)
    p2p_velocity: float = Field(default=4.2, ge=0.0, le=20.0)
    bnpl_rollover_count: float = Field(default=3.0, ge=0.0, le=20.0)
    # Party 3 (Telecom & Utility) features
    sim_tenure_months: float = Field(default=18.0, ge=1.0, le=240.0)
    prepaid_recharge_consistency: float = Field(default=0.48, ge=0.0, le=1.0)
    electricity_ontime_ratio: float = Field(default=0.52, ge=0.0, le=1.0)
    device_price_tier: float = Field(default=2.5, ge=1.0, le=5.0)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if _CONSOLE_HTML_PATH.exists():
        return HTMLResponse(_CONSOLE_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>SplitBureau Credit Bureau API</h1><p>Use /docs or /bench.</p>"
    )


@app.post("/psi/align")
def psi_align(req: PSIAlignRequest, ident: dict = Depends(ops.require("align"))) -> Dict[str, Any]:
    if any(len(i) > 64 for ids in (req.bank_ids, req.upi_ids, req.telecom_ids or []) for i in ids):
        raise HTTPException(status_code=422, detail="identifiers are limited to 64 characters")
    if req.telecom_ids is not None and len(req.telecom_ids) > 0:
        res3 = align_three_parties(req.bank_ids, req.upi_ids, req.telecom_ids)
        matched_ids = [req.bank_ids[i] for i in res3.aligned_indices_bank]
        return {
            "mode": "three_party_dh_psi",
            "matched_customer_ids_order": matched_ids,
            "result": dataclasses.asdict(res3),
            "warning": PSI_WARNING,
        }
    res2 = align_two_parties(req.bank_ids, req.upi_ids)
    matched_ids = [req.bank_ids[i] for i in res2.aligned_indices_a]
    return {
        "mode": "two_party_dh_psi",
        "matched_customer_ids_order": matched_ids,
        "result": dataclasses.asdict(res2),
        "warning": PSI_WARNING,
    }


@app.post("/train")
def train_endpoint(req: TrainSplitNNRequest, ident: dict = Depends(ops.require("train"))) -> Dict[str, Any]:
    if not _TRAIN_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="a training run is already in progress")
    try:
        return _train(req, ident["user"] or "demo")
    finally:
        _TRAIN_LOCK.release()


def _train(req: TrainSplitNNRequest, user: str) -> Dict[str, Any]:
    ds = generate_vertical_credit_dataset(n_samples=req.n_samples, seed=req.seed)
    bank_metrics, _ = train_bank_only_baseline(
        ds.x_bank_train,
        ds.y_train,
        ds.x_bank_test,
        ds.y_test,
        epochs=req.epochs,
        seed=req.seed,
    )

    unprot_model = SplitCreditNet(
        embed_dim=8,
        hidden_dim=16,
        top_hidden_dim=16,
        defense_config=CutLayerDefenseConfig(enabled=False),
        seed=req.seed,
    )
    unprot_model.fit(
        ds.x_bank_train,
        ds.x_upi_train,
        ds.x_telecom_train,
        ds.y_train,
        epochs=req.epochs,
    )
    unprot_metrics = unprot_model.evaluate(
        ds.x_bank_test,
        ds.x_upi_test,
        ds.x_telecom_test,
        ds.y_test,
        model_name="Unprotected_Vertical_SplitNN",
    )
    unprot_inv = evaluate_cut_layer_leakage(
        unprot_model,
        ds.x_bank_train,
        ds.x_upi_train,
        ds.x_telecom_train,
        ds.x_bank_test,
        ds.x_upi_test,
        ds.x_telecom_test,
        model_name="Unprotected_Vertical_SplitNN",
    )

    def_cfg = CutLayerDefenseConfig(
        enabled=True,
        clip_norm=req.clip_norm,
        decorr_lambda=req.decorr_lambda,
        noise_sigma=req.noise_sigma,
        bottleneck_rank=1,
    )
    def_model = SplitCreditNet(
        embed_dim=8,
        hidden_dim=16,
        top_hidden_dim=16,
        defense_config=def_cfg,
        seed=req.seed,
    )
    def_model.fit(
        ds.x_bank_train,
        ds.x_upi_train,
        ds.x_telecom_train,
        ds.y_train,
        epochs=req.epochs,
    )
    def_metrics = def_model.evaluate(
        ds.x_bank_test,
        ds.x_upi_test,
        ds.x_telecom_test,
        ds.y_test,
        model_name="Defended_Vertical_SplitNN",
    )
    def_inv = evaluate_cut_layer_leakage(
        def_model,
        ds.x_bank_train,
        ds.x_upi_train,
        ds.x_telecom_train,
        ds.x_bank_test,
        ds.x_upi_test,
        ds.x_telecom_test,
        model_name="Defended_Vertical_SplitNN",
    )

    version = registry.save(def_model, {k: getattr(ds, k) for k in STATS}, dataclasses.asdict(def_metrics), user)
    log.info("model trained", extra={"fields": {"version": version, "user": user}})

    return {
        "model_version": version,
        "defense_config": dataclasses.asdict(def_cfg),
        "bank_only": dataclasses.asdict(bank_metrics),
        "unprotected_splitnn": {
            "metrics": dataclasses.asdict(unprot_metrics),
            "inversion_attack": dataclasses.asdict(unprot_inv),
        },
        "defended_splitnn": {
            "metrics": dataclasses.asdict(def_metrics),
            "inversion_attack": dataclasses.asdict(def_inv),
        },
        "summary": {
            "roc_auc_gain_over_bank": round(def_metrics.roc_auc - bank_metrics.roc_auc, 4),
            "pr_auc_gain_over_bank": round(def_metrics.pr_auc - bank_metrics.pr_auc, 4),
            "roc_auc_drop_vs_unprotected": round(unprot_metrics.roc_auc - def_metrics.roc_auc, 4),
            "attacker_r2_unprotected": unprot_inv.overall_mean_r2,
            "attacker_r2_defended": def_inv.overall_mean_r2,
        },
    }


@app.post("/score")
def score_borrower(req: ScoreBorrowerRequest, ident: dict = Depends(ops.require("score"))) -> Dict[str, Any]:
    served = current_model()
    model, st = served.model, served.stats

    xb_raw = np.array(
        [[req.savings_balance, req.salary_regularity, req.bounce_count, req.credit_utilization]],
        dtype=np.float64,
    )
    xu_raw = np.array(
        [[req.merchant_spend_diversity, req.late_night_betting_share, req.p2p_velocity, req.bnpl_rollover_count]],
        dtype=np.float64,
    )
    xt_raw = np.array(
        [[req.sim_tenure_months, req.prepaid_recharge_consistency, req.electricity_ontime_ratio, req.device_price_tier]],
        dtype=np.float64,
    )

    xb = (xb_raw - st["bank_mean"]) / st["bank_std"]
    xu = (xu_raw - st["upi_mean"]) / st["upi_std"]
    xt = (xt_raw - st["telecom_mean"]) / st["telecom_std"]

    z1, z2, z3 = model.forward_cut_layers(xb, xu, xt, training=False)
    z_concat = np.hstack([z1, z2, z3])
    prob_default = float(model.top_net.forward(z_concat, training=False)[0])

    # The same top network with the other parties' embeddings zeroed. This is not a model
    # trained on bank data alone, only a rough view of how much the external parties move the score.
    z_bank_only = np.hstack([z1, np.zeros_like(z2), np.zeros_like(z3)])
    prob_bank_only = float(model.top_net.forward(z_bank_only, training=False)[0])

    credit_score = int(round(900 - 600 * prob_default))
    if prob_default < 0.20:
        tier = "PRIME_LOW_RISK"
        decision = "APPROVE_STANDARD_APR"
    elif prob_default < 0.45:
        tier = "NEAR_PRIME_MODERATE_RISK"
        decision = "APPROVE_WITH_RISK_SPREAD"
    elif prob_default < 0.70:
        tier = "SUBPRIME_ELEVATED_RISK"
        decision = "CAPPED_LIMIT_MANUAL_REVIEW"
    else:
        tier = "HIGH_RISK_DEFAULT_ALERT"
        decision = "DECLINE_CREDIT_LINE"
    registry.log_score(ident["user"] or "demo", req.customer_id, served.version, prob_default, decision)
    ops.DECISIONS.labels(decision=decision).inc()

    return {
        "customer_id": req.customer_id,
        "model_version": served.version,
        "splitnn_default_probability": round(prob_default, 4),
        "external_embeddings_zeroed_probability": round(prob_bank_only, 4),
        "multi_party_risk_shift": round(prob_default - prob_bank_only, 4),
        "bureau_credit_score": credit_score,
        "risk_tier": tier,
        "recommended_decision": decision,
        "cut_layer_telemetry": {
            "z_bank_norm": round(float(np.linalg.norm(z1)), 4),
            "z_upi_norm": round(float(np.linalg.norm(z2)), 4),
            "z_telecom_norm": round(float(np.linalg.norm(z3)), 4),
            "z_upi_embedding_preview": [round(float(v), 4) for v in z2[0, :4]],
            "z_telecom_embedding_preview": [round(float(v), 4) for v in z3[0, :4]],
        },
        "note": (
            "Demo only: this server received all three parties' raw features. The model is trained "
            "on synthetic data; the score, tier and decision are illustrative, not a credit decision."
        ),
    }


@app.get("/bench")
def get_benchmark(ident: dict = Depends(ops.require("bench"))) -> JSONResponse:
    global _cached_bench
    if _cached_bench is None:
        _cached_bench = run_full_benchmark(n_common=900, epochs=160, seed=42)
    return JSONResponse(_cached_bench.to_dict())
