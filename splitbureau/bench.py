"""
End-to-End Benchmark Suite for Vertically Partitioned DH-PSI & Split-Learning
Credit Bureau with Cut-Layer Feature-Inversion Defense.

Benchmarks:
  1. Elliptic-curve Diffie-Hellman PSI alignment across 3 institutions
     (checked against the plaintext IDs, which only a simulation can do).
  2. Single-Party (Retail Bank Alone) vs Unprotected 3-Party Vertical SplitNN
     vs Privacy-Defended 3-Party Vertical SplitNN vs Centralized Oracle MLP.
  3. Semi-honest Coordinator Feature-Inversion Attack R^2 for the unprotected and the
     defended model, alongside the accuracy each one gives up.

All data is synthetic (see data.py).
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Any, Dict

from .data import (
    build_aligned_dataset_from_registries,
    generate_unaligned_registries,
)
from .inversion import InversionAttackReport, evaluate_cut_layer_leakage
from .psi import ThreePartyPSIResult
from .splitnn import (
    CutLayerDefenseConfig,
    SplitCreditNet,
    SplitNNMetrics,
    train_bank_only_baseline,
    train_centralized_oracle,
)


@dataclasses.dataclass
class BenchmarkSummary:
    """Comprehensive benchmark results across PSI, SplitNN models, and Inversion attacks."""

    psi_result: ThreePartyPSIResult
    bank_only_metrics: SplitNNMetrics
    unprotected_splitnn_metrics: SplitNNMetrics
    defended_splitnn_metrics: SplitNNMetrics
    centralized_oracle_metrics: SplitNNMetrics
    unprotected_inversion: InversionAttackReport
    defended_inversion: InversionAttackReport
    roc_auc_gain_over_bank: float
    pr_auc_gain_over_bank: float
    roc_auc_drop_vs_unprotected: float
    inversion_r2_reduction: float
    all_assertions_passed: bool
    elapsed_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def run_full_benchmark(
    n_common: int = 900,
    epochs: int = 160,
    seed: int = 42,
) -> BenchmarkSummary:
    """
    Execute the full 3-party DH-PSI + Vertical SplitNN + Feature-Inversion
    benchmark suite.
    """
    t0 = time.perf_counter()

    # 1. Generate unaligned multi-institution customer registries and align via DH-PSI
    registries = generate_unaligned_registries(
        n_common=n_common,
        n_bank_extra=180,
        n_upi_extra=240,
        n_telecom_extra=210,
        seed=seed,
    )
    dataset, psi_res = build_aligned_dataset_from_registries(
        registries, test_fraction=0.25, seed=seed
    )

    # 2. Single-Party Baseline (Bank Alone - 4 features)
    bank_metrics, _ = train_bank_only_baseline(
        dataset.x_bank_train,
        dataset.y_train,
        dataset.x_bank_test,
        dataset.y_test,
        epochs=epochs,
        seed=seed,
    )

    # 3. Unprotected 3-Party Vertical SplitNN (no clipping, no decorr, no noise)
    unprotected_cfg = CutLayerDefenseConfig(
        enabled=False,
        clip_norm=100.0,
        decorr_lambda=0.0,
        noise_sigma=0.0,
    )
    unprotected_model = SplitCreditNet(
        embed_dim=8,
        hidden_dim=16,
        top_hidden_dim=16,
        defense_config=unprotected_cfg,
        seed=seed,
    )
    unprotected_model.fit(
        dataset.x_bank_train,
        dataset.x_upi_train,
        dataset.x_telecom_train,
        dataset.y_train,
        epochs=epochs,
        batch_size=64,
        lr=0.025,
    )
    unprotected_metrics = unprotected_model.evaluate(
        dataset.x_bank_test,
        dataset.x_upi_test,
        dataset.x_telecom_test,
        dataset.y_test,
        model_name="Unprotected_Vertical_SplitNN",
    )
    unprotected_inv = evaluate_cut_layer_leakage(
        unprotected_model,
        dataset.x_bank_train,
        dataset.x_upi_train,
        dataset.x_telecom_train,
        dataset.x_bank_test,
        dataset.x_upi_test,
        dataset.x_telecom_test,
        model_name="Unprotected_Vertical_SplitNN",
    )

    # 4. Privacy-Defended 3-Party Vertical SplitNN
    #    (rank-1 bottleneck + L2 norm clipping + cross-covariance penalty + Gaussian noise)
    defended_cfg = CutLayerDefenseConfig(
        enabled=True,
        clip_norm=0.95,
        decorr_lambda=0.55,
        noise_sigma=0.45,
        bottleneck_rank=1,
    )
    defended_model = SplitCreditNet(
        embed_dim=8,
        hidden_dim=16,
        top_hidden_dim=16,
        defense_config=defended_cfg,
        seed=seed,
    )
    defended_model.fit(
        dataset.x_bank_train,
        dataset.x_upi_train,
        dataset.x_telecom_train,
        dataset.y_train,
        epochs=epochs,
        batch_size=64,
        lr=0.025,
    )
    defended_metrics = defended_model.evaluate(
        dataset.x_bank_test,
        dataset.x_upi_test,
        dataset.x_telecom_test,
        dataset.y_test,
        model_name="Defended_Vertical_SplitNN",
    )
    defended_inv = evaluate_cut_layer_leakage(
        defended_model,
        dataset.x_bank_train,
        dataset.x_upi_train,
        dataset.x_telecom_train,
        dataset.x_bank_test,
        dataset.x_upi_test,
        dataset.x_telecom_test,
        model_name="Defended_Vertical_SplitNN",
    )

    # 5. Centralized Oracle MLP (Upper Bound - Pools All 12 Raw Features)
    oracle_metrics, _ = train_centralized_oracle(
        dataset.x_bank_train,
        dataset.x_upi_train,
        dataset.x_telecom_train,
        dataset.y_train,
        dataset.x_bank_test,
        dataset.x_upi_test,
        dataset.x_telecom_test,
        dataset.y_test,
        epochs=epochs,
        seed=seed,
    )

    roc_gain = round(defended_metrics.roc_auc - bank_metrics.roc_auc, 4)
    pr_gain = round(defended_metrics.pr_auc - bank_metrics.pr_auc, 4)
    roc_drop = round(unprotected_metrics.roc_auc - defended_metrics.roc_auc, 4)
    r2_reduction = round(
        unprotected_inv.overall_mean_r2 - defended_inv.overall_mean_r2, 4
    )

    # Sanity checks only: the PSI found exactly the common customers, the defence reduced
    # leakage, and the three-party model still beats the bank on its own.
    assertions_ok = bool(
        psi_res.exact_recovery_rate == 1.0
        and psi_res.intersection_size == n_common
        and defended_inv.overall_mean_r2 < unprotected_inv.overall_mean_r2
        and roc_gain > 0.0
    )

    elapsed_sec = round(time.perf_counter() - t0, 3)

    return BenchmarkSummary(
        psi_result=psi_res,
        bank_only_metrics=bank_metrics,
        unprotected_splitnn_metrics=unprotected_metrics,
        defended_splitnn_metrics=defended_metrics,
        centralized_oracle_metrics=oracle_metrics,
        unprotected_inversion=unprotected_inv,
        defended_inversion=defended_inv,
        roc_auc_gain_over_bank=roc_gain,
        pr_auc_gain_over_bank=pr_gain,
        roc_auc_drop_vs_unprotected=roc_drop,
        inversion_r2_reduction=r2_reduction,
        all_assertions_passed=assertions_ok,
        elapsed_sec=elapsed_sec,
    )


ABLATION_CONFIGS = {
    "no defence": dict(enabled=False),
    "rank-1 bottleneck only": dict(enabled=True, clip_norm=100.0, decorr_lambda=0.0, noise_sigma=0.0),
    "bottleneck + clipping + penalty": dict(enabled=True, noise_sigma=0.0),
    "bottleneck + noise only": dict(enabled=True, clip_norm=100.0, decorr_lambda=0.0),
    "full defence": dict(enabled=True),
}


def run_ablation(n_common: int = 900, epochs: int = 160, seed: int = 42) -> Dict[str, Dict[str, float]]:
    """Which part of the cut-layer defence buys the privacy, and what it costs in ROC-AUC."""
    registries = generate_unaligned_registries(n_common=n_common, seed=seed)
    ds, _ = build_aligned_dataset_from_registries(registries, test_fraction=0.25, seed=seed)
    out: Dict[str, Dict[str, float]] = {}
    for name, kw in ABLATION_CONFIGS.items():
        model = SplitCreditNet(defense_config=CutLayerDefenseConfig(**kw), seed=seed)
        model.fit(ds.x_bank_train, ds.x_upi_train, ds.x_telecom_train, ds.y_train, epochs=epochs)
        metrics = model.evaluate(ds.x_bank_test, ds.x_upi_test, ds.x_telecom_test, ds.y_test)
        leak = evaluate_cut_layer_leakage(
            model, ds.x_bank_train, ds.x_upi_train, ds.x_telecom_train,
            ds.x_bank_test, ds.x_upi_test, ds.x_telecom_test,
        )
        out[name] = {"roc_auc": metrics.roc_auc, "attacker_r2": leak.overall_mean_r2}
    return out


def format_ablation(rows: Dict[str, Dict[str, float]]) -> str:
    lines = [f"  {'configuration':<34} {'ROC-AUC':>8} {'attacker R^2':>13}"]
    for name, r in rows.items():
        lines.append(f"  {name:<34} {r['roc_auc']:>8.4f} {r['attacker_r2']:>13.4f}")
    return "\n".join(lines)


def format_benchmark_report(summary: BenchmarkSummary) -> str:
    """Format a human-readable ASCII table & JSON report of the benchmark run."""
    lines = [
        "=" * 86,
        "  SPLITBUREAU: 3-PARTY DH-PSI & VERTICAL SPLIT-LEARNING CREDIT BUREAU BENCHMARK",
        "=" * 86,
        "",
        "[1] ELLIPTIC-CURVE DIFFIE-HELLMAN PRIVATE SET INTERSECTION (DH-PSI)",
        f"    Party 1 (Bank) Raw Records     : {summary.psi_result.bank_count}",
        f"    Party 2 (UPI App) Raw Records  : {summary.psi_result.upi_count}",
        f"    Party 3 (Telecom) Raw Records  : {summary.psi_result.telecom_count}",
        f"    3-Way Aligned Intersection     : {summary.psi_result.intersection_size}",
        f"    Exact Intersection Recovery    : {summary.psi_result.exact_recovery_rate * 100:.2f}%",
        f"    DH-PSI Alignment Latency       : {summary.psi_result.elapsed_ms:.2f} ms",
        "",
        "[2] CREDIT DEFAULT PREDICTION vs CUT-LAYER FEATURE-INVERSION ATTACK",
        "-" * 86,
        f"{'Model Architecture':<30} | {'ROC-AUC':>8} | {'PR-AUC':>8} | {'Brier':>7} | {'Inversion R2':>13} | {'dCor(X,Z)':>9}",
        "-" * 86,
        (
            f"{'1. Bank-Only (Single Party)':<30} | "
            f"{summary.bank_only_metrics.roc_auc:>8.4f} | "
            f"{summary.bank_only_metrics.pr_auc:>8.4f} | "
            f"{summary.bank_only_metrics.brier_score:>7.4f} | "
            f"{'N/A (Local)':>13} | {'0.0000':>9}"
        ),
        (
            f"{'2. Unprotected Vertical SplitNN':<30} | "
            f"{summary.unprotected_splitnn_metrics.roc_auc:>8.4f} | "
            f"{summary.unprotected_splitnn_metrics.pr_auc:>8.4f} | "
            f"{summary.unprotected_splitnn_metrics.brier_score:>7.4f} | "
            f"{summary.unprotected_inversion.overall_mean_r2:>13.4f} | "
            f"{0.5 * (summary.unprotected_splitnn_metrics.distance_corr_upi + summary.unprotected_splitnn_metrics.distance_corr_telecom):>9.4f}"
        ),
        (
            f"{'3. Defended Vertical SplitNN':<30} | "
            f"{summary.defended_splitnn_metrics.roc_auc:>8.4f} | "
            f"{summary.defended_splitnn_metrics.pr_auc:>8.4f} | "
            f"{summary.defended_splitnn_metrics.brier_score:>7.4f} | "
            f"{summary.defended_inversion.overall_mean_r2:>13.4f} | "
            f"{0.5 * (summary.defended_splitnn_metrics.distance_corr_upi + summary.defended_splitnn_metrics.distance_corr_telecom):>9.4f}"
        ),
        (
            f"{'4. Centralized Oracle (12-Feat)':<30} | "
            f"{summary.centralized_oracle_metrics.roc_auc:>8.4f} | "
            f"{summary.centralized_oracle_metrics.pr_auc:>8.4f} | "
            f"{summary.centralized_oracle_metrics.brier_score:>7.4f} | "
            f"{'1.0000 (Raw)':>13} | {'1.0000':>9}"
        ),
        "-" * 86,
        "",
        "[3] PRIVACY-UTILITY TRADEOFF VERIFICATION",
        f"    ROC-AUC Gain (Defended SplitNN vs Bank-Only) : +{summary.roc_auc_gain_over_bank:.4f}",
        f"    PR-AUC Gain  (Defended SplitNN vs Bank-Only) : +{summary.pr_auc_gain_over_bank:.4f}",
        f"    ROC-AUC Drop (Defended vs Unprotected)       : {summary.roc_auc_drop_vs_unprotected:+.4f}",
        f"    Unprotected Attacker Reconstruction R^2      : {summary.unprotected_inversion.overall_mean_r2:.4f}",
        f"    Defended Attacker Reconstruction R^2         : {summary.defended_inversion.overall_mean_r2:.4f}",
        f"    Attacker R^2 Reduced By                      : {summary.inversion_r2_reduction:.4f}",
        f"    Sanity Checks Passed                         : {summary.all_assertions_passed}",
        "=" * 86,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    res = run_full_benchmark()
    print(format_benchmark_report(res))
    print(json.dumps(res.to_dict(), indent=2))
