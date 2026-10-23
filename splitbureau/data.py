"""
Vertically Partitioned Multi-Institution Credit Bureau Dataset Generator.

Models three financial & alternative-data institutions holding disjoint vertical
feature slices for customers identified by PAN / phone number:
  - Party 1 (Retail Bank - Coordinator & Label Holder):
      1. savings_balance      (INR Lakhs, [0.05, 25.0])
      2. salary_regularity    (consistency index, [0.0, 1.0])
      3. bounce_count         (NACH / check bounces in 12m, [0, 8])
      4. credit_utilization   (revolving line utilization, [0.02, 0.99])
      + default label y in {0, 1}
  - Party 2 (UPI Super-App):
      1. merchant_spend_diversity (normalized Shannon entropy, [0.05, 1.0])
      2. late_night_betting_share (share of 00:00-05:00 gaming/betting txns, [0.0, 0.85])
      3. p2p_velocity             (monthly P2P transfer velocity z-score, [0.0, 10.0])
      4. bnpl_rollover_count      (micro-BNPL rollover extensions in 6m, [0, 10])
  - Party 3 (Telecom & Utility):
      1. sim_tenure_months            (active SIM age in months, [1, 120])
      2. prepaid_recharge_consistency (regularity of top-ups, [0.05, 1.0])
      3. electricity_ontime_ratio     (utility bills paid before due date, [0.0, 1.0])
      4. device_price_tier            (handset tier 1..5)
"""

from __future__ import annotations

import dataclasses
from typing import List, Tuple

import numpy as np

from .psi import ThreePartyPSIResult, align_three_parties

BANK_FEATURES: Tuple[str, ...] = (
    "savings_balance",
    "salary_regularity",
    "bounce_count",
    "credit_utilization",
)

UPI_FEATURES: Tuple[str, ...] = (
    "merchant_spend_diversity",
    "late_night_betting_share",
    "p2p_velocity",
    "bnpl_rollover_count",
)

TELECOM_FEATURES: Tuple[str, ...] = (
    "sim_tenure_months",
    "prepaid_recharge_consistency",
    "electricity_ontime_ratio",
    "device_price_tier",
)


@dataclasses.dataclass
class VerticalCreditDataset:
    """Aligned vertically partitioned dataset ready for Split Neural Network training."""

    customer_ids: List[str]
    # Raw domain-scale features (N, 4)
    x_bank_raw: np.ndarray
    x_upi_raw: np.ndarray
    x_telecom_raw: np.ndarray
    # Standardized features (N, 4) per party
    x_bank_train: np.ndarray
    x_upi_train: np.ndarray
    x_telecom_train: np.ndarray
    y_train: np.ndarray
    x_bank_test: np.ndarray
    x_upi_test: np.ndarray
    x_telecom_test: np.ndarray
    y_test: np.ndarray
    # Normalization statistics for live inference scoring
    bank_mean: np.ndarray
    bank_std: np.ndarray
    upi_mean: np.ndarray
    upi_std: np.ndarray
    telecom_mean: np.ndarray
    telecom_std: np.ndarray
    default_rate: float


@dataclasses.dataclass
class UnalignedBureauRegistries:
    """
    Unaligned customer registries across the 3 institutions prior to DH-PSI,
    including non-overlapping customer records at each institution.
    """

    bank_ids: List[str]
    bank_features_raw: np.ndarray
    bank_labels: np.ndarray
    upi_ids: List[str]
    upi_features_raw: np.ndarray
    telecom_ids: List[str]
    telecom_features_raw: np.ndarray
    true_common_ids: List[str]


def _synthesize_cohort_features(
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate realistic raw features for Bank, UPI, and Telecom along with
    ground-truth 12-month credit default indicator y.
    """
    # Party 1: Retail Bank (4 features)
    savings_balance = np.clip(rng.lognormal(mean=0.8, sigma=0.75, size=n_samples), 0.05, 25.0)
    salary_regularity = np.clip(rng.beta(a=2.8, b=1.8, size=n_samples), 0.02, 0.99)
    bounce_count = np.clip(rng.poisson(lam=1.2, size=n_samples) + rng.normal(0, 0.35, size=n_samples), 0.0, 8.0)
    credit_utilization = np.clip(rng.beta(a=2.0, b=2.2, size=n_samples), 0.02, 0.99)

    x_bank_raw = np.column_stack([
        savings_balance,
        salary_regularity,
        bounce_count,
        credit_utilization,
    ])

    # Party 2: UPI Super-App (4 features)
    merchant_spend_diversity = np.clip(rng.beta(a=2.5, b=2.0, size=n_samples), 0.05, 0.99)
    late_night_betting_share = np.clip(rng.beta(a=1.2, b=3.2, size=n_samples), 0.0, 0.85)
    p2p_velocity = np.clip(rng.gamma(shape=2.2, scale=1.3, size=n_samples), 0.1, 10.0)
    bnpl_rollover_count = np.clip(
        rng.poisson(lam=1.8, size=n_samples) + rng.normal(0, 0.4, size=n_samples),
        0.0,
        10.0,
    )

    x_upi_raw = np.column_stack([
        merchant_spend_diversity,
        late_night_betting_share,
        p2p_velocity,
        bnpl_rollover_count,
    ])

    # Party 3: Telecom & Utility (4 features)
    sim_tenure_months = np.clip(rng.gamma(shape=3.0, scale=14.0, size=n_samples), 1.0, 120.0)
    prepaid_recharge_consistency = np.clip(rng.beta(a=2.6, b=1.9, size=n_samples), 0.05, 0.99)
    electricity_ontime_ratio = np.clip(rng.beta(a=2.8, b=1.7, size=n_samples), 0.02, 0.99)
    device_price_tier = np.clip(
        rng.normal(loc=3.0, scale=1.1, size=n_samples),
        1.0,
        5.0,
    )

    x_telecom_raw = np.column_stack([
        sim_tenure_months,
        prepaid_recharge_consistency,
        electricity_ontime_ratio,
        device_price_tier,
    ])

    # Standardize for structural risk score construction
    z_bank = (x_bank_raw - x_bank_raw.mean(axis=0)) / (x_bank_raw.std(axis=0) + 1e-8)
    z_upi = (x_upi_raw - x_upi_raw.mean(axis=0)) / (x_upi_raw.std(axis=0) + 1e-8)
    z_tel = (x_telecom_raw - x_telecom_raw.mean(axis=0)) / (x_telecom_raw.std(axis=0) + 1e-8)

    # Party-specific latent risk scores:
    # Balanced across all 4 features per party so no single feature dominates the party's
    # task-sufficient summary, while Bank alone has realistic moderate signal (~0.76 AUC).
    risk_bank = (
        -0.65 * z_bank[:, 0]
        - 0.65 * z_bank[:, 1]
        + 0.70 * z_bank[:, 2]
        + 0.70 * z_bank[:, 3]
    )
    risk_upi = (
        -0.85 * z_upi[:, 0]
        + 0.90 * z_upi[:, 1]
        + 0.85 * z_upi[:, 2]
        + 0.90 * z_upi[:, 3]
        + 0.35 * np.tanh(z_upi[:, 1] * z_upi[:, 3] - z_upi[:, 0] * z_upi[:, 2])
    )
    risk_tel = (
        -0.85 * z_tel[:, 0]
        - 0.90 * z_tel[:, 1]
        - 0.90 * z_tel[:, 2]
        - 0.85 * z_tel[:, 3]
        + 0.35 * np.tanh(z_tel[:, 0] * z_tel[:, 3] - z_tel[:, 1] * z_tel[:, 2])
    )

    # Cross-institution interaction: high credit utilization + BNPL rollovers + missed utility bills
    cross_interaction = (
        0.45 * np.maximum(0.0, z_bank[:, 3]) * np.maximum(0.0, z_upi[:, 3])
        + 0.40 * np.maximum(0.0, z_upi[:, 1]) * np.maximum(0.0, -z_tel[:, 2])
    )

    latent_logit = (
        -0.65
        + 1.05 * risk_bank
        + 1.35 * risk_upi
        + 1.35 * risk_tel
        + cross_interaction
        + rng.normal(0.0, 0.45, size=n_samples)
    )
    prob_default = 1.0 / (1.0 + np.exp(-latent_logit))
    y = (prob_default > 0.50).astype(np.float64)

    return x_bank_raw, x_upi_raw, x_telecom_raw, y


def generate_unaligned_registries(
    n_common: int = 900,
    n_bank_extra: int = 180,
    n_upi_extra: int = 240,
    n_telecom_extra: int = 210,
    seed: int = 42,
) -> UnalignedBureauRegistries:
    """
    Generate unaligned, independently shuffled customer registries for the 3
    institutions, where only `n_common` customers overlap across all 3 parties.
    """
    rng = np.random.default_rng(seed)
    n_total = n_common + n_bank_extra + n_upi_extra + n_telecom_extra
    xb, xu, xt, y = _synthesize_cohort_features(n_total, rng)

    common_ids = [f"PAN_COMMON_{i:05d}" for i in range(n_common)]
    bank_only_ids = [f"PAN_BANK_ONLY_{i:05d}" for i in range(n_bank_extra)]
    upi_only_ids = [f"PAN_UPI_ONLY_{i:05d}" for i in range(n_upi_extra)]
    tel_only_ids = [f"PAN_TEL_ONLY_{i:05d}" for i in range(n_telecom_extra)]

    # Slice cohorts
    c_end = n_common
    b_end = c_end + n_bank_extra
    u_end = b_end + n_upi_extra

    bank_ids = common_ids + bank_only_ids
    bank_feat = np.vstack([xb[:c_end], xb[c_end:b_end]])
    bank_y = np.concatenate([y[:c_end], y[c_end:b_end]])

    upi_ids = common_ids + upi_only_ids
    upi_feat = np.vstack([xu[:c_end], xu[b_end:u_end]])

    tel_ids = common_ids + tel_only_ids
    tel_feat = np.vstack([xt[:c_end], xt[u_end:]])

    # Independently permute each party's local database rows
    perm_bank = rng.permutation(len(bank_ids))
    perm_upi = rng.permutation(len(upi_ids))
    perm_tel = rng.permutation(len(tel_ids))

    return UnalignedBureauRegistries(
        bank_ids=[bank_ids[i] for i in perm_bank],
        bank_features_raw=bank_feat[perm_bank],
        bank_labels=bank_y[perm_bank],
        upi_ids=[upi_ids[i] for i in perm_upi],
        upi_features_raw=upi_feat[perm_upi],
        telecom_ids=[tel_ids[i] for i in perm_tel],
        telecom_features_raw=tel_feat[perm_tel],
        true_common_ids=sorted(common_ids),
    )


def build_aligned_dataset_from_registries(
    registries: UnalignedBureauRegistries,
    test_fraction: float = 0.25,
    seed: int = 42,
) -> Tuple[VerticalCreditDataset, ThreePartyPSIResult]:
    """
    Execute 3-party DH-PSI on the unaligned registries and construct the
    standardized train/test vertical dataset on the aligned intersection.
    """
    psi_res = align_three_parties(
        registries.bank_ids,
        registries.upi_ids,
        registries.telecom_ids,
    )

    idx_b = psi_res.aligned_indices_bank
    idx_u = psi_res.aligned_indices_upi
    idx_t = psi_res.aligned_indices_telecom

    aligned_ids = [registries.bank_ids[i] for i in idx_b]
    xb_raw = registries.bank_features_raw[idx_b]
    xu_raw = registries.upi_features_raw[idx_u]
    xt_raw = registries.telecom_features_raw[idx_t]
    y_all = registries.bank_labels[idx_b]

    dataset = _split_and_standardize(
        customer_ids=aligned_ids,
        xb_raw=xb_raw,
        xu_raw=xu_raw,
        xt_raw=xt_raw,
        y_all=y_all,
        test_fraction=test_fraction,
        seed=seed,
    )
    return dataset, psi_res


def _split_and_standardize(
    customer_ids: List[str],
    xb_raw: np.ndarray,
    xu_raw: np.ndarray,
    xt_raw: np.ndarray,
    y_all: np.ndarray,
    test_fraction: float = 0.25,
    seed: int = 42,
) -> VerticalCreditDataset:
    rng = np.random.default_rng(seed)
    n = len(y_all)
    perm = rng.permutation(n)
    n_test = max(int(round(n * test_fraction)), 50)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    xb_tr_raw, xb_te_raw = xb_raw[train_idx], xb_raw[test_idx]
    xu_tr_raw, xu_te_raw = xu_raw[train_idx], xu_raw[test_idx]
    xt_tr_raw, xt_te_raw = xt_raw[train_idx], xt_raw[test_idx]
    y_tr, y_te = y_all[train_idx], y_all[test_idx]

    bank_mean = xb_tr_raw.mean(axis=0)
    bank_std = xb_tr_raw.std(axis=0) + 1e-8
    upi_mean = xu_tr_raw.mean(axis=0)
    upi_std = xu_tr_raw.std(axis=0) + 1e-8
    tel_mean = xt_tr_raw.mean(axis=0)
    tel_std = xt_tr_raw.std(axis=0) + 1e-8

    return VerticalCreditDataset(
        customer_ids=customer_ids,
        x_bank_raw=xb_raw,
        x_upi_raw=xu_raw,
        x_telecom_raw=xt_raw,
        x_bank_train=(xb_tr_raw - bank_mean) / bank_std,
        x_upi_train=(xu_tr_raw - upi_mean) / upi_std,
        x_telecom_train=(xt_tr_raw - tel_mean) / tel_std,
        y_train=y_tr,
        x_bank_test=(xb_te_raw - bank_mean) / bank_std,
        x_upi_test=(xu_te_raw - upi_mean) / upi_std,
        x_telecom_test=(xt_te_raw - tel_mean) / tel_std,
        y_test=y_te,
        bank_mean=bank_mean,
        bank_std=bank_std,
        upi_mean=upi_mean,
        upi_std=upi_std,
        telecom_mean=tel_mean,
        telecom_std=tel_std,
        default_rate=float(np.mean(y_all)),
    )


def generate_vertical_credit_dataset(
    n_samples: int = 900,
    test_fraction: float = 0.25,
    seed: int = 42,
) -> VerticalCreditDataset:
    """
    Generate an aligned 3-party vertically partitioned credit bureau dataset.
    """
    rng = np.random.default_rng(seed)
    xb_raw, xu_raw, xt_raw, y_all = _synthesize_cohort_features(n_samples, rng)
    cids = [f"PAN_COMMON_{i:05d}" for i in range(n_samples)]
    return _split_and_standardize(
        customer_ids=cids,
        xb_raw=xb_raw,
        xu_raw=xu_raw,
        xt_raw=xt_raw,
        y_all=y_all,
        test_fraction=test_fraction,
        seed=seed,
    )
