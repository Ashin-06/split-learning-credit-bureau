"""
Cut-Layer Feature-Reconstruction (Model Inversion) Attack against Vertical SplitNN.

Models a semi-honest Coordinator (Party 1) attempting to invert the transmitted
cut-layer embeddings z_k in R^d of external institutions:
  - Party 2 (UPI Super-App): 4 private features
  - Party 3 (Telecom & Utility): 4 private features

The attacker fits two decoders D_k: R^d -> R^4 on observed cut-layer embeddings of training
records whose raw features it knows (a strong assumption that favours the attacker): a ridge
regression on a quadratic basis and a small MLP. Each party's leakage is the better of the two
on held-out records. Neither is the strongest possible attack, so a low R^2 bounds leakage only
against these decoders.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

from .data import TELECOM_FEATURES, UPI_FEATURES
from .splitnn import SplitCreditNet


@dataclasses.dataclass(frozen=True)
class InversionAttackReport:
    """Detailed feature-inversion attack metrics for external parties (UPI & Telecom)."""

    model_name: str
    party_upi_r2: float
    party_telecom_r2: float
    overall_mean_r2: float
    overall_mse: float
    per_feature_r2: Dict[str, float]
    privacy_verdict: str


class FeatureInversionAttack:
    """
    Regularized Ridge + Quadratic Basis Decoder that reconstructs raw private
    features x_k from transmitted cut-layer embeddings z_k.
    """

    def __init__(self, ridge_alpha: float = 2.0, include_quadratic: bool = True) -> None:
        self.ridge_alpha = ridge_alpha
        self.include_quadratic = include_quadratic
        self.weights_: np.ndarray = np.empty((0, 0))
        self.feat_mean_: np.ndarray = np.empty((0,))
        self.feat_std_: np.ndarray = np.empty((0,))

    def _expand_basis(self, z: np.ndarray) -> np.ndarray:
        if self.include_quadratic:
            return np.hstack([z, z ** 2, np.tanh(z)])
        return z

    def fit(self, z_train: np.ndarray, x_train: np.ndarray) -> "FeatureInversionAttack":
        phi = self._expand_basis(z_train)
        self.feat_mean_ = phi.mean(axis=0, keepdims=True)
        self.feat_std_ = phi.std(axis=0, keepdims=True) + 1e-6
        phi_norm = (phi - self.feat_mean_) / self.feat_std_

        n, m = phi_norm.shape
        design = np.hstack([phi_norm, np.ones((n, 1))])
        reg = self.ridge_alpha * np.eye(m + 1)
        reg[-1, -1] = 1e-6  # do not regularize intercept

        # Closed-form Ridge regression: (Phi^T Phi + alpha I)^{-1} Phi^T X
        self.weights_ = np.linalg.solve(design.T @ design + reg, design.T @ x_train)
        return self

    def reconstruct(self, z_test: np.ndarray) -> np.ndarray:
        phi = self._expand_basis(z_test)
        phi_norm = (phi - self.feat_mean_) / self.feat_std_
        design = np.hstack([phi_norm, np.ones((z_test.shape[0], 1))])
        return design @ self.weights_

    def score_reconstruction(
        self,
        z_test: np.ndarray,
        x_test: np.ndarray,
        feature_names: Sequence[str],
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Return (mean_r2, mean_mse, per_feature_r2_dict).
        """
        x_hat = self.reconstruct(z_test)
        per_feat: Dict[str, float] = {}
        r2_vals = []
        for idx, name in enumerate(feature_names):
            raw_r2 = float(r2_score(x_test[:, idx], x_hat[:, idx]))
            # Clamp negative out-of-sample R^2 to 0.0 for interpretable variance-explained metric
            clamped_r2 = max(0.0, raw_r2)
            per_feat[name] = round(clamped_r2, 4)
            r2_vals.append(clamped_r2)

        mean_r2 = float(np.mean(r2_vals))
        mse = float(mean_squared_error(x_test, x_hat))
        return round(mean_r2, 4), round(mse, 4), per_feat


class MLPInversionAttack(FeatureInversionAttack):
    """Two-hidden-layer MLP decoder from cut-layer embeddings to raw features."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.mlp_ = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=seed)

    def fit(self, z_train: np.ndarray, x_train: np.ndarray) -> "MLPInversionAttack":
        self.mlp_.fit(z_train, x_train)
        return self

    def reconstruct(self, z_test: np.ndarray) -> np.ndarray:
        return self.mlp_.predict(z_test)


def _strongest_attack(
    z_tr: np.ndarray,
    x_tr: np.ndarray,
    z_te: np.ndarray,
    x_te: np.ndarray,
    feature_names: Sequence[str],
    ridge_alpha: float,
) -> Tuple[float, float, Dict[str, float]]:
    results = [
        FeatureInversionAttack(ridge_alpha=ridge_alpha).fit(z_tr, x_tr).score_reconstruction(z_te, x_te, feature_names),
        MLPInversionAttack().fit(z_tr, x_tr).score_reconstruction(z_te, x_te, feature_names),
    ]
    return max(results, key=lambda r: r[0])


def evaluate_cut_layer_leakage(
    model: SplitCreditNet,
    x_bank_train: np.ndarray,
    x_upi_train: np.ndarray,
    x_telecom_train: np.ndarray,
    x_bank_test: np.ndarray,
    x_upi_test: np.ndarray,
    x_telecom_test: np.ndarray,
    model_name: str = "SplitCreditNet",
    ridge_alpha: float = 2.0,
) -> InversionAttackReport:
    """
    Run the semi-honest Coordinator Feature-Inversion Attack on Party 2 (UPI)
    and Party 3 (Telecom) cut-layer embeddings and report reconstruction R^2.
    """
    _, z2_tr, z3_tr = model.forward_cut_layers(
        x_bank_train, x_upi_train, x_telecom_train, training=False
    )
    _, z2_te, z3_te = model.forward_cut_layers(
        x_bank_test, x_upi_test, x_telecom_test, training=False
    )

    upi_r2, upi_mse, upi_per_feat = _strongest_attack(
        z2_tr, x_upi_train, z2_te, x_upi_test, UPI_FEATURES, ridge_alpha
    )
    tel_r2, tel_mse, tel_per_feat = _strongest_attack(
        z3_tr, x_telecom_train, z3_te, x_telecom_test, TELECOM_FEATURES, ridge_alpha
    )

    overall_r2 = round(0.5 * (upi_r2 + tel_r2), 4)
    overall_mse = round(0.5 * (upi_mse + tel_mse), 4)

    combined_per_feat: Dict[str, float] = {}
    for k, v in upi_per_feat.items():
        combined_per_feat[f"upi.{k}"] = v
    for k, v in tel_per_feat.items():
        combined_per_feat[f"telecom.{k}"] = v

    if overall_r2 < 0.12:
        verdict = "STRONG_DEFENSE (R2 < 0.12)"
    elif overall_r2 < 0.45:
        verdict = "MODERATE_LEAKAGE"
    else:
        verdict = "CRITICAL_LEAKAGE (R2 > 0.80)" if overall_r2 > 0.80 else "HIGH_LEAKAGE"

    return InversionAttackReport(
        model_name=model_name,
        party_upi_r2=upi_r2,
        party_telecom_r2=tel_r2,
        overall_mean_r2=overall_r2,
        overall_mse=overall_mse,
        per_feature_r2=combined_per_feat,
        privacy_verdict=verdict,
    )
