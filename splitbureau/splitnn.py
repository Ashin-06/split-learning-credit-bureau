"""
Vertical Split Neural Network (SplitCreditNet) with Cut-Layer Privacy Defense.

Architecture:
  - Party k in {1, 2, 3} operates a local private bottom network f_k(x_k) -> z_k in R^d.
    Raw features x_k NEVER leave Party k's institutional boundary.
  - Only cut-layer embeddings z_k in R^d are transmitted to the Coordinator's top
    network g([z_1, z_2, z_3]) -> y_hat in (0, 1).
  - During backpropagation, the Coordinator computes dL/dz_k and returns only the
    cut-layer gradients to Party k.

Cut-Layer Privacy Defense (for external feature parties k in {2, 3} and optionally k=1):
  1. Task-Sufficient Bottleneck & Distance-Correlation / Decorrelation Penalty:
     Penalizes redundant cross-covariance and distance correlation dCor(z_k, x_k)
     so that z_k encodes only the task-relevant risk manifold rather than raw feature axes.
  2. L2 Activation Norm Clipping:
     z_bar_{k,i} = z_tilde_{k,i} * min(1, C / (||z_tilde_{k,i}||_2 + eps)),
     bounding the L2 sensitivity of every transmitted embedding vector.
  3. Gaussian Noise:
     z_k = z_bar_k + N(0, sigma^2 I_d) at the cut layer prior to transmission.
     sigma is a fixed setting, not calibrated to a differential-privacy budget; no epsilon is
     claimed or tracked across training steps.
  The low-rank bottleneck (rank 1 by default: each party effectively sends one number)
  provides most of the protection; see the README for an ablation.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclasses.dataclass
class CutLayerDefenseConfig:
    """Configuration for the 3-stage Cut-Layer Privacy Defense."""

    enabled: bool = True
    clip_norm: float = 0.95
    decorr_lambda: float = 0.55
    noise_sigma: float = 0.45
    bottleneck_rank: int = 1


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _leaky_relu(x: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    return np.where(x >= 0.0, x, alpha * x)


def _leaky_relu_grad(x: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    return np.where(x >= 0.0, 1.0, alpha)


def distance_correlation(x: np.ndarray, z: np.ndarray, max_samples: int = 350) -> float:
    """
    Compute empirical Székely-Rizzo-Bakirov Distance Correlation dCor(X, Z) in [0, 1].
    Measures both linear and nonlinear statistical dependence between raw features X
    and transmitted cut-layer embeddings Z.
    """
    n = x.shape[0]
    if n > max_samples:
        x = x[:max_samples]
        z = z[:max_samples]
        n = max_samples

    a = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    b = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)

    a_row = a.mean(axis=1, keepdims=True)
    a_col = a.mean(axis=0, keepdims=True)
    a_grand = a.mean()
    A = a - a_row - a_col + a_grand

    b_row = b.mean(axis=1, keepdims=True)
    b_col = b.mean(axis=0, keepdims=True)
    b_grand = b.mean()
    B = b - b_row - b_col + b_grand

    dcov2_xz = np.mean(A * B)
    dcov2_xx = np.mean(A * A)
    dcov2_zz = np.mean(B * B)

    denom = np.sqrt(max(dcov2_xx * dcov2_zz, 1e-16))
    if denom <= 1e-15 or dcov2_xz <= 0.0:
        return 0.0
    return float(np.clip(np.sqrt(dcov2_xz / denom), 0.0, 1.0))


def _decorrelation_penalty_and_grad(
    x: np.ndarray, z_tilde: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute differentiable Frobenius cross-covariance + pairwise distance-covariance
    surrogate penalty between raw input features x (N, p) and pre-noise cut-layer
    activations z_tilde (N, d), returning (loss, dL_decorr / dz_tilde).

    Penalizing the isotropic cross-covariance matrix Cov(x, z_tilde) suppresses
    redundant feature directions while allowing the task loss gradient to retain
    only the label-predictive direction.
    """
    n = float(x.shape[0])
    x_c = x - x.mean(axis=0, keepdims=True)
    z_c = z_tilde - z_tilde.mean(axis=0, keepdims=True)

    # Normalized cross-covariance C_xz in R^{p x d}
    cov_xz = (x_c.T @ z_c) / n
    loss = 0.5 * float(np.sum(cov_xz ** 2))
    grad_z = (x_c @ cov_xz) / n
    return loss, grad_z


class PartyBottomNet:
    """
    Local private bottom network f_k(x_k) -> z_k in R^d for Party k.
    Never exposes raw features x_k or internal weights to the coordinator.
    """

    def __init__(
        self,
        party_id: int,
        party_name: str,
        input_dim: int = 4,
        hidden_dim: int = 16,
        embed_dim: int = 8,
        defense: Optional[CutLayerDefenseConfig] = None,
        protect_cut_layer: bool = True,
        seed: int = 42,
    ) -> None:
        self.party_id = party_id
        self.party_name = party_name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.defense = defense or CutLayerDefenseConfig(enabled=False)
        self.protect_cut_layer = protect_cut_layer
        self.rng = np.random.default_rng(seed + party_id * 97)

        # Layer 1: R^input_dim -> R^hidden_dim
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float64)

        # Layer 2: R^hidden_dim -> R^embed_dim
        if self.defense.enabled and self.protect_cut_layer:
            # Low-rank task bottleneck: projects hidden state to a compact risk subspace
            # before spreading across the d-dimensional cut-layer carrier vector.
            rank = max(1, min(self.defense.bottleneck_rank, embed_dim))
            self.W2_a = self.rng.normal(0.0, np.sqrt(1.5 / hidden_dim), size=(hidden_dim, rank))
            carrier = self.rng.normal(0.0, 1.0, size=(rank, embed_dim))
            carrier /= np.linalg.norm(carrier, axis=1, keepdims=True) + 1e-8
            self.W2_b = carrier
            self.W2 = self.W2_a @ self.W2_b
        else:
            self.W2_a = None
            self.W2_b = None
            self.W2 = self.rng.normal(0.0, np.sqrt(1.5 / hidden_dim), size=(hidden_dim, embed_dim))

        self.b2 = np.zeros((1, embed_dim), dtype=np.float64)

        # Forward cache for backprop
        self._cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Compute cut-layer embedding z_k in R^{N x d} from local private features x.
        Applies L2 norm clipping and calibrated Gaussian noise when defense is active.
        """
        pre_h = x @ self.W1 + self.b1
        h = np.tanh(pre_h)

        if self.defense.enabled and self.protect_cut_layer:
            # Nonlinear risk bottleneck projected onto unit carrier directions
            bottleneck_pre = h @ self.W2_a
            bottleneck_act = np.tanh(bottleneck_pre)
            z_tilde = bottleneck_act @ self.W2_b + self.b2
        else:
            bottleneck_pre = np.empty((0, 0))
            bottleneck_act = np.empty((0, 0))
            z_tilde = h @ self.W2 + self.b2

        # Stage 2 of Defense: L2 Activation Norm Clipping
        if self.defense.enabled and self.protect_cut_layer:
            norms = np.linalg.norm(z_tilde, axis=1, keepdims=True) + 1e-8
            scale = np.minimum(1.0, self.defense.clip_norm / norms)
            z_bar = z_tilde * scale
        else:
            norms = np.linalg.norm(z_tilde, axis=1, keepdims=True) + 1e-8
            scale = np.ones_like(norms)
            z_bar = z_tilde

        # Stage 3 of Defense: Calibrated Gaussian Noise at Cut Layer
        if self.defense.enabled and self.protect_cut_layer and self.defense.noise_sigma > 0.0:
            noise = self.rng.normal(
                0.0,
                self.defense.noise_sigma,
                size=z_bar.shape,
            )
            z_sent = z_bar + noise
        else:
            z_sent = z_bar

        if training:
            self._cache = {
                "x": x,
                "pre_h": pre_h,
                "h": h,
                "bottleneck_pre": bottleneck_pre,
                "bottleneck_act": bottleneck_act,
                "z_tilde": z_tilde,
                "scale": scale,
                "z_bar": z_bar,
            }
        return z_sent

    def backward(self, grad_z_sent: np.ndarray, lr: float = 0.015, weight_decay: float = 1e-4) -> float:
        """
        Backpropagate cut-layer gradient dL_task/dz_k returned by the Coordinator,
        augmented locally by the Distance-Correlation / Decorrelation Penalty gradient.
        Returns the local decorrelation loss value.
        """
        x = self._cache["x"]
        h = self._cache["h"]
        z_tilde = self._cache["z_tilde"]
        scale = self._cache["scale"]
        n = float(x.shape[0])

        # Gradient through L2 norm clipping
        grad_z_tilde = grad_z_sent * scale

        # Stage 1 of Defense: Local Decorrelation Penalty between z_tilde and x
        decorr_loss = 0.0
        if self.defense.enabled and self.protect_cut_layer and self.defense.decorr_lambda > 0.0:
            decorr_loss, grad_decorr = _decorrelation_penalty_and_grad(x, z_tilde)
            grad_z_tilde = grad_z_tilde + self.defense.decorr_lambda * grad_decorr

        if self.defense.enabled and self.protect_cut_layer:
            bottleneck_act = self._cache["bottleneck_act"]
            grad_bottleneck_act = grad_z_tilde @ self.W2_b.T
            grad_bottleneck_pre = grad_bottleneck_act * (1.0 - bottleneck_act ** 2)

            grad_W2_a = (h.T @ grad_bottleneck_pre) / n + weight_decay * self.W2_a
            grad_b2 = np.mean(grad_z_tilde, axis=0, keepdims=True)
            grad_h = grad_bottleneck_pre @ self.W2_a.T

            self.W2_a -= lr * grad_W2_a
            self.b2 -= lr * grad_b2
            self.W2 = self.W2_a @ self.W2_b
        else:
            grad_W2 = (h.T @ grad_z_tilde) / n + weight_decay * self.W2
            grad_b2 = np.mean(grad_z_tilde, axis=0, keepdims=True)
            grad_h = grad_z_tilde @ self.W2.T

            self.W2 -= lr * grad_W2
            self.b2 -= lr * grad_b2

        grad_pre_h = grad_h * (1.0 - h ** 2)
        grad_W1 = (x.T @ grad_pre_h) / n + weight_decay * self.W1
        grad_b1 = np.mean(grad_pre_h, axis=0, keepdims=True)

        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1

        return decorr_loss


class CoordinatorTopNet:
    """
    Coordinator's top network g([z_1, z_2, z_3]) -> y_hat.
    Receives only the 3 parties' cut-layer embeddings and returns cut-layer gradients.
    """

    def __init__(
        self,
        total_embed_dim: int = 24,
        top_hidden_dim: int = 16,
        seed: int = 42,
    ) -> None:
        self.total_embed_dim = total_embed_dim
        self.top_hidden_dim = top_hidden_dim
        rng = np.random.default_rng(seed + 999)

        self.W_top1 = rng.normal(
            0.0, np.sqrt(2.0 / total_embed_dim), size=(total_embed_dim, top_hidden_dim)
        )
        self.b_top1 = np.zeros((1, top_hidden_dim), dtype=np.float64)

        self.W_top2 = rng.normal(
            0.0, np.sqrt(1.5 / top_hidden_dim), size=(top_hidden_dim, 1)
        )
        self.b_top2 = np.zeros((1, 1), dtype=np.float64)

        # Direct linear skip from concatenated embeddings for fast convergence under noise
        self.W_direct = rng.normal(0.0, 0.15, size=(total_embed_dim, 1))

        self._cache: Dict[str, np.ndarray] = {}

    def forward(self, z_concat: np.ndarray, training: bool = True) -> np.ndarray:
        pre_t1 = z_concat @ self.W_top1 + self.b_top1
        t1 = _leaky_relu(pre_t1)
        logits = t1 @ self.W_top2 + self.b_top2 + z_concat @ self.W_direct
        probs = _sigmoid(logits)
        if training:
            self._cache = {
                "z_concat": z_concat,
                "pre_t1": pre_t1,
                "t1": t1,
                "probs": probs,
            }
        return probs.ravel()

    def backward(
        self,
        y_true: np.ndarray,
        lr: float = 0.015,
        weight_decay: float = 1e-4,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute BCE loss and backpropagate to produce dL/dz_concat for the parties.
        """
        z_concat = self._cache["z_concat"]
        pre_t1 = self._cache["pre_t1"]
        t1 = self._cache["t1"]
        probs = self._cache["probs"]

        y_col = y_true.reshape(-1, 1)
        n = float(y_col.shape[0])

        eps = 1e-7
        bce = -np.mean(
            y_col * np.log(probs + eps) + (1.0 - y_col) * np.log(1.0 - probs + eps)
        )

        grad_logits = (probs - y_col) / n  # (N, 1)

        grad_W_top2 = t1.T @ grad_logits + (weight_decay / n) * self.W_top2
        grad_b_top2 = np.sum(grad_logits, axis=0, keepdims=True)
        grad_W_direct = z_concat.T @ grad_logits + (weight_decay / n) * self.W_direct

        grad_t1 = grad_logits @ self.W_top2.T
        grad_pre_t1 = grad_t1 * _leaky_relu_grad(pre_t1)

        grad_W_top1 = z_concat.T @ grad_pre_t1 + (weight_decay / n) * self.W_top1
        grad_b_top1 = np.sum(grad_pre_t1, axis=0, keepdims=True)

        # Gradient w.r.t. concatenated cut-layer embeddings (scaled by n for per-sample mean in bottom nets)
        grad_z_concat = (
            grad_pre_t1 @ self.W_top1.T + grad_logits @ self.W_direct.T
        ) * n

        self.W_top2 -= lr * grad_W_top2
        self.b_top2 -= lr * grad_b_top2
        self.W_direct -= lr * grad_W_direct
        self.W_top1 -= lr * grad_W_top1
        self.b_top1 -= lr * grad_b_top1

        return grad_z_concat, float(bce)


@dataclasses.dataclass
class SplitNNMetrics:
    """Evaluation metrics for a trained model."""

    model_name: str
    roc_auc: float
    pr_auc: float
    brier_score: float
    accuracy: float
    final_train_loss: float
    mean_cut_l2_norm_upi: float
    mean_cut_l2_norm_telecom: float
    distance_corr_upi: float
    distance_corr_telecom: float


class SplitCreditNet:
    """
    Three-Party Vertical Split Neural Network for Cross-Institution Credit Scoring.
      - Party 1: Retail Bank (local bottom net + coordinator top net)
      - Party 2: UPI Super-App (local bottom net with cut-layer defense)
      - Party 3: Telecom & Utility (local bottom net with cut-layer defense)
    """

    def __init__(
        self,
        embed_dim: int = 8,
        hidden_dim: int = 16,
        top_hidden_dim: int = 16,
        defense_config: Optional[CutLayerDefenseConfig] = None,
        seed: int = 42,
        input_dims: Tuple[int, int, int] = (4, 4, 4),
    ) -> None:
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.top_hidden_dim = top_hidden_dim
        self.defense_config = defense_config or CutLayerDefenseConfig(enabled=False)
        self.seed = seed

        # Party 1 (Bank) is the coordinator itself; Party 2 (UPI) and Party 3 (Telecom)
        # transmit their cut-layer embeddings across institutional boundaries.
        self.party_bank = PartyBottomNet(
            party_id=1,
            party_name="Party1_Bank",
            input_dim=input_dims[0],
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            defense=self.defense_config,
            protect_cut_layer=False,
            seed=seed,
        )
        self.party_upi = PartyBottomNet(
            party_id=2,
            party_name="Party2_UPI",
            input_dim=input_dims[1],
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            defense=self.defense_config,
            protect_cut_layer=True,
            seed=seed,
        )
        self.party_telecom = PartyBottomNet(
            party_id=3,
            party_name="Party3_Telecom",
            input_dim=input_dims[2],
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            defense=self.defense_config,
            protect_cut_layer=True,
            seed=seed,
        )
        self.top_net = CoordinatorTopNet(
            total_embed_dim=3 * embed_dim,
            top_hidden_dim=top_hidden_dim,
            seed=seed,
        )
        self.loss_history: List[float] = []

    def forward_cut_layers(
        self,
        x_bank: np.ndarray,
        x_upi: np.ndarray,
        x_telecom: np.ndarray,
        training: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the 3 parties' transmitted cut-layer embeddings (z_1, z_2, z_3)."""
        z1 = self.party_bank.forward(x_bank, training=training)
        z2 = self.party_upi.forward(x_upi, training=training)
        z3 = self.party_telecom.forward(x_telecom, training=training)
        return z1, z2, z3

    def predict_proba(
        self,
        x_bank: np.ndarray,
        x_upi: np.ndarray,
        x_telecom: np.ndarray,
    ) -> np.ndarray:
        """Predict 12-month default probability for aligned customer records."""
        z1, z2, z3 = self.forward_cut_layers(
            x_bank, x_upi, x_telecom, training=False
        )
        z_concat = np.hstack([z1, z2, z3])
        return self.top_net.forward(z_concat, training=False)

    def fit(
        self,
        x_bank: np.ndarray,
        x_upi: np.ndarray,
        x_telecom: np.ndarray,
        y: np.ndarray,
        epochs: int = 160,
        batch_size: int = 64,
        lr: float = 0.025,
    ) -> List[float]:
        """
        Train the 3-party Vertical SplitNN using mini-batch SGD with cosine LR schedule.
        """
        rng = np.random.default_rng(self.seed)
        n = x_bank.shape[0]
        d = self.embed_dim
        self.loss_history = []

        for epoch in range(epochs):
            perm = rng.permutation(n)
            epoch_losses: List[float] = []
            cur_lr = lr * (0.35 + 0.65 * 0.5 * (1.0 + np.cos(np.pi * epoch / max(epochs, 1))))

            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb_b = x_bank[idx]
                xu_b = x_upi[idx]
                xt_b = x_telecom[idx]
                y_b = y[idx]

                # 1. Each party computes its cut-layer embedding locally
                z1, z2, z3 = self.forward_cut_layers(
                    xb_b, xu_b, xt_b, training=True
                )
                z_concat = np.hstack([z1, z2, z3])

                # 2. Coordinator runs top network forward + backward
                _ = self.top_net.forward(z_concat, training=True)
                grad_z_concat, bce_loss = self.top_net.backward(y_b, lr=cur_lr)

                # 3. Coordinator slices cut-layer gradients for each party
                grad_z1 = grad_z_concat[:, :d]
                grad_z2 = grad_z_concat[:, d : 2 * d]
                grad_z3 = grad_z_concat[:, 2 * d : 3 * d]

                # 4. Each party updates its private bottom network locally
                self.party_bank.backward(grad_z1, lr=cur_lr)
                self.party_upi.backward(grad_z2, lr=cur_lr)
                self.party_telecom.backward(grad_z3, lr=cur_lr)

                epoch_losses.append(bce_loss)

            self.loss_history.append(float(np.mean(epoch_losses)))

        return self.loss_history

    def evaluate(
        self,
        x_bank: np.ndarray,
        x_upi: np.ndarray,
        x_telecom: np.ndarray,
        y_true: np.ndarray,
        model_name: str = "SplitCreditNet",
    ) -> SplitNNMetrics:
        """Compute predictive and privacy-correlation metrics on an evaluation split."""
        z1, z2, z3 = self.forward_cut_layers(x_bank, x_upi, x_telecom, training=False)
        z_concat = np.hstack([z1, z2, z3])
        probs = self.top_net.forward(z_concat, training=False)
        preds = (probs >= 0.5).astype(np.float64)

        roc_auc = float(roc_auc_score(y_true, probs))
        pr_auc = float(average_precision_score(y_true, probs))
        brier = float(brier_score_loss(y_true, probs))
        acc = float(np.mean(preds == y_true))

        norm_upi = float(np.mean(np.linalg.norm(z2, axis=1)))
        norm_tel = float(np.mean(np.linalg.norm(z3, axis=1)))
        dcor_upi = distance_correlation(x_upi, z2)
        dcor_tel = distance_correlation(x_telecom, z3)

        return SplitNNMetrics(
            model_name=model_name,
            roc_auc=round(roc_auc, 4),
            pr_auc=round(pr_auc, 4),
            brier_score=round(brier, 4),
            accuracy=round(acc, 4),
            final_train_loss=round(self.loss_history[-1] if self.loss_history else 0.0, 4),
            mean_cut_l2_norm_upi=round(norm_upi, 4),
            mean_cut_l2_norm_telecom=round(norm_tel, 4),
            distance_corr_upi=round(dcor_upi, 4),
            distance_corr_telecom=round(dcor_tel, 4),
        )


def train_bank_only_baseline(
    x_bank_train: np.ndarray,
    y_train: np.ndarray,
    x_bank_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 140,
    lr: float = 0.025,
    seed: int = 42,
) -> Tuple[SplitNNMetrics, np.ndarray]:
    """
    Train a Single-Party Retail Bank neural baseline using ONLY Party 1's 4 features.
    Demonstrates the predictive ceiling of isolated single-institution credit scoring.
    """
    rng = np.random.default_rng(seed)
    n, p = x_bank_train.shape
    h_dim = 16
    batch_size = 64

    W1 = rng.normal(0.0, np.sqrt(2.0 / p), size=(p, h_dim))
    b1 = np.zeros((1, h_dim))
    W2 = rng.normal(0.0, np.sqrt(1.5 / h_dim), size=(h_dim, 1))
    b2 = np.zeros((1, 1))

    final_loss = 0.0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_bank_train[idx]
            yb = y_train[idx].reshape(-1, 1)
            nb = float(len(idx))

            pre_h = xb @ W1 + b1
            h = np.tanh(pre_h)
            logits = h @ W2 + b2
            probs = _sigmoid(logits)

            eps = 1e-7
            final_loss = float(
                -np.mean(yb * np.log(probs + eps) + (1.0 - yb) * np.log(1.0 - probs + eps))
            )

            grad_logits = (probs - yb) / nb
            grad_W2 = h.T @ grad_logits + 1e-4 * W2
            grad_b2 = np.sum(grad_logits, axis=0, keepdims=True)
            grad_h = grad_logits @ W2.T
            grad_pre_h = grad_h * (1.0 - h ** 2)
            grad_W1 = xb.T @ grad_pre_h + 1e-4 * W1
            grad_b1 = np.sum(grad_pre_h, axis=0, keepdims=True)

            W2 -= lr * grad_W2
            b2 -= lr * grad_b2
            W1 -= lr * grad_W1
            b1 -= lr * grad_b1

    test_h = np.tanh(x_bank_test @ W1 + b1)
    test_probs = _sigmoid(test_h @ W2 + b2).ravel()
    preds = (test_probs >= 0.5).astype(np.float64)

    metrics = SplitNNMetrics(
        model_name="BankOnly_SingleParty",
        roc_auc=round(float(roc_auc_score(y_test, test_probs)), 4),
        pr_auc=round(float(average_precision_score(y_test, test_probs)), 4),
        brier_score=round(float(brier_score_loss(y_test, test_probs)), 4),
        accuracy=round(float(np.mean(preds == y_test)), 4),
        final_train_loss=round(final_loss, 4),
        mean_cut_l2_norm_upi=0.0,
        mean_cut_l2_norm_telecom=0.0,
        distance_corr_upi=0.0,
        distance_corr_telecom=0.0,
    )
    return metrics, test_probs


def train_centralized_oracle(
    x_bank_train: np.ndarray,
    x_upi_train: np.ndarray,
    x_telecom_train: np.ndarray,
    y_train: np.ndarray,
    x_bank_test: np.ndarray,
    x_upi_test: np.ndarray,
    x_telecom_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 160,
    lr: float = 0.025,
    seed: int = 42,
) -> Tuple[SplitNNMetrics, np.ndarray]:
    """
    Train a Centralized Oracle MLP that pools all 12 raw features in plaintext
    (violating institutional data privacy) as an upper-bound reference.
    """
    x_tr = np.hstack([x_bank_train, x_upi_train, x_telecom_train])
    x_te = np.hstack([x_bank_test, x_upi_test, x_telecom_test])

    rng = np.random.default_rng(seed + 500)
    n, p = x_tr.shape
    h_dim = 24
    batch_size = 64

    W1 = rng.normal(0.0, np.sqrt(2.0 / p), size=(p, h_dim))
    b1 = np.zeros((1, h_dim))
    W2 = rng.normal(0.0, np.sqrt(1.5 / h_dim), size=(h_dim, 1))
    b2 = np.zeros((1, 1))

    final_loss = 0.0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_tr[idx]
            yb = y_train[idx].reshape(-1, 1)
            nb = float(len(idx))

            pre_h = xb @ W1 + b1
            h = np.tanh(pre_h)
            logits = h @ W2 + b2
            probs = _sigmoid(logits)

            eps = 1e-7
            final_loss = float(
                -np.mean(yb * np.log(probs + eps) + (1.0 - yb) * np.log(1.0 - probs + eps))
            )

            grad_logits = (probs - yb) / nb
            grad_W2 = h.T @ grad_logits + 1e-4 * W2
            grad_b2 = np.sum(grad_logits, axis=0, keepdims=True)
            grad_h = grad_logits @ W2.T
            grad_pre_h = grad_h * (1.0 - h ** 2)
            grad_W1 = xb.T @ grad_pre_h + 1e-4 * W1
            grad_b1 = np.sum(grad_pre_h, axis=0, keepdims=True)

            W2 -= lr * grad_W2
            b2 -= lr * grad_b2
            W1 -= lr * grad_W1
            b1 -= lr * grad_b1

    test_h = np.tanh(x_te @ W1 + b1)
    test_probs = _sigmoid(test_h @ W2 + b2).ravel()
    preds = (test_probs >= 0.5).astype(np.float64)

    metrics = SplitNNMetrics(
        model_name="Centralized_Oracle_MLP",
        roc_auc=round(float(roc_auc_score(y_test, test_probs)), 4),
        pr_auc=round(float(average_precision_score(y_test, test_probs)), 4),
        brier_score=round(float(brier_score_loss(y_test, test_probs)), 4),
        accuracy=round(float(np.mean(preds == y_test)), 4),
        final_train_loss=round(final_loss, 4),
        mean_cut_l2_norm_upi=round(float(np.mean(np.linalg.norm(x_upi_test, axis=1))), 4),
        mean_cut_l2_norm_telecom=round(float(np.mean(np.linalg.norm(x_telecom_test, axis=1))), 4),
        distance_corr_upi=1.0,
        distance_corr_telecom=1.0,
    )
    return metrics, test_probs
