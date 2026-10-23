"""Tests for splitbureau: PSI alignment, split-learning training, cut-layer leakage and API."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from splitbureau.api import app
from splitbureau.data import build_aligned_dataset_from_registries, generate_unaligned_registries
from splitbureau.psi import align_three_parties, align_two_parties
from splitbureau.splitnn import CutLayerDefenseConfig, SplitCreditNet, distance_correlation, train_bank_only_baseline


def test_two_party_psi_finds_exactly_the_common_ids() -> None:
    a = ["p1", "P2", "only_a", "p3"]
    b = ["x", " p3", "p2", "only_b", "P1"]
    res = align_two_parties(a, b)
    pairs = {(a[i].strip().upper(), b[j].strip().upper()) for i, j in zip(res.aligned_indices_a, res.aligned_indices_b)}
    assert pairs == {("P1", "P1"), ("P2", "P2"), ("P3", "P3")}
    assert res.exact_recovery_rate == 1.0


def test_three_party_psi_on_generated_registries() -> None:
    reg = generate_unaligned_registries(n_common=60, n_bank_extra=10, n_upi_extra=12, n_telecom_extra=8, seed=3)
    res = align_three_parties(reg.bank_ids, reg.upi_ids, reg.telecom_ids)
    assert res.intersection_size == 60
    for ib, iu, it in zip(res.aligned_indices_bank, res.aligned_indices_upi, res.aligned_indices_telecom):
        assert reg.bank_ids[ib] == reg.upi_ids[iu] == reg.telecom_ids[it]


def test_distance_correlation_properties() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    assert distance_correlation(x, x) == pytest.approx(1.0)
    assert distance_correlation(x, 2.0 * x + 1.0) == pytest.approx(1.0)
    assert distance_correlation(x, rng.normal(size=(200, 2))) < 0.25


@pytest.fixture(scope="module")
def dataset():
    reg = generate_unaligned_registries(n_common=500, seed=7)
    ds, _ = build_aligned_dataset_from_registries(reg, seed=7)
    return ds


def test_split_model_beats_bank_only(dataset) -> None:
    bank, _ = train_bank_only_baseline(dataset.x_bank_train, dataset.y_train, dataset.x_bank_test, dataset.y_test, epochs=60)
    model = SplitCreditNet(defense_config=CutLayerDefenseConfig(enabled=False), seed=7)
    model.fit(dataset.x_bank_train, dataset.x_upi_train, dataset.x_telecom_train, dataset.y_train, epochs=60)
    split = model.evaluate(dataset.x_bank_test, dataset.x_upi_test, dataset.x_telecom_test, dataset.y_test)
    assert split.roc_auc > bank.roc_auc + 0.1


def test_api_endpoints() -> None:
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    r = client.post("/psi/align", json={})
    assert r.status_code == 200 and len(r.json()["matched_customer_ids_order"]) == 3
    r = client.post("/score", json={})
    assert r.status_code == 200 and 0.0 <= r.json()["splitnn_default_probability"] <= 1.0
